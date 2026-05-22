# Video-to-3D-Reconstruction

Phone video to a semantically labeled, queryable 3D scene using Meta **VGGT-Omega** for geometry and **SAM 3** for open-vocabulary semantics.

Given a short handheld video of an indoor area, this system produces (a) a geometrically coherent 3D point cloud with the predicted camera trajectory, (b) per-point semantic labels aligned to the geometry by construction, and (c) an open-vocabulary text query interface that highlights matching 3D regions on demand.

**Geometry–semantics coherence is enforced by construction**: every semantic label is sampled from the exact pixel that produced its 3D point, so labels cannot drift relative to the underlying geometry.

## Output Examples

| Semantic point cloud | Poisson Disk Sampling | Open-vocabulary `chair` query |
| --- | --- | --- |
| ![Semantic point cloud](Images/Semantic_Point_Cloud.png) | ![Poisson mesh](Images/Poisson_disk_sampling.png) | ![Chair query](Images/chair_query.png) |
| Fused cloud colored by SAM 3 concept ID | Poisson Disk Sampling over the same points | Top-scoring matches for the prompt `chair` highlighted in red |

## Approach

```text
phone video
  → sharp frame sampling (fps cap + max-frames)
  → VGGT-Omega: per-frame camera, intrinsics, depth, confidence
  → SAM 3: per-frame text-prompted semantic masks
  → confidence filtering + voxel fusion (majority-vote labels)
  → RGB cloud + semantic cloud + camera trajectory viewer
  → optional SAM 3 text query over fused 3D points
```

The core idea is **pixel-aligned fusion**. VGGT-Omega predicts dense 3D geometry per frame; SAM 3 predicts text-prompted masks for the same frames; semantic IDs are sampled at the exact pixels used for 3D unprojection. Each fused point stores both its saved-frame and VGGT-grid pixel provenance, so open-vocabulary queries can re-prompt SAM 3 at the correct resolution after the fact.

## Install

```bash
git clone https://github.com/ayushmaankaria/Video-to-3D-Reconstruction.git
cd Video-to-3D-Reconstruction
pip install -r requirements-colab.txt
```

VGGT-Omega and SAM 3 are gated. Request access on Hugging Face, generate a read token, and authenticate (`hf auth login`) before running. An L4 or A100 is recommended.

## Run

```bash
python -m spatial_recon.cli run \
  --video "/content/drive/MyDrive/desk_video.mp4" \
  --out runs/desk \
  --checkpoint checkpoints/vggt-omega-1b-512/model.pt \
  --image-resolution 512 \
  --max-frames 48 --fps 2.0 \
  --concepts "desk,chair,monitor,keyboard,mouse,cup,wall,floor,cable,guitar,toothbrush,medicine,watch" \
  --conf-percentile 30 --sample-stride 2 --voxel-size 0.01
```

Open `runs/desk/exports/viewer.html`, or load the `.ply` outputs in MeshLab.

### Open-vocabulary query

```bash
python -m spatial_recon.cli query \
  --run runs/desk \
  --text "monitor" \
  --topk-percent 8 \
  --score-threshold 0.5
```

Writes `runs/desk/exports/query_monitor.ply` with the top-scoring matches painted red. The query path re-runs SAM 3 with the new prompt on the saved frames and samples each 3D point's score using its stored source frame and pixel coordinates.

## Outputs

```text
runs/desk/
  frames/                         # sampled frames
  frames_meta.json
  predictions.npz                 # VGGT-Omega depth, cameras, confidence, points
  semantics/
    label_maps/*.npy              # per-frame SAM 3 semantic label maps
    previews/*.png
    labels.json                   # concept ↔ id map
  exports/
    reconstruction_rgb.ply        # geometry colored by RGB
    reconstruction_semantic.ply   # geometry colored by concept
    reconstruction_semantic.glb   # point-cloud GLB (best-effort)
    viewer.html                   # interactive cloud + camera path
    semantic_legend.json
    fused_points.npz              # points, labels, confidence, source pixels
    query_<text>.ply              # written by the query subcommand
  REPORT.md                       # quality stats and design notes
```

## Reference Run

A handheld iPhone 16 Pro clip (4K, 60 fps) of a desk area, processed end-to-end on a single **NVIDIA L4** in **~8 minutes**:

| Metric | Value |
| --- | --- |
| Frames sampled | 48 |
| Fused points (after voxel filtering, 1 cm voxels) | ~90.5k |
| Semantic classes present | 12 |
| Mean / median VGGT-Omega confidence | 6.58 / 6.34 |
| Scene extent (bbox) | ~1.34 m × 1.59 m × 1.10 m |

Concept distribution after fusion: `unknown` 25.0k, `wall` 24.2k, `floor` 14.4k, `chair` 8.8k, `desk` 7.3k, `monitor` 6.1k, `cable` 1.5k, `guitar` 1.5k, `keyboard` 0.8k, `medicine` 0.5k, `mouse` 0.2k, `toothbrush` 0.2k.

## Design Choices

- **VGGT-Omega over plain VGGT and over COLMAP/SfM + MVS.** I first ran the baseline VGGT model and got usable but subpar geometry on phone-video desk scenes. VGGT-Omega (released May 18, 2026 by Meta) gave noticeably cleaner depth and more consistent cameras, and avoided the matching-failure modes of classical SfM on textureless office surfaces.
- **SAM 3 over Mask2Former / CLIPSeg.** Both alternatives were tried; SAM 3 produced cleaner instance-level masks for indoor concepts and, critically, supports the same open-vocabulary text prompts at reconstruction time *and* at query time, which is what makes the query subcommand possible without a second model.
- **Point cloud as the primary output.** Raw points preserve VGGT-Omega's geometry directly. Poisson meshes look fuller but can hallucinate curved shells around sparse or noisy regions, so the mesh is treated as a visualization rather than the source of truth.
- **Pixel-aligned semantics, not post-hoc projection.** SAM 3 labels are sampled at the same pixels used for 3D unprojection — geometry and semantics share a coordinate system by construction.
- **Voxel fusion with confidence-weighted majority vote.** Each voxel is represented by its highest-confidence point and labeled by the most common class inside it (ties broken by confidence). Per-point provenance for both the saved frame and the VGGT prediction grid is preserved for downstream querying.
- **Camera trajectory in the viewer.** Predicted camera poses are rendered alongside the cloud so a reviewer can see how the phone moved through the scene and judge geometric plausibility directly.

## What Works, What Doesn't

**Reconstructs well:** the desk, monitors, chair, and small objects on the desk (medicine bottles, keyboard, mouse) are clean and recognisable in 3D. Monitor screens, often a failure mode for dense methods, hold up reasonably.

**Struggles:**
- Reflective surfaces such as the side of the guitar are noisy and partially missing, as expected from view-dependent reflections.
- Floor and walls show mild deformation, likely a function of recording style — pure-rotation segments and limited parallax across textureless surfaces give VGGT-Omega less to anchor against.

## Limitations

- SAM 3 masks are prompt-dependent. The chosen concept list directly affects which objects receive labels in the fused cloud, and unlisted objects fall into `unknown`.
- Open-vocabulary querying re-runs SAM 3 per query. A more efficient version would cache dense per-frame features or distill them into per-point feature embeddings for instant queries.
- A 3D Gaussian Splatting visualisation path would likely produce a more photorealistic output than point-cloud rendering, at the cost of a heavier optimisation stage.

## Recording Tips

- Walk at normal or slightly slow pace; avoid sudden motion.
- Translate as well as rotate — pure rotation gives VGGT-Omega weaker geometry.
- Keep the scene static and avoid letting reflective screens dominate the frame.
- Capture overlapping views of object boundaries: chair legs, desk edges, monitors, walls, floor.

## References

- VGGT-Omega: https://github.com/facebookresearch/vggt-omega · https://huggingface.co/facebook/vggt-omega
- SAM 3: https://github.com/facebookresearch/sam3 · https://huggingface.co/facebook/sam3