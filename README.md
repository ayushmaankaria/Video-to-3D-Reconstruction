# Semantic VGGT Reconstruction

This system takes a short phone video of a small indoor scene, samples useful frames, reconstructs geometry with Meta's VGGT, projects semantic masks into 3D, and exports RGB and semantic point clouds for inspection.

## Why this approach

VGGT is a strong fit for the challenge because it predicts camera poses, intrinsics, depth maps, point maps, and confidence directly from multiple views. I use those dense per-pixel outputs as the geometric backbone, then attach semantic labels at the same pixels before fusion. The result is simple: every semantic point came from a real VGGT 3D point, so labels stay aligned with geometry instead of being pasted on afterward.

Pipeline:

1. Extract sharp, temporally spread frames from the video.
2. Run VGGT on the selected frames.
3. Segment each frame with Mask2Former ADE20K semantics.
4. Project semantic labels onto VGGT's 3D points.
5. Filter low-confidence/outlier points and voxel-fuse duplicates.
6. Export RGB geometry, semantic geometry, a GLB, an HTML viewer, a label legend, and a run report.

```text
phone video
  -> sharp frame sampling
  -> VGGT camera/depth/point prediction
  -> Mask2Former 2D semantic segmentation
  -> pixel-aligned 3D semantic fusion
  -> RGB point cloud + semantic point cloud + viewer + report
```

## Outputs

Each run creates:

```text
runs/desk/
  frames/                         # sampled input frames
  predictions_vggt.npz            # VGGT depth, cameras, point maps, confidence
  semantics/
    label_maps/*.npy              # per-frame semantic labels
    previews/*.png                # color previews of 2D semantics
    labels.json
  exports/
    reconstruction_rgb.ply        # geometry colored by input RGB
    reconstruction_semantic.ply   # geometry colored by semantic class
    reconstruction_semantic.glb   # quick 3D viewer artifact
    viewer.html                   # interactive point-cloud viewer with camera path
    semantic_legend.json
    fused_points.npz              # reusable points, colors, labels, confidence, source pixels
  REPORT.md                       # design choices and run summary
```

`fused_points.npz` stores both original saved-frame pixel provenance (`source_y`, `source_x`) and VGGT-resolution pixel provenance (`source_y_vggt`, `source_x_vggt`). That distinction matters for future CLIP querying because image patches should be sampled from the saved frame resolution, not blindly from VGGT's internal tensor resolution.

A lightweight synthetic example is included at `examples/sample_output/`. It exists so the export/report path can be checked without downloading VGGT weights.

## Colab setup

Colab with a T4/A100/L4 GPU is recommended. My M4 MacBook Air can run the preprocessing and viewer pieces, but VGGT-1B is much more practical on CUDA.

```bash
git clone https://github.com/ayushmaankaria/Video-to-3D-Reconstruction.git
cd Video-to-3D-Reconstruction
pip install -r requirements-colab.txt
```

Use `requirements-colab.txt` in Colab because Colab already includes CUDA-enabled PyTorch. Installing the full local `requirements.txt` can waste a lot of time by trying to resolve or reinstall large Torch wheels.

The input video is intentionally not committed to GitHub because phone videos are usually too large for a normal repository. In Colab, keep the video in Google Drive or upload it during the session, then point `--video` at that local Colab path.

Example with Google Drive:

```bash
python -m spatial_recon.cli run \
  --video "/content/drive/MyDrive/desk_video.mp4" \
  --out runs/desk \
  --checkpoint facebook/VGGT-1B \
  --max-frames 24 \
  --conf-percentile 35 \
  --sample-stride 2 \
  --voxel-size 0.015
```

If you want the commercial-use VGGT checkpoint, request access from the model page and use:

```bash
python -m spatial_recon.cli run \
  --video "/content/drive/MyDrive/desk_video.mp4" \
  --out runs/desk \
  --checkpoint facebook/VGGT-1B-Commercial \
  --max-frames 24
```

For normal research/demo use:

```bash
python -m spatial_recon.cli run \
  --video "/content/drive/MyDrive/desk_video.mp4" \
  --out runs/desk \
  --checkpoint facebook/VGGT-1B \
  --max-frames 24 \
  --conf-percentile 35 \
  --sample-stride 2 \
  --voxel-size 0.015
```

Open `runs/desk/exports/viewer.html` to inspect the semantic reconstruction. The `.ply` files can also be opened in MeshLab, CloudCompare, Blender, or Open3D tooling.

## Gradio demo

For a simple upload-and-view interface:

```bash
python -m spatial_recon.app
```

This still runs the same pipeline and writes outputs under `runs/gradio_desk/`.

## Synthetic sanity check

This does not use VGGT. It only verifies exporters and report generation.

```bash
python scripts/create_sample_scene.py
```

Then open:

```text
examples/sample_output/exports/viewer.html
```

## Recording tips

For a desk or small room:

- Record 10-25 seconds at normal walking speed.
- Move laterally as well as rotating; pure rotation gives weaker geometry.
- Keep the scene static and avoid reflective screens dominating the frame.
- Capture overlapping views of object boundaries: chair legs, desk edges, monitor, walls, floor.
- Use 16-32 frames for a first run; increase if the scene is sparse or large.

## Design tradeoffs

- Point cloud over mesh: VGGT gives dense geometry quickly, and a point cloud keeps the submission robust without a fragile meshing step. A mesh or Gaussian splat can be added from the exported COLMAP-style cameras later.
- Mask2Former semantics: ADE20K has useful indoor classes such as wall, floor, table, chair, cabinet, shelf, desk, and monitor. It is zero-shot enough for a desk scene and easy to run in Colab.
- Pixel-aligned semantic projection: semantics are sampled at each unprojected pixel, which prioritizes 3D/2D alignment over class-level smoothness.
- Confidence filtering: the default removes the lowest 35 percent of VGGT confidence values, trims far outliers, and voxel-fuses nearby points. This usually makes phone-video reconstructions cleaner while preserving scene layout.
- Majority-vote voxel labels: nearby points are merged spatially, and each voxel receives the most common semantic class inside it. This is less noisy than simply taking the highest-confidence point label.
- Camera trajectory: the HTML viewer overlays predicted camera positions, making it easier to understand how the phone moved and where the reconstruction came from.

## Limitations and next steps

- Raw point clouds are the faithful output. Poisson meshes can look fuller, but they may hallucinate curved shells around sparse or noisy phone-video geometry.
- Desk scenes are challenging because reflective monitors, thin chair legs, motion blur, and textureless flat surfaces are difficult for dense reconstruction.
- The next major upgrade is open-vocabulary 3D querying: use the exported source frame/pixel indices, sample CLIP patch features, and let users type queries like `blue mug` or `notebook` to highlight matching 3D points.
- Another strong upgrade is exporting VGGT cameras to a Gaussian Splatting pipeline for a more photorealistic visual representation.

## Useful options

```bash
# Use an existing frame folder instead of a video
python -m spatial_recon.cli run --images-dir data/desk_frames --out runs/desk_frames

# More detailed point cloud, slower viewer
python -m spatial_recon.cli run --video "/content/drive/MyDrive/desk_video.mp4" --out runs/dense --sample-stride 1 --max-points 900000

# Geometry only
python -m spatial_recon.cli run --video "/content/drive/MyDrive/desk_video.mp4" --out runs/no_semantics --no-semantics

# Use VGGT's point-map branch instead of depth unprojection
python -m spatial_recon.cli run --video "/content/drive/MyDrive/desk_video.mp4" --out runs/pointmap --use-point-map

# Run lightweight deterministic fusion tests
python -m unittest tests/test_fusion.py
```

## References

- Meta VGGT repository: https://github.com/facebookresearch/vggt
- VGGT project page: https://vgg-t.github.io/
- Mask2Former ADE20K model: https://huggingface.co/facebook/mask2former-swin-large-ade-semantic
