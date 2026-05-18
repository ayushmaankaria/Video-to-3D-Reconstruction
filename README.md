# Semantic VGGT Reconstruction

Phone video to a semantically labeled, searchable 3D scene using Meta VGGT, Mask2Former, and CLIPSeg.

<!-- Add final result GIF here before submission. Suggested path: docs/desk_reconstruction.gif -->

```bash
git clone https://github.com/ayushmaankaria/Video-to-3D-Reconstruction.git
cd Video-to-3D-Reconstruction
pip install -r requirements-colab.txt
```

```bash
python -m spatial_recon.cli run --video "/content/drive/MyDrive/desk_video.mp4" --out runs/desk --checkpoint facebook/VGGT-1B
python -m spatial_recon.cli query --run runs/desk --text "chair" --topk-percent 8
```

Open `runs/desk/exports/viewer.html`, or load the `.ply` outputs in MeshLab/CloudCompare.

## What It Does

This project reconstructs a small indoor area from a phone video, assigns semantic labels to the reconstructed 3D points, and lets users run open-vocabulary text queries such as `chair`, `screen`, `keyboard`, or `blue mug` to highlight matching 3D regions.

The core idea is pixel-aligned fusion: VGGT predicts dense 3D geometry for each frame, Mask2Former predicts 2D semantic masks for each frame, and the system transfers labels through the same pixels used to create the 3D points. That keeps semantic predictions tied to the underlying geometry instead of being pasted on afterward.

```text
phone video
  -> sharp frame sampling
  -> VGGT camera/depth/point prediction
  -> Mask2Former 2D semantic segmentation
  -> confidence filtering + voxel fusion
  -> RGB point cloud + semantic point cloud + camera trajectory viewer
  -> optional CLIPSeg text query over 3D points
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
    viewer.html                   # interactive point cloud with camera path
    semantic_legend.json
    fused_points.npz              # points, labels, confidence, source pixels
  REPORT.md                       # run summary, quality stats, design notes
```

`fused_points.npz` stores both original saved-frame pixel provenance (`source_y`, `source_x`) and VGGT-resolution pixel provenance (`source_y_vggt`, `source_x_vggt`). That distinction matters for CLIP querying because image patches/heatmaps should be sampled from the saved frame resolution, not blindly from VGGT's internal tensor resolution.

## Colab Workflow

Use an L4 or A100 GPU. The input video is intentionally not committed to GitHub because phone videos are usually too large for a normal repository. Put the video in Google Drive, then set the `--video` path.

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

For a denser final pass:

```bash
python -m spatial_recon.cli run \
  --video "/content/drive/MyDrive/desk_video.mp4" \
  --out runs/desk_hq \
  --checkpoint facebook/VGGT-1B \
  --max-frames 32 \
  --conf-percentile 45 \
  --sample-stride 1 \
  --voxel-size 0.008 \
  --max-points 900000 \
  --overwrite-frames \
  --overwrite-predictions \
  --overwrite-semantics
```

If you want the commercial-use VGGT checkpoint, request access from the model page and replace the checkpoint with `facebook/VGGT-1B-Commercial`.

## Open-Vocabulary Querying

After reconstruction:

```bash
python -m spatial_recon.cli query \
  --run runs/desk \
  --text "screen" \
  --topk-percent 8
```

This writes `runs/desk/exports/query_screen.ply`, where the top-scoring query matches are painted red and all other points are muted gray. The query path uses CLIPSeg heatmaps on the original saved frames, then samples those heatmaps using each 3D point's source frame and pixel.

## Design Choices

- VGGT over classical SfM: VGGT directly predicts camera pose, intrinsics, depth, point maps, and confidence from multiple images, making the pipeline compact and robust for a short internship challenge.
- Point cloud as the primary output: raw points preserve VGGT's geometry directly. Poisson meshes can look fuller, but they can also hallucinate curved shells around sparse/noisy phone-video geometry.
- Pixel-aligned semantics: labels are sampled at the same pixels used for 3D unprojection, keeping semantics aligned with geometry.
- Majority-vote voxel labels: nearby points are merged spatially, and each voxel receives the most common semantic class inside it.
- Camera trajectory: the HTML viewer overlays predicted camera positions so reviewers can see how the phone moved through the scene.

## Limitations

- Desk scenes are challenging because reflective displays, thin chair legs, motion blur, and textureless flat surfaces are difficult for dense reconstruction.
- Mask2Former ADE20K labels are useful for broad indoor categories such as walls, floors, tables, chairs, shelves, cabinets, and screens/displays, but they are not object-instance labels.
- Open-vocabulary querying currently uses CLIPSeg heatmaps sampled at source pixels. A deeper version would cache dense CLIP/MaskCLIP features per frame and support faster repeated queries.
- A Gaussian Splatting visualization path would likely produce a more photorealistic result than point-cloud rendering.

## Recording Tips

- Record 10-25 seconds at normal walking speed.
- Move laterally as well as rotating; pure rotation gives weaker geometry.
- Keep the scene static and avoid reflective screens dominating the frame.
- Capture overlapping views of object boundaries: chair legs, desk edges, screen, walls, floor.
- Use 16-32 frames for a first run; increase only if the scene is sparse or large.

## Local Check

```bash
python -m unittest tests/test_fusion.py
```

## References

- Meta VGGT repository: https://github.com/facebookresearch/vggt
- VGGT project page: https://vgg-t.github.io/
- Mask2Former ADE20K model: https://huggingface.co/facebook/mask2former-swin-large-ade-semantic
- CLIPSeg model: https://huggingface.co/CIDAS/clipseg-rd64-refined
