# Semantic VGGT Reconstruction

Video-to-3D reconstruction for the Humanoid Perception and Spatial AI internship challenge.

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
    viewer.html                   # interactive point-cloud viewer
    semantic_legend.json
  REPORT.md                       # design choices and run summary
```

A lightweight synthetic example is included at `examples/sample_output/`. It exists so the export/report path can be checked without downloading VGGT weights.

## Colab setup

Colab with a T4/A100/L4 GPU is recommended. My M4 MacBook Air can run the preprocessing and viewer pieces, but VGGT-1B is much more practical on CUDA.

```bash
git clone https://github.com/YOUR_USERNAME/semantic-vggt-reconstruction.git
cd semantic-vggt-reconstruction
pip install -r requirements.txt
```

If you want the commercial-use VGGT checkpoint, request access from the model page and use:

```bash
python -m spatial_recon.cli run \
  --video data/desk_video.mp4 \
  --out runs/desk \
  --checkpoint facebook/VGGT-1B-Commercial \
  --max-frames 24
```

For normal research/demo use:

```bash
python -m spatial_recon.cli run \
  --video data/desk_video.mp4 \
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

## Useful options

```bash
# Use an existing frame folder instead of a video
python -m spatial_recon.cli run --images-dir data/desk_frames --out runs/desk_frames

# More detailed point cloud, slower viewer
python -m spatial_recon.cli run --video data/desk_video.mp4 --out runs/dense --sample-stride 1 --max-points 900000

# Geometry only
python -m spatial_recon.cli run --video data/desk_video.mp4 --out runs/no_semantics --no-semantics

# Use VGGT's point-map branch instead of depth unprojection
python -m spatial_recon.cli run --video data/desk_video.mp4 --out runs/pointmap --use-point-map
```

## References

- Meta VGGT repository: https://github.com/facebookresearch/vggt
- VGGT project page: https://vgg-t.github.io/
- Mask2Former ADE20K model: https://huggingface.co/facebook/mask2former-swin-large-ade-semantic

