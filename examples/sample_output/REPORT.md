# Reconstruction Report

## Run summary

- Input video: `synthetic`
- Frames reconstructed: `12`
- VGGT checkpoint: `sample`
- Semantic model: `sample`
- Exported points: `61800`

## Semantic inventory

- `floor`: 23400 points
- `wall`: 23400 points
- `desk`: 9000 points
- `chair`: 3500 points
- `monitor`: 2500 points

## Design choices

- Frames are selected with a hybrid coverage/sharpness sampler so the video contributes stable views instead of near-duplicates.
- VGGT predicts camera poses, depth, point maps, and confidence in one feed-forward pass.
- 2D semantic masks are projected at the same pixels used for 3D unprojection, so labels remain locked to the reconstructed geometry.
- Confidence percentile filtering, radius trimming, and voxel fusion reduce floating outliers while keeping object-level structure visible.
- The output includes both RGB geometry and semantic-color point clouds for easy inspection.
