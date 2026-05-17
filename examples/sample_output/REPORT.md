# Reconstruction Report

## Run summary

- Input video: `synthetic`
- Frames reconstructed: `12`
- VGGT checkpoint: `sample`
- Semantic model: `sample`
- Exported points: `61800`

## Quality stats

- Mean confidence: `1.0000`
- Median confidence: `1.0000`
- Bounding box min: `[-1.399999976158142, -1.0, 0.0]`
- Bounding box max: `[1.399999976158142, 1.0499999523162842, 1.399999976158142]`
- Semantic classes present: `5`

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
- Voxel fusion uses majority-vote semantic labels for stability while preserving the highest-confidence source pixel for traceability.
- The output includes both RGB geometry and semantic-color point clouds, plus source frame/pixel indices for future open-vocabulary querying.
