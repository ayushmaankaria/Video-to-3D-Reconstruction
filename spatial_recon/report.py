from __future__ import annotations

from pathlib import Path

import numpy as np

from .fusion import FusedPointCloud
from .utils import ensure_dir


def write_report(
    path: str | Path,
    cloud: FusedPointCloud,
    frame_count: int,
    video_path: str | None,
    checkpoint: str,
    semantic_model: str | None,
    command: str | None = None,
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    used = []
    for label_id in sorted(set(int(x) for x in cloud.semantic_ids.tolist())):
        name = cloud.labels.get(label_id, "unknown")
        count = int(np.sum(cloud.semantic_ids == label_id))
        if count:
            used.append((name, count))
    used.sort(key=lambda item: item[1], reverse=True)

    lines = [
        "# Reconstruction Report",
        "",
        "## Run summary",
        "",
        f"- Input video: `{video_path or 'image folder'}`",
        f"- Frames reconstructed: `{frame_count}`",
        f"- VGGT checkpoint: `{checkpoint}`",
        f"- Semantic model: `{semantic_model or 'disabled'}`",
        f"- Exported points: `{len(cloud.points)}`",
        "",
        "## Quality stats",
        "",
        f"- Mean confidence: `{float(np.mean(cloud.confidence)):.4f}`",
        f"- Median confidence: `{float(np.median(cloud.confidence)):.4f}`",
        f"- Bounding box min: `{np.min(cloud.points, axis=0).round(4).tolist()}`",
        f"- Bounding box max: `{np.max(cloud.points, axis=0).round(4).tolist()}`",
        f"- Semantic classes present: `{len(set(int(x) for x in cloud.semantic_ids.tolist()))}`",
        "",
        "## Semantic inventory",
        "",
    ]
    if used:
        lines.extend([f"- `{name}`: {count} points" for name, count in used[:25]])
    else:
        lines.append("- No semantic labels were projected.")

    lines.extend(
        [
            "",
            "## Design choices",
            "",
            "- Frames are selected with a hybrid coverage/sharpness sampler so the video contributes stable views instead of near-duplicates.",
            "- VGGT predicts camera poses, depth, point maps, and confidence in one feed-forward pass.",
            "- 2D semantic masks are projected at the same pixels used for 3D unprojection, so labels remain locked to the reconstructed geometry.",
            "- Confidence percentile filtering, radius trimming, and voxel fusion reduce floating outliers while keeping object-level structure visible.",
            "- Voxel fusion uses majority-vote semantic labels for stability while preserving the highest-confidence source pixel for traceability.",
            "- The output includes both RGB geometry and semantic-color point clouds, plus source frame/pixel indices for future open-vocabulary querying.",
        ]
    )
    if command:
        lines.extend(["", "## Reproduce", "", f"```bash\n{command}\n```"])

    path.write_text("\n".join(lines) + "\n")
    return path
