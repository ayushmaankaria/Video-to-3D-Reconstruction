from __future__ import annotations

from pathlib import Path

import numpy as np

from .fusion import FusedPointCloud
from .utils import ensure_dir


def write_report(
    path: str | Path,
    cloud: FusedPointCloud,
    *,
    frame_count: int,
    video_path: str | None,
    checkpoint: str,
    semantic_model: str | None = "sam3",
    command: str | None = None,
) -> Path:
    path = Path(path); ensure_dir(path.parent)

    used: list[tuple[str, int]] = []
    if len(cloud.semantic_ids):
        for lid in sorted(set(int(x) for x in cloud.semantic_ids.tolist())):
            name = cloud.labels.get(lid, "unknown")
            count = int(np.sum(cloud.semantic_ids == lid))
            if count:
                used.append((name, count))
    used.sort(key=lambda kv: kv[1], reverse=True)

    if len(cloud.points):
        bb_min = np.min(cloud.points, axis=0).round(4).tolist()
        bb_max = np.max(cloud.points, axis=0).round(4).tolist()
        mean_c = float(np.mean(cloud.confidence))
        med_c  = float(np.median(cloud.confidence))
    else:
        bb_min = bb_max = [0, 0, 0]; mean_c = med_c = 0.0

    lines = [
        "# Reconstruction Report",
        "",
        "## Run summary",
        "",
        f"- Input video: `{video_path or 'image folder'}`",
        f"- Frames reconstructed: `{frame_count}`",
        f"- VGGT-Omega checkpoint: `{checkpoint}`",
        f"- Semantic model: `{semantic_model or 'disabled'}`",
        f"- Exported points: `{len(cloud.points)}`",
        "",
        "## Quality stats",
        "",
        f"- Mean confidence: `{mean_c:.4f}`",
        f"- Median confidence: `{med_c:.4f}`",
        f"- Bounding box min: `{bb_min}`",
        f"- Bounding box max: `{bb_max}`",
        f"- Semantic classes present: `{len(used)}`",
        "",
        "## Semantic inventory",
        "",
    ]
    lines += ([f"- `{name}`: {count} points" for name, count in used[:25]]
              if used else ["- No semantic labels were projected."])
    lines += [
        "",
        "## Design choices",
        "",
        "- Frames are sampled at a target fps and then capped at --max-frames so the video contributes stable views instead of near-duplicates.",
        "- VGGT-Omega predicts camera poses, intrinsics, depth and per-pixel confidence in a single forward pass.",
        "- SAM 3 is prompted with each concept separately per frame; the highest-scoring mask wins per pixel.",
        "- Labels are sampled at the same pixels used for 3D unprojection, so semantics stay locked to geometry.",
        "- Voxel fusion uses the highest-confidence point per voxel as the representative and majority vote (confidence-broken ties) for the semantic id.",
        "- Both VGGT-grid and saved-frame pixel provenance are kept per point so open-vocabulary queries can sample SAM 3 masks at the right resolution.",
    ]
    if command:
        lines += ["", "## Reproduce", "", f"```bash\n{command}\n```"]

    path.write_text("\n".join(lines) + "\n")
    return path