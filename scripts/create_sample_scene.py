from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_recon.export import write_cloud_npz, write_glb, write_html_viewer, write_legend, write_ply
from spatial_recon.fusion import FusedPointCloud
from spatial_recon.report import write_report
from spatial_recon.utils import ensure_dir, semantic_color


def box(center, size, n, label_id, color):
    rng = np.random.default_rng(label_id)
    pts = rng.uniform(-0.5, 0.5, size=(n, 3)) * np.asarray(size) + np.asarray(center)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (n, 1))
    labels = np.full(n, label_id, dtype=np.int32)
    return pts.astype(np.float32), colors, labels


def main() -> None:
    out = ensure_dir(Path("examples/sample_output/exports"))
    floor_x, floor_y = np.meshgrid(np.linspace(-1.4, 1.4, 180), np.linspace(-1.0, 1.0, 130))
    floor = np.stack([floor_x.ravel(), floor_y.ravel(), np.zeros(floor_x.size)], axis=1).astype(np.float32)
    wall = np.stack([floor_x.ravel(), np.full(floor_x.size, 1.05), np.linspace(0, 1.4, floor_x.size)], axis=1).astype(np.float32)

    chunks = [
        (floor, np.tile([170, 160, 145], (len(floor), 1)).astype(np.uint8), np.full(len(floor), 1, dtype=np.int32)),
        (wall, np.tile([210, 215, 220], (len(wall), 1)).astype(np.uint8), np.full(len(wall), 2, dtype=np.int32)),
        box((0.0, 0.0, 0.55), (1.1, 0.55, 0.12), 9000, 3, [120, 88, 55]),
        box((-0.55, -0.35, 0.32), (0.22, 0.22, 0.64), 3500, 4, [45, 48, 56]),
        box((0.3, -0.15, 0.72), (0.32, 0.22, 0.06), 2500, 5, [30, 30, 30]),
    ]
    points = np.concatenate([c[0] for c in chunks])
    rgb = np.concatenate([c[1] for c in chunks])
    labels = np.concatenate([c[2] for c in chunks])
    semantic_colors = np.asarray([semantic_color(int(x)) for x in labels], dtype=np.uint8)
    cloud = FusedPointCloud(
        points=points,
        colors_rgb=rgb,
        semantic_ids=labels,
        semantic_colors=semantic_colors,
        confidence=np.ones(len(points), dtype=np.float32),
        labels={0: "unknown", 1: "floor", 2: "wall", 3: "desk", 4: "chair", 5: "monitor"},
        source_frame=np.zeros(len(points), dtype=np.int32),
        source_y=np.zeros(len(points), dtype=np.int32),
        source_x=np.zeros(len(points), dtype=np.int32),
        source_y_vggt=np.zeros(len(points), dtype=np.int32),
        source_x_vggt=np.zeros(len(points), dtype=np.int32),
    )
    write_ply(out / "reconstruction_rgb.ply", cloud)
    write_ply(out / "reconstruction_semantic.ply", cloud, semantic_colors=True)
    write_glb(out / "reconstruction_semantic.glb", cloud)
    write_html_viewer(out / "viewer.html", cloud)
    write_legend(out / "semantic_legend.json", cloud)
    write_cloud_npz(out / "fused_points.npz", cloud)
    write_report(Path("examples/sample_output/REPORT.md"), cloud, 12, "synthetic", "sample", "sample")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
