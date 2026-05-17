from __future__ import annotations

from pathlib import Path

import numpy as np

from .fusion import FusedPointCloud
from .utils import ensure_dir, write_json


def write_ply(path: str | Path, cloud: FusedPointCloud, semantic_colors: bool = False) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    colors = cloud.semantic_colors if semantic_colors else cloud.colors_rgb
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(cloud.points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int semantic_id\n")
        f.write("property float confidence\n")
        f.write("end_header\n")
        for point, color, label_id, conf in zip(cloud.points, colors, cloud.semantic_ids, cloud.confidence):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {int(label_id)} {float(conf):.6f}\n"
            )
    return path


def write_glb(path: str | Path, cloud: FusedPointCloud, semantic_colors: bool = True) -> Path | None:
    try:
        import trimesh
    except ImportError:
        return None

    path = Path(path)
    ensure_dir(path.parent)
    colors = cloud.semantic_colors if semantic_colors else cloud.colors_rgb
    rgba = np.concatenate([colors, np.full((len(colors), 1), 255, dtype=np.uint8)], axis=1)
    geom = trimesh.points.PointCloud(vertices=cloud.points, colors=rgba)
    geom.export(path)
    return path


def write_legend(path: str | Path, cloud: FusedPointCloud) -> Path:
    used_ids = sorted(set(int(x) for x in cloud.semantic_ids.tolist()))
    payload = {
        str(label_id): {
            "name": cloud.labels.get(label_id, "unknown"),
            "color_rgb": cloud.semantic_colors[np.where(cloud.semantic_ids == label_id)[0][0]].tolist()
            if np.any(cloud.semantic_ids == label_id)
            else [160, 160, 160],
            "points": int(np.sum(cloud.semantic_ids == label_id)),
        }
        for label_id in used_ids
    }
    write_json(path, payload)
    return Path(path)


def write_html_viewer(path: str | Path, cloud: FusedPointCloud, max_points: int = 80_000) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    n = len(cloud.points)
    if n > max_points:
        rng = np.random.default_rng(11)
        idx = rng.choice(n, size=max_points, replace=False)
    else:
        idx = np.arange(n)

    try:
        import plotly.graph_objects as go

        pts = cloud.points[idx]
        colors = [f"rgb({r},{g},{b})" for r, g, b in cloud.semantic_colors[idx]]
        names = [cloud.labels.get(int(label_id), "unknown") for label_id in cloud.semantic_ids[idx]]
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    marker={"size": 1.5, "color": colors},
                    text=names,
                    hovertemplate="%{text}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="Semantic VGGT Reconstruction",
            scene={"aspectmode": "data"},
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
        )
        fig.write_html(path, include_plotlyjs="cdn")
    except ImportError:
        path.write_text(
            "<html><body><h1>Semantic VGGT Reconstruction</h1>"
            "<p>Install plotly to generate the interactive viewer.</p></body></html>\n"
        )
    return path

