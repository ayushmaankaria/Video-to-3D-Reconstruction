from __future__ import annotations

from pathlib import Path

import numpy as np

from .fusion import FusedPointCloud
from .utils import ensure_dir, semantic_color, write_json


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
            "color_rgb": list(semantic_color(label_id)),
            "points": int(np.sum(cloud.semantic_ids == label_id)),
        }
        for label_id in used_ids
    }
    write_json(path, payload)
    return Path(path)


def write_cloud_npz(path: str | Path, cloud: FusedPointCloud) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        points=cloud.points,
        colors_rgb=cloud.colors_rgb,
        semantic_ids=cloud.semantic_ids,
        semantic_colors=cloud.semantic_colors,
        confidence=cloud.confidence,
        source_frame=cloud.source_frame,
        source_y=cloud.source_y,
        source_x=cloud.source_x,
        source_y_vggt=cloud.source_y_vggt,
        source_x_vggt=cloud.source_x_vggt,
        label_ids=np.asarray(sorted(cloud.labels), dtype=np.int32),
        label_names=np.asarray([cloud.labels[k] for k in sorted(cloud.labels)], dtype=object),
    )
    return path


def _camera_centers(extrinsic: np.ndarray | None) -> np.ndarray | None:
    if extrinsic is None:
        return None
    extrinsic = np.asarray(extrinsic)
    if extrinsic.ndim == 4:
        extrinsic = extrinsic.squeeze(0)
    if extrinsic.ndim != 3 or extrinsic.shape[1:] != (3, 4):
        return None
    rotation = extrinsic[:, :, :3]
    translation = extrinsic[:, :, 3]
    return -np.einsum("nij,nj->ni", np.transpose(rotation, (0, 2, 1)), translation)


def write_html_viewer(
    path: str | Path,
    cloud: FusedPointCloud,
    max_points: int = 80_000,
    extrinsic: np.ndarray | None = None,
) -> Path:
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
        traces = [
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker={"size": 1.5, "color": colors},
                text=names,
                hovertemplate="%{text}<extra></extra>",
                name="semantic points",
            )
        ]
        cameras = _camera_centers(extrinsic)
        if cameras is not None and len(cameras):
            traces.append(
                go.Scatter3d(
                    x=cameras[:, 0],
                    y=cameras[:, 1],
                    z=cameras[:, 2],
                    mode="lines+markers",
                    marker={"size": 4, "color": "black", "symbol": "diamond"},
                    line={"width": 4, "color": "black"},
                    name="camera trajectory",
                    hovertemplate="camera %{text}<extra></extra>",
                    text=[str(i) for i in range(len(cameras))],
                )
            )
        fig = go.Figure(data=traces)
        fig.update_layout(
            title="Semantic VGGT Reconstruction",
            scene={"aspectmode": "data"},
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
            legend={"orientation": "h"},
        )
        fig.write_html(path, include_plotlyjs="cdn")
    except ImportError:
        path.write_text(
            "<html><body><h1>Semantic VGGT Reconstruction</h1>"
            "<p>Install plotly to generate the interactive viewer.</p></body></html>\n"
        )
    return path
