from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import ensure_dir, semantic_color, write_json


def write_ply(path: str | Path, cloud: dict) -> Path:
    """Binary little-endian PLY with vertex colors.
    cloud = {"points": [N,3] float32, "colors": [N,3] uint8}.
    """
    path = Path(path); ensure_dir(path.parent)
    pts  = np.ascontiguousarray(cloud["points"], dtype=np.float32)
    cols = np.ascontiguousarray(cloud["colors"], dtype=np.uint8)
    if pts.shape[0] != cols.shape[0]:
        raise ValueError(f"points/colors length mismatch: {pts.shape} vs {cols.shape}")
    n = pts.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("r","u1"),("g","u1"),("b","u1")])
    rec = np.empty(n, dtype=dtype)
    rec["x"], rec["y"], rec["z"] = pts[:,0], pts[:,1], pts[:,2]
    rec["r"], rec["g"], rec["b"] = cols[:,0], cols[:,1], cols[:,2]
    with open(path, "wb") as f:
        f.write(header); f.write(rec.tobytes())
    return path


def write_glb(path: str | Path, *, points: np.ndarray, colors: np.ndarray) -> Path:
    """GLB with a single POINTS primitive. Requires ``trimesh``."""
    import trimesh
    path = Path(path); ensure_dir(path.parent)
    pts  = np.ascontiguousarray(points, dtype=np.float32)
    cols = np.ascontiguousarray(colors, dtype=np.uint8)
    if cols.shape[1] == 3:
        alpha = np.full((cols.shape[0], 1), 255, dtype=np.uint8)
        cols = np.concatenate([cols, alpha], axis=1)
    pc = trimesh.PointCloud(vertices=pts, colors=cols)
    trimesh.Scene(pc).export(path, file_type="glb")
    return path


def semantic_colors_for_cloud(cloud) -> np.ndarray:
    """[N, 3] uint8 colors derived from semantic_ids."""
    out = np.zeros((len(cloud.points), 3), dtype=np.uint8)
    for lid in np.unique(cloud.semantic_ids):
        out[cloud.semantic_ids == int(lid)] = semantic_color(int(lid))
    return out


def write_legend(path: str | Path, cloud) -> Path:
    legend = {}
    for lid in sorted(set(int(x) for x in cloud.semantic_ids.tolist())):
        name = cloud.labels.get(lid, "unknown")
        rgb = semantic_color(lid)
        count = int(np.sum(cloud.semantic_ids == lid))
        legend[name] = {"id": lid, "rgb": list(rgb), "count": count}
    return write_json(path, legend)


def write_html_viewer(
    path: str | Path,
    cloud,
    *,
    extrinsics: np.ndarray | None = None,
    max_points_inline: int = 200_000,
) -> Path:
    """Self-contained Plotly viewer with optional camera trajectory."""
    import plotly.graph_objects as go
    path = Path(path); ensure_dir(path.parent)

    pts = cloud.points
    cols = cloud.colors
    sem = cloud.semantic_ids
    n = len(pts)
    if n > max_points_inline:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, max_points_inline, replace=False)
        pts, cols, sem = pts[idx], cols[idx], sem[idx]

    rgb_strs = [f"rgb({r},{g},{b})" for r, g, b in cols.tolist()]
    sem_colors = np.array([semantic_color(int(s)) for s in sem], dtype=np.uint8)
    sem_strs = [f"rgb({r},{g},{b})" for r, g, b in sem_colors.tolist()]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        mode="markers", marker=dict(size=1.5, color=rgb_strs),
        name="RGB",
    ))
    fig.add_trace(go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        mode="markers", marker=dict(size=1.5, color=sem_strs),
        name="Semantic", visible="legendonly",
    ))
    if extrinsics is not None and len(extrinsics) > 0:
        centers = []
        for ext in extrinsics:
            ext = np.asarray(ext)
            R = ext[:3, :3]; t = ext[:3, 3]
            centers.append(-R.T @ t)
        centers = np.asarray(centers)
        fig.add_trace(go.Scatter3d(
            x=centers[:,0], y=centers[:,1], z=centers[:,2],
            mode="lines+markers",
            line=dict(color="orange", width=4),
            marker=dict(size=4, color="orange"),
            name="Camera trajectory",
        ))

    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(x=0, y=1),
    )
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path