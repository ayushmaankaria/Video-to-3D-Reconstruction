from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_): return it

from .utils import ensure_dir


@dataclass
class FusedPointCloud:
    points: np.ndarray             # [N, 3] float32   world coords
    colors: np.ndarray             # [N, 3] uint8
    confidence: np.ndarray         # [N]    float32
    semantic_ids: np.ndarray       # [N]    int32
    labels: dict[int, str] = field(default_factory=dict)
    source_frame: np.ndarray = None        # [N] int32
    source_y_vggt: np.ndarray = None       # [N] int32 (VGGT prediction grid)
    source_x_vggt: np.ndarray = None       # [N] int32
    source_y: np.ndarray = None            # [N] int32 (original saved-frame res)
    source_x: np.ndarray = None            # [N] int32


def _to_numpy(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _resize_label_map(lm: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    H, W = target_hw
    lm = np.asarray(lm)
    if lm.shape == (H, W):
        return lm.astype(np.int32)
    # PIL "I" mode handles arbitrary int32 label values without quantization.
    img = Image.fromarray(lm.astype(np.int32), mode="I")
    return np.asarray(img.resize((W, H), Image.Resampling.NEAREST), dtype=np.int32)


def _frame_image_to_uint8_hwc(frame: np.ndarray) -> np.ndarray:
    """Accepts (3, H, W) float [0,1] or uint8, or (H, W, 3); returns (H, W, 3) uint8."""
    arr = frame
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _voxel_reduce(
    points: np.ndarray,
    confs: np.ndarray,
    labels: np.ndarray,
    *,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse points sharing a voxel.

    Returns
    -------
    keep_idx       : indices into the input arrays for the chosen representatives
                     (one per voxel; the one with the highest confidence).
    majority_label : the most common label inside each voxel; ties broken by the
                     max-confidence value attached to each tied label.
    """
    n = len(points)
    if voxel_size is None or voxel_size <= 0:
        return np.arange(n, dtype=np.int64), labels.astype(np.int32, copy=True)

    keys = np.floor(points / voxel_size).astype(np.int64)
    keys = keys - keys.min(axis=0, keepdims=True)
    dims = keys.max(axis=0) + 1
    # 1-D hash; .ravel() guards against numpy 2.x return_inverse multi-D outputs
    # if anyone ever rewrites this with np.unique(return_inverse=True).
    flat = (keys[:, 0] * (dims[1] * dims[2]) + keys[:, 1] * dims[2] + keys[:, 2]).ravel()

    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    is_new = np.concatenate([[True], flat_sorted[1:] != flat_sorted[:-1]])
    starts = np.flatnonzero(is_new)
    ends = np.concatenate([starts[1:], [n]])
    n_vox = len(starts)

    confs_sorted  = confs[order]
    labels_sorted = labels[order]
    keep_idx   = np.empty(n_vox, dtype=np.int64)
    maj_labels = np.empty(n_vox, dtype=np.int32)

    for i in range(n_vox):
        s, e = int(starts[i]), int(ends[i])
        lc = confs_sorted[s:e]
        ll = labels_sorted[s:e]
        keep_idx[i] = order[s + int(np.argmax(lc))]
        if e - s == 1:
            maj_labels[i] = ll[0]
            continue
        u, c = np.unique(ll, return_counts=True)
        top = c.max()
        cand = u[c == top]
        if cand.size == 1:
            maj_labels[i] = cand[0]
        else:
            best_label, best_conf = cand[0], -np.inf
            for k in cand:
                mc = float(lc[ll == k].max())
                if mc > best_conf:
                    best_conf, best_label = mc, k
            maj_labels[i] = best_label
    return keep_idx, maj_labels


def fuse_predictions(
    predictions: dict,
    *,
    label_maps: list[np.ndarray] | None = None,
    labels: dict[int, str] | None = None,
    conf_percentile: float = 35.0,
    sample_stride: int = 2,
    voxel_size: float = 0.015,
    max_points: int | None = 2_000_000,
    save_to: str | Path | None = None,
) -> FusedPointCloud:
    """Fuse per-frame VGGT-Omega geometry + SAM 3 label maps into one cloud."""
    world_points = _to_numpy(predictions["world_points_from_depth"]).astype(np.float32)
    depth_conf   = _to_numpy(predictions["depth_conf"]).astype(np.float32)
    images       = _to_numpy(predictions["images"])
    source_hw    = np.asarray(predictions["source_image_hw"], dtype=np.int32)

    if world_points.ndim != 4 or world_points.shape[-1] != 3:
        raise ValueError(f"world_points_from_depth must be [F,H,W,3], got {world_points.shape}")
    F, H, W, _ = world_points.shape
    if depth_conf.shape != (F, H, W):
        raise ValueError(f"depth_conf shape {depth_conf.shape} != ({F},{H},{W})")
    if source_hw.shape != (F, 2):
        raise ValueError(f"source_image_hw shape {source_hw.shape} != ({F},2)")

    if labels is None:
        labels = {0: "unknown"}
    labels = {int(k): v for k, v in labels.items()}

    if label_maps is None:
        label_maps = [np.zeros((H, W), dtype=np.int32) for _ in range(F)]
    if len(label_maps) != F:
        raise ValueError(f"Got {len(label_maps)} label_maps but predictions cover {F} frames.")

    # ----- global confidence percentile -----
    flat_conf = depth_conf.reshape(-1)
    finite_conf = flat_conf[np.isfinite(flat_conf) & (flat_conf > 0)]
    if conf_percentile <= 0 or finite_conf.size == 0:
        conf_threshold = 0.0
    else:
        conf_threshold = float(np.percentile(finite_conf, conf_percentile))
    if F * H * W <= 64:
        print(f"[Fusion] conf threshold @ p{conf_percentile:.0f} = {conf_threshold:.4f}")

    # ----- pixel sampling grid (shared across frames) -----
    stride = max(1, int(sample_stride))
    ys = np.arange(0, H, stride, dtype=np.int32)
    xs = np.arange(0, W, stride, dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    yy = yy.reshape(-1); xx = xx.reshape(-1)

    bins = {k: [] for k in ("pts", "cols", "conf", "lab", "fid", "yv", "xv", "y", "x")}

    iterator = range(F) if F == 1 else tqdm(range(F), desc="Fusing frames")
    for f in iterator:
        lm = _resize_label_map(label_maps[f], (H, W))
        pts = world_points[f, yy, xx]
        cnf = depth_conf[f, yy, xx]
        lab = lm[yy, xx].astype(np.int32)

        valid = np.isfinite(pts).all(axis=1) & np.isfinite(cnf) & (cnf >= conf_threshold)
        if not valid.any():
            continue
        pts = pts[valid]; cnf = cnf[valid]; lab = lab[valid]
        yv = yy[valid].astype(np.int32); xv = xx[valid].astype(np.int32)

        rgb_hwc = _frame_image_to_uint8_hwc(images[f])
        cols = rgb_hwc[yv, xv]

        src_h, src_w = int(source_hw[f, 0]), int(source_hw[f, 1])
        sy = np.clip(np.round(yv.astype(np.float32) * (src_h / max(H, 1))).astype(np.int32), 0, src_h - 1)
        sx = np.clip(np.round(xv.astype(np.float32) * (src_w / max(W, 1))).astype(np.int32), 0, src_w - 1)

        bins["pts"].append(pts);  bins["cols"].append(cols);   bins["conf"].append(cnf)
        bins["lab"].append(lab);  bins["fid"].append(np.full(len(pts), f, np.int32))
        bins["yv"].append(yv);    bins["xv"].append(xv)
        bins["y"].append(sy);     bins["x"].append(sx)

    if not bins["pts"]:
        empty = lambda shape, dt: np.zeros(shape, dtype=dt)
        return FusedPointCloud(
            points=empty((0, 3), np.float32), colors=empty((0, 3), np.uint8),
            confidence=empty((0,), np.float32), semantic_ids=empty((0,), np.int32),
            labels=labels,
            source_frame=empty((0,), np.int32),
            source_y_vggt=empty((0,), np.int32), source_x_vggt=empty((0,), np.int32),
            source_y=empty((0,), np.int32),      source_x=empty((0,), np.int32),
        )

    points = np.concatenate(bins["pts"], axis=0)
    colors = np.concatenate(bins["cols"], axis=0).astype(np.uint8)
    confs  = np.concatenate(bins["conf"], axis=0).astype(np.float32)
    sem    = np.concatenate(bins["lab"], axis=0).astype(np.int32)
    fid    = np.concatenate(bins["fid"], axis=0).astype(np.int32)
    yv_all = np.concatenate(bins["yv"], axis=0).astype(np.int32)
    xv_all = np.concatenate(bins["xv"], axis=0).astype(np.int32)
    y_all  = np.concatenate(bins["y"], axis=0).astype(np.int32)
    x_all  = np.concatenate(bins["x"], axis=0).astype(np.int32)

    # ----- voxel reduction + majority-vote semantics -----
    keep, maj = _voxel_reduce(points, confs, sem, voxel_size=voxel_size)
    points = points[keep]; colors = colors[keep]; confs = confs[keep]
    fid = fid[keep]
    yv_all = yv_all[keep]; xv_all = xv_all[keep]
    y_all  = y_all[keep];  x_all  = x_all[keep]
    sem = maj.astype(np.int32)

    # ----- cap so viewer.html stays responsive -----
    if max_points is not None and len(points) > max_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(points), size=max_points, replace=False))
        points = points[idx]; colors = colors[idx]; confs = confs[idx]
        sem = sem[idx]; fid = fid[idx]
        yv_all = yv_all[idx]; xv_all = xv_all[idx]
        y_all  = y_all[idx];  x_all  = x_all[idx]

    cloud = FusedPointCloud(
        points=points, colors=colors, confidence=confs,
        semantic_ids=sem, labels=labels,
        source_frame=fid,
        source_y_vggt=yv_all, source_x_vggt=xv_all,
        source_y=y_all,       source_x=x_all,
    )

    if save_to is not None:
        save_fused_cloud(save_to, cloud)
    return cloud


def save_fused_cloud(path: str | Path, cloud: FusedPointCloud) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        points=cloud.points, colors=cloud.colors,
        confidence=cloud.confidence, semantic_ids=cloud.semantic_ids,
        source_frame=cloud.source_frame,
        source_y_vggt=cloud.source_y_vggt, source_x_vggt=cloud.source_x_vggt,
        source_y=cloud.source_y, source_x=cloud.source_x,
        labels_json=np.array(json.dumps({str(k): v for k, v in cloud.labels.items()})),
    )
    return path


def load_fused_cloud(path: str | Path) -> FusedPointCloud:
    with np.load(path, allow_pickle=True) as z:
        labels_raw = json.loads(str(z["labels_json"]))
        labels = {int(k): v for k, v in labels_raw.items()}
        return FusedPointCloud(
            points=z["points"], colors=z["colors"],
            confidence=z["confidence"], semantic_ids=z["semantic_ids"],
            labels=labels,
            source_frame=z["source_frame"],
            source_y_vggt=z["source_y_vggt"], source_x_vggt=z["source_x_vggt"],
            source_y=z["source_y"], source_x=z["source_x"],
        )