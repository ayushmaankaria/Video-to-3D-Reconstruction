from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import normalize_uint8, resize_label_map_nearest, semantic_color


@dataclass
class FusedPointCloud:
    points: np.ndarray
    colors_rgb: np.ndarray
    semantic_ids: np.ndarray
    semantic_colors: np.ndarray
    confidence: np.ndarray
    labels: dict[int, str]
    source_frame: np.ndarray
    source_y: np.ndarray
    source_x: np.ndarray
    source_y_vggt: np.ndarray
    source_x_vggt: np.ndarray


def _prediction_points(predictions: dict, use_point_map: bool) -> tuple[np.ndarray, np.ndarray]:
    if use_point_map and "world_points" in predictions:
        return predictions["world_points"], predictions.get("world_points_conf", predictions["depth_conf"])
    return predictions["world_points_from_depth"], predictions["depth_conf"]


def _voxel_reduce(
    points: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    conf: np.ndarray,
    source_frame: np.ndarray,
    source_y: np.ndarray,
    source_x: np.ndarray,
    source_y_vggt: np.ndarray,
    source_x_vggt: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if voxel_size <= 0 or len(points) == 0:
        return points, colors, labels, conf, source_frame, source_y, source_x, source_y_vggt, source_x_vggt

    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    n = int(inverse.max()) + 1

    sums = np.zeros((n, 3), dtype=np.float64)
    color_sums = np.zeros((n, 3), dtype=np.float64)
    conf_sums = np.zeros(n, dtype=np.float64)
    counts = np.bincount(inverse).astype(np.float64)
    np.add.at(sums, inverse, points)
    np.add.at(color_sums, inverse, colors.astype(np.float64))
    np.add.at(conf_sums, inverse, conf)

    unique_labels, compressed_labels = np.unique(labels.astype(np.int64), return_inverse=True)
    votes = np.zeros((n, len(unique_labels)), dtype=np.int32)
    np.add.at(votes, (inverse, compressed_labels), 1)
    voxel_label = unique_labels[votes.argmax(axis=1)].astype(np.int32)

    order = np.argsort(-conf, kind="stable")
    _, first = np.unique(inverse[order], return_index=True)
    representative = order[first]

    return (
        (sums / counts[:, None]).astype(np.float32),
        np.clip(color_sums / counts[:, None], 0, 255).astype(np.uint8),
        voxel_label.astype(np.int32),
        (conf_sums / counts).astype(np.float32),
        source_frame[representative].astype(np.int32),
        source_y[representative].astype(np.int32),
        source_x[representative].astype(np.int32),
        source_y_vggt[representative].astype(np.int32),
        source_x_vggt[representative].astype(np.int32),
    )


def _original_pixel_indices(
    predictions: dict,
    source_frame: np.ndarray,
    source_y_vggt: np.ndarray,
    source_x_vggt: np.ndarray,
    points_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = points_hw
    source_image_hw = np.asarray(predictions.get("source_image_hw", []))
    if source_image_hw.ndim != 2 or source_image_hw.shape[1] != 2 or source_image_hw.shape[0] <= int(source_frame.max(initial=0)):
        return source_y_vggt.copy(), source_x_vggt.copy()

    frame_hw = source_image_hw[source_frame]
    scale_y = frame_hw[:, 0] / float(h)
    scale_x = frame_hw[:, 1] / float(w)
    source_y = np.rint((source_y_vggt + 0.5) * scale_y - 0.5).astype(np.int32)
    source_x = np.rint((source_x_vggt + 0.5) * scale_x - 0.5).astype(np.int32)
    source_y = np.clip(source_y, 0, frame_hw[:, 0] - 1)
    source_x = np.clip(source_x, 0, frame_hw[:, 1] - 1)
    return source_y.astype(np.int32), source_x.astype(np.int32)


def fuse_predictions(
    predictions: dict,
    label_maps: list[np.ndarray] | None = None,
    labels: dict[int, str] | None = None,
    conf_percentile: float = 35.0,
    sample_stride: int = 2,
    voxel_size: float = 0.015,
    max_points: int = 450_000,
    use_point_map: bool = False,
) -> FusedPointCloud:
    points_map, conf_map = _prediction_points(predictions, use_point_map=use_point_map)
    images = predictions["images"]
    if images.shape[1] == 3:
        images = images.transpose(0, 2, 3, 1)

    if points_map.shape[-1] != 3:
        raise ValueError(f"Expected points with final dimension 3, got {points_map.shape}")

    stride = max(int(sample_stride), 1)
    points = points_map[:, ::stride, ::stride, :].reshape(-1, 3)
    conf = conf_map[:, ::stride, ::stride].reshape(-1)
    colors = normalize_uint8(images[:, ::stride, ::stride, :]).reshape(-1, 3)
    frame_grid, y_grid, x_grid = np.indices(points_map.shape[:3])
    source_frame = frame_grid[:, ::stride, ::stride].reshape(-1).astype(np.int32)
    source_y_vggt = y_grid[:, ::stride, ::stride].reshape(-1).astype(np.int32)
    source_x_vggt = x_grid[:, ::stride, ::stride].reshape(-1).astype(np.int32)
    source_y, source_x = _original_pixel_indices(
        predictions, source_frame, source_y_vggt, source_x_vggt, points_map.shape[1:3]
    )

    semantic_ids = np.zeros(len(points), dtype=np.int32)
    if label_maps:
        resized = []
        h, w = points_map.shape[1:3]
        for label_map in label_maps[: points_map.shape[0]]:
            resized.append(resize_label_map_nearest(label_map, (h, w))[::stride, ::stride])
        if resized:
            semantic_ids = np.stack(resized).reshape(-1).astype(np.int32)

    finite = np.isfinite(points).all(axis=1) & np.isfinite(conf)
    threshold = np.percentile(conf[finite], conf_percentile) if finite.any() else 0.0
    keep = finite & (conf >= threshold) & (conf > 1e-6)

    points = points[keep]
    colors = colors[keep]
    semantic_ids = semantic_ids[keep]
    conf = conf[keep]
    source_frame = source_frame[keep]
    source_y = source_y[keep]
    source_x = source_x[keep]
    source_y_vggt = source_y_vggt[keep]
    source_x_vggt = source_x_vggt[keep]

    if len(points) > 20:
        center = np.median(points, axis=0)
        radius = np.linalg.norm(points - center, axis=1)
        keep_radius = radius <= np.percentile(radius, 99.5)
        points, colors, semantic_ids, conf = points[keep_radius], colors[keep_radius], semantic_ids[keep_radius], conf[keep_radius]
        source_frame, source_y, source_x = source_frame[keep_radius], source_y[keep_radius], source_x[keep_radius]
        source_y_vggt, source_x_vggt = source_y_vggt[keep_radius], source_x_vggt[keep_radius]

    points, colors, semantic_ids, conf, source_frame, source_y, source_x, source_y_vggt, source_x_vggt = _voxel_reduce(
        points, colors, semantic_ids, conf, source_frame, source_y, source_x, source_y_vggt, source_x_vggt, voxel_size
    )

    if len(points) > max_points:
        rng = np.random.default_rng(7)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points, colors, semantic_ids, conf = points[idx], colors[idx], semantic_ids[idx], conf[idx]
        source_frame, source_y, source_x = source_frame[idx], source_y[idx], source_x[idx]
        source_y_vggt, source_x_vggt = source_y_vggt[idx], source_x_vggt[idx]

    labels = labels or {0: "unknown"}
    semantic_colors = np.asarray([semantic_color(int(label_id)) for label_id in semantic_ids], dtype=np.uint8)
    return FusedPointCloud(
        points=points.astype(np.float32),
        colors_rgb=colors.astype(np.uint8),
        semantic_ids=semantic_ids.astype(np.int32),
        semantic_colors=semantic_colors,
        confidence=conf.astype(np.float32),
        labels=labels,
        source_frame=source_frame.astype(np.int32),
        source_y=source_y.astype(np.int32),
        source_x=source_x.astype(np.int32),
        source_y_vggt=source_y_vggt.astype(np.int32),
        source_x_vggt=source_x_vggt.astype(np.int32),
    )
