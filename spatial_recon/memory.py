import json
import numpy as np
from pathlib import Path
import argparse
from sklearn.cluster import DBSCAN

from .export import write_ply

# Rule-based affordance tags used by the scene memory query demo.
AFFORDANCES = {
    "chair": ["sittable", "movable"],
    "desk": ["support_surface", "placeable"],
    "table": ["support_surface", "placeable"],
    "monitor": ["readable", "fragile"],
    "keyboard": ["typable", "movable"],
    "mouse": ["clickable", "movable"],
    "cup": ["graspable", "drinkable", "movable"],
    "cable": ["graspable", "flexible"],
    "medicine": ["graspable", "consumable"],
    "toothbrush": ["graspable", "usable"],
    "guitar": ["playable", "movable", "fragile"]
}

BACKGROUND_CLASSES = {"wall", "floor", "ceiling", "unknown"}
NEAR_XY_DISTANCE_M = 0.55
NEAR_3D_DISTANCE_M = 0.80
NEAR_Z_GAP_M = 0.12
NEAR_Z_CENTROID_M = 0.35
ON_TOP_Z_TOLERANCE_M = 0.30
ON_TOP_MIN_XY_OVERLAP = 0.05
ON_TOP_MAX_AREA_RATIO = 0.65
ON_TOP_MAX_VOLUME_RATIO = 0.65

OBJECT_PALETTE = np.array([
    [230, 57, 70],
    [29, 185, 84],
    [58, 134, 255],
    [255, 183, 3],
    [131, 56, 236],
    [255, 111, 97],
    [0, 166, 166],
    [255, 127, 17],
    [112, 214, 255],
    [181, 23, 158],
    [124, 252, 0],
    [255, 0, 110],
], dtype=np.uint8)


def _object_color(index: int) -> np.ndarray:
    return OBJECT_PALETTE[index % len(OBJECT_PALETTE)]


def _bounds(obj: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(obj["bbox_min_xyz"]), np.asarray(obj["bbox_max_xyz"])


def _xy_distance(obj_a: dict, obj_b: dict) -> float:
    center_a = np.asarray(obj_a["centroid_xyz"])[:2]
    center_b = np.asarray(obj_b["centroid_xyz"])[:2]
    return float(np.linalg.norm(center_a - center_b))


def _bbox_xy_overlap_fraction(obj_a: dict, obj_b: dict) -> float:
    min_a, max_a = _bounds(obj_a)
    min_b, max_b = _bounds(obj_b)
    overlap_min = np.maximum(min_a[:2], min_b[:2])
    overlap_max = np.minimum(max_a[:2], max_b[:2])
    overlap = np.maximum(overlap_max - overlap_min, 0.0)
    overlap_area = float(overlap[0] * overlap[1])
    area_a = float(np.prod(np.maximum(max_a[:2] - min_a[:2], 1e-6)))
    area_b = float(np.prod(np.maximum(max_b[:2] - min_b[:2], 1e-6)))
    return overlap_area / max(min(area_a, area_b), 1e-6)


def _vertical_gap(obj_a: dict, obj_b: dict) -> float:
    min_a, max_a = _bounds(obj_a)
    min_b, max_b = _bounds(obj_b)
    if min_a[2] > max_b[2]:
        return float(min_a[2] - max_b[2])
    if min_b[2] > max_a[2]:
        return float(min_b[2] - max_a[2])
    return 0.0


def _is_near(obj_a: dict, obj_b: dict) -> bool:
    centroid_a = np.asarray(obj_a["centroid_xyz"])
    centroid_b = np.asarray(obj_b["centroid_xyz"])
    xy_distance = _xy_distance(obj_a, obj_b)
    radius_distance = float(np.linalg.norm(centroid_a - centroid_b))
    z_centroid_distance = float(abs(centroid_a[2] - centroid_b[2]))
    z_gap = _vertical_gap(obj_a, obj_b)
    vertical_ok = z_gap <= NEAR_Z_GAP_M or z_centroid_distance <= NEAR_Z_CENTROID_M
    return (
        xy_distance <= NEAR_XY_DISTANCE_M
        and radius_distance <= NEAR_3D_DISTANCE_M
        and vertical_ok
    )


def _is_on_top_of(obj_a: dict, obj_b: dict) -> bool:
    if obj_a["id"] == obj_b["id"]:
        return False
    if "support_surface" not in obj_b.get("affordances", []):
        return False
    if obj_a["label"] in {"desk", "table", "chair"}:
        return False

    min_a, _ = _bounds(obj_a)
    min_b, max_b = _bounds(obj_b)
    xy_overlap = _bbox_xy_overlap_fraction(obj_a, obj_b)
    extent_a = np.asarray(obj_a["bbox_extent_xyz"])
    extent_b = np.asarray(obj_b["bbox_extent_xyz"])
    xy_area_a = float(np.prod(np.maximum(extent_a[:2], 1e-6)))
    xy_area_b = float(np.prod(np.maximum(extent_b[:2], 1e-6)))
    area_ratio = xy_area_a / max(xy_area_b, 1e-6)
    volume_ratio = float(obj_a["volume"]) / max(float(obj_b["volume"]), 1e-6)

    bottom_a = float(min_a[2])
    top_b = float(max_b[2])
    lower_surface_b = float(min_b[2])
    surface_gap = min(abs(bottom_a - top_b), abs(bottom_a - lower_surface_b))
    return (
        surface_gap <= ON_TOP_Z_TOLERANCE_M
        and xy_overlap >= ON_TOP_MIN_XY_OVERLAP
        and area_ratio <= ON_TOP_MAX_AREA_RATIO
        and volume_ratio <= ON_TOP_MAX_VOLUME_RATIO
    )


def _write_memory_instance_ply(path: Path, points: np.ndarray, object_point_indices: dict[str, np.ndarray]) -> None:
    colors = np.full((len(points), 3), 85, dtype=np.uint8)
    for index, indices in enumerate(object_point_indices.values()):
        colors[indices] = _object_color(index)
    write_ply(path, {"points": points, "colors": colors})


def _write_memory_instance_html(
    path: Path,
    points: np.ndarray,
    scene_memory: dict,
    object_point_indices: dict[str, np.ndarray],
    max_background_points: int = 80_000,
    max_points_per_object: int = 35_000,
) -> None:
    import plotly.graph_objects as go

    rng = np.random.default_rng(0)
    assigned = np.zeros(len(points), dtype=bool)
    for indices in object_point_indices.values():
        assigned[indices] = True

    background_indices = np.flatnonzero(~assigned)
    if len(background_indices) > max_background_points:
        background_indices = rng.choice(background_indices, max_background_points, replace=False)

    fig = go.Figure()
    if len(background_indices):
        bg = points[background_indices]
        fig.add_trace(go.Scatter3d(
            x=bg[:, 0], y=bg[:, 1], z=bg[:, 2],
            mode="markers",
            marker=dict(size=1, color="rgb(80,80,80)", opacity=0.18),
            name="unclustered/background",
            hoverinfo="skip",
        ))

    for index, (obj_id, indices) in enumerate(object_point_indices.items()):
        if len(indices) > max_points_per_object:
            indices = rng.choice(indices, max_points_per_object, replace=False)
        obj = points[indices]
        color = _object_color(index)
        label = scene_memory[obj_id]["label"]
        point_count = scene_memory[obj_id]["num_points"]
        confidence = scene_memory[obj_id]["mean_confidence"]
        hover = (
            f"{obj_id}<br>"
            f"label: {label}<br>"
            f"points: {point_count}<br>"
            f"mean confidence: {confidence:.3f}"
            "<extra></extra>"
        )
        fig.add_trace(go.Scatter3d(
            x=obj[:, 0], y=obj[:, 1], z=obj[:, 2],
            mode="markers",
            marker=dict(size=2.2, color=f"rgb({color[0]},{color[1]},{color[2]})"),
            name=obj_id,
            hovertemplate=hover,
        ))

    centroids = np.array([data["centroid_xyz"] for data in scene_memory.values()], dtype=np.float32)
    names = list(scene_memory.keys())
    if len(centroids):
        fig.add_trace(go.Scatter3d(
            x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
            mode="markers+text",
            marker=dict(size=5, color="white", line=dict(color="black", width=2)),
            text=names,
            textposition="top center",
            name="object centroids",
            hoverinfo="text",
        ))

    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(x=0, y=1),
    )
    fig.write_html(str(path), include_plotlyjs="cdn")


def build_scene_memory(run_dir: Path, eps=0.15, min_samples=20):
    """Clusters semantic points into discrete object entities."""
    exports_dir = run_dir / "exports"
    semantics_dir = run_dir / "semantics"
    
    # Load fused points and labels
    fused_data = np.load(exports_dir / "fused_points.npz")
    points = fused_data["points"]
    if "semantic_ids" in fused_data:
        labels = fused_data["semantic_ids"]
    elif "labels" in fused_data:
        labels = fused_data["labels"]
    else:
        raise KeyError("fused_points.npz must contain either 'semantic_ids' or 'labels'.")
    confidences = fused_data["confidence"]
    
    with open(semantics_dir / "labels.json", "r") as f:
        metadata = json.load(f)
        id_to_concept = metadata.get("labels", metadata)
        # Handle both string and int keys depending on how it was saved
        id_to_concept = {int(k): v for k, v in id_to_concept.items()}

    scene_memory = {}
    object_point_indices = {}
    
    for class_id, concept in id_to_concept.items():
        if concept in BACKGROUND_CLASSES:
            continue
            
        # Extract points for this specific semantic class
        class_mask = (labels == class_id)
        class_indices = np.flatnonzero(class_mask)
        class_points = points[class_mask]
        class_conf = confidences[class_mask]
        
        if len(class_points) < min_samples:
            continue
            
        # Cluster points into distinct objects
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(class_points)
        unique_labels = set(clustering.labels_)
        
        instance_count = 1
        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue # Ignore noise points
                
            cluster_mask = (clustering.labels_ == cluster_id)
            obj_points = class_points[cluster_mask]
            obj_conf = class_conf[cluster_mask]
            obj_indices = class_indices[cluster_mask]
            
            # Compute spatial properties
            centroid = np.mean(obj_points, axis=0)
            min_bound = np.min(obj_points, axis=0)
            max_bound = np.max(obj_points, axis=0)
            bbox_extent = max_bound - min_bound
            
            obj_id = f"{concept}_{instance_count:02d}"
            
            scene_memory[obj_id] = {
                "id": obj_id,
                "label": concept,
                "centroid_xyz": centroid.tolist(),
                "bbox_min_xyz": min_bound.tolist(),
                "bbox_max_xyz": max_bound.tolist(),
                "bbox_extent_xyz": bbox_extent.tolist(),
                "volume": float(np.prod(bbox_extent)),
                "num_points": int(len(obj_points)),
                "mean_confidence": float(np.mean(obj_conf)),
                "affordances": AFFORDANCES.get(concept, ["unknown"])
            }
            object_point_indices[obj_id] = obj_indices
            instance_count += 1

    # Compute spatial relations from object centroids and 3D bounding boxes.
    object_ids = list(scene_memory.keys())
    for obj_a_id in object_ids:
        near_objects = []
        on_top_of = []
        obj_a = scene_memory[obj_a_id]

        for obj_b_id in object_ids:
            if obj_a_id == obj_b_id:
                continue
            obj_b = scene_memory[obj_b_id]

            if _is_near(obj_a, obj_b):
                near_objects.append(obj_b_id)
            if _is_on_top_of(obj_a, obj_b):
                on_top_of.append(obj_b_id)
                
        scene_memory[obj_a_id]["relations"] = {
            "near": near_objects,
            "on_top_of": on_top_of,
        }

    for obj_id in object_ids:
        supported = [
            other_id
            for other_id in object_ids
            if obj_id in scene_memory[other_id]["relations"]["on_top_of"]
        ]
        scene_memory[obj_id]["relations"]["supports"] = supported

    # Export memory
    memory_path = exports_dir / "scene_memory.json"
    with open(memory_path, "w") as f:
        json.dump(scene_memory, f, indent=4)

    _write_memory_instance_ply(exports_dir / "memory_instances.ply", points, object_point_indices)
    _write_memory_instance_html(exports_dir / "memory_instances.html", points, scene_memory, object_point_indices)
        
    print(f"Exported {len(scene_memory)} object entities to {memory_path}")
    print(f"Wrote visual checks to {exports_dir / 'memory_instances.ply'} and {exports_dir / 'memory_instances.html'}")
    return scene_memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build object-level scene memory from a semantic point cloud.")
    parser.add_argument("--run", type=str, required=True, help="Path to the run directory (e.g., runs/desk)")
    parser.add_argument("--eps", type=float, default=0.15, help="DBSCAN clustering radius (meters)")
    parser.add_argument("--min-samples", type=int, default=20, help="Minimum points needed to form an object cluster")
    args = parser.parse_args()
    
    build_scene_memory(Path(args.run), eps=args.eps, min_samples=args.min_samples)
