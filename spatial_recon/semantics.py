from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: object):
        return iterable

from .utils import ensure_dir, list_images, semantic_color, write_json


def _device_index(device: str | None) -> int:
    if device == "auto":
        try:
            import torch

            return 0 if torch.cuda.is_available() else -1
        except ImportError:
            return -1
    if device and device.startswith("cuda"):
        return 0
    return -1


def _paint_label_map(label_map: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for label_id in np.unique(label_map):
        rgb[label_map == label_id] = semantic_color(int(label_id))
    return rgb


def run_semantics(
    image_dir: str | Path,
    out_dir: str | Path,
    model_name: str = "facebook/mask2former-swin-large-ade-semantic",
    device: str | None = "auto",
) -> dict:
    """Run 2D semantic segmentation and save per-frame label maps.

    The 3D fusion step later samples these labels at the same pixels that VGGT
    unprojects, which keeps semantics aligned with geometry.
    """
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError("Semantic segmentation needs `transformers`. Install requirements.txt first.") from exc

    image_paths = list_images(image_dir)
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    segment_device = _device_index(device)
    print(f"[Semantics] Using {'cuda:0' if segment_device == 0 else 'cpu'} for segmentation")
    out_dir = ensure_dir(out_dir)
    masks_dir = ensure_dir(out_dir / "label_maps")
    previews_dir = ensure_dir(out_dir / "previews")

    segmenter = pipeline(
        "image-segmentation",
        model=model_name,
        device=segment_device,
    )

    label_to_id = {"unknown": 0}
    frames = []
    for frame_idx, image_path in enumerate(tqdm(image_paths, desc="semantic segmentation")):
        image = Image.open(image_path).convert("RGB")
        segments = segmenter(image)
        label_map = np.zeros((image.height, image.width), dtype=np.int32)

        for segment in sorted(segments, key=lambda item: float(item.get("score", 0.0))):
            label = str(segment.get("label", "unknown")).lower().replace(" ", "_")
            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)
            mask = np.asarray(segment["mask"].resize(image.size, Image.Resampling.NEAREST))
            label_map[mask > 0] = label_to_id[label]

        np.save(masks_dir / f"{frame_idx:04d}.npy", label_map.astype(np.int16))
        Image.fromarray(_paint_label_map(label_map)).save(previews_dir / f"{frame_idx:04d}.png")
        frames.append({"image": image_path.name, "label_map": f"label_maps/{frame_idx:04d}.npy"})

    id_to_label = {str(idx): label for label, idx in label_to_id.items()}
    metadata = {
        "model": model_name,
        "labels": id_to_label,
        "frames": frames,
    }
    write_json(out_dir / "labels.json", metadata)
    return metadata


def load_label_maps(semantics_dir: str | Path) -> tuple[list[np.ndarray], dict[int, str]]:
    semantics_dir = Path(semantics_dir)
    metadata_path = semantics_dir / "labels.json"
    if not metadata_path.exists():
        return [], {0: "unknown"}
    import json

    metadata = json.loads(metadata_path.read_text())
    labels = {int(k): v for k, v in metadata.get("labels", {}).items()}
    maps = []
    for frame in metadata.get("frames", []):
        maps.append(np.load(semantics_dir / frame["label_map"]))
    return maps, labels
