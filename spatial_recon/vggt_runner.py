from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from .utils import ensure_dir, list_images


def _resolve_device(device: str | None) -> str:
    import torch

    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _autocast_context(device: str):
    import torch

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        return torch.cuda.amp.autocast(dtype=dtype)
    if device == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)
    return nullcontext()


def run_vggt(
    image_dir: str | Path,
    out_npz: str | Path,
    checkpoint: str = "facebook/VGGT-1B",
    device: str | None = "auto",
) -> dict[str, Any]:
    """Run Meta VGGT and save dense geometry predictions to an NPZ archive."""
    import torch

    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError as exc:
        raise ImportError(
            "VGGT is not installed. Install project dependencies with "
            "`pip install -r requirements.txt`, or in Colab run the setup cell in the README."
        ) from exc

    image_paths = list_images(image_dir)
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    resolved_device = _resolve_device(device)
    model = VGGT.from_pretrained(checkpoint).to(resolved_device)
    model.eval()

    images = load_and_preprocess_images([str(p) for p in image_paths]).to(resolved_device)
    with torch.no_grad():
        with _autocast_context(resolved_device):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    packed: dict[str, Any] = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            packed[key] = value.detach().cpu().numpy().squeeze(0)
    depth = packed["depth"]
    packed["world_points_from_depth"] = unproject_depth_map_to_point_map(
        depth, packed["extrinsic"], packed["intrinsic"]
    )
    packed["source_images"] = np.asarray([p.name for p in image_paths])

    out_npz = Path(out_npz)
    ensure_dir(out_npz.parent)
    np.savez_compressed(out_npz, **packed)
    return packed


def load_predictions(path: str | Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}
