from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for predict.py") from exc

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_): return it

from .utils import ensure_dir, list_images


_MODEL_CACHE: dict[str, Any] = {}


def _load_vggt_omega(checkpoint: str | Path, device: str):
    """Build VGGT-Omega and load weights from a local .pt file."""
    key = f"{checkpoint}:{device}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    try:
        from vggt_omega.models import VGGTOmega
    except ImportError as exc:
        raise ImportError(
            "VGGT-Omega is not installed. Run:\n"
            "  git clone https://github.com/facebookresearch/vggt-omega\n"
            "  cd vggt-omega && pip install -r requirements.txt && pip install -e ."
        ) from exc

    print(f"[Predict] Loading VGGT-Omega checkpoint: {checkpoint}")
    model = VGGTOmega().to(device).eval()
    state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict) and "model" in state and "pose_enc" not in state:
        state = state["model"]
    model.load_state_dict(state)
    _MODEL_CACHE[key] = model
    return model


def _unproject_depth_to_world(
    depth: np.ndarray,        # [F, H, W]
    extrinsics: np.ndarray,   # [F, 4, 4] world->camera (VGGT convention)
    intrinsics: np.ndarray,   # [F, 3, 3]
) -> np.ndarray:
    """Lift per-frame depth to a single world-space point grid.

    Convention: extrinsics[i] is [R | t] with p_cam = R @ p_world + t, so
    p_world = R^T @ (p_cam - t).
    """
    F, H, W = depth.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pix_h = np.stack([xx, yy, np.ones_like(xx)], axis=-1).astype(np.float64)  # (H,W,3)

    out = np.zeros((F, H, W, 3), dtype=np.float32)
    for i in range(F):
        K = np.asarray(intrinsics[i], dtype=np.float64).reshape(3, 3)
        K_inv = np.linalg.inv(K)
        rays = pix_h @ K_inv.T                                # (H, W, 3)
        cam = rays * depth[i, :, :, None].astype(np.float64)  # (H, W, 3)

        ext = np.asarray(extrinsics[i], dtype=np.float64)
        if ext.shape == (4, 4):
            ext = ext[:3, :4]
        R = ext[:, :3]
        t = ext[:, 3]
        world = (cam.reshape(-1, 3) - t) @ R                  # == (R^T @ (p-t)).T
        out[i] = world.reshape(H, W, 3).astype(np.float32)
    return out


@torch.inference_mode()
def predict_geometry(
    run_dir: str | Path,
    *,
    checkpoint: str | Path,
    image_resolution: int = 512,
    device: str | None = None,
    save: bool = True,
) -> dict[str, np.ndarray]:
    """Run VGGT-Omega on frames in ``run_dir/frames`` and save predictions.npz.

    Output keys:
        depth                    : [F, H, W]      float32
        depth_conf               : [F, H, W]      float32
        intrinsics               : [F, 3, 3]      float32   (model-grid pixels)
        extrinsics               : [F, 4, 4]      float32   (world->camera)
        images                   : [F, 3, H, W]   float32   in [0, 1]
        world_points_from_depth  : [F, H, W, 3]   float32
        source_image_hw          : [F, 2]         int32
        source_image_names       : [F]            <U
    """
    try:
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera
    except ImportError as exc:
        raise ImportError("VGGT-Omega is not installed (see _load_vggt_omega).") from exc

    run_dir = Path(run_dir)
    frames_dir = run_dir / "frames"
    image_paths = list_images(frames_dir)
    if not image_paths:
        raise FileNotFoundError(f"No frames in {frames_dir}")

    # Capture each frame's on-disk resolution before VGGT's own preprocessing.
    source_hw: list[tuple[int, int]] = []
    for p in image_paths:
        with Image.open(p) as im:
            w, h = im.size
        source_hw.append((h, w))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Predict] Device: {device}")
    if device == "cuda":
        print(f"[Predict] GPU: {torch.cuda.get_device_name(0)}")

    model = _load_vggt_omega(checkpoint, device)

    images = load_and_preprocess_images(
        [str(p) for p in image_paths], image_resolution=image_resolution
    ).to(device)
    print(f"[Predict] Preprocessed image tensor shape: {tuple(images.shape)}")

    print("[Predict] Running VGGT-Omega…")
    preds = model(images)
    print("[Predict] Forward pass complete.")

    proc_h, proc_w = preds["images"].shape[-2:]
    extr, intr = encoding_to_camera(preds["pose_enc"], (proc_h, proc_w))

    def _to_np(t):
        if isinstance(t, torch.Tensor):
            a = t.detach().cpu().to(torch.float32).numpy()
        else:
            a = np.asarray(t)
        # Drop a leading singleton batch dim if present (B=1 is common).
        if a.ndim > 0 and a.shape[0] == 1:
            a = a[0]
        return a

    depth      = _to_np(preds["depth"])
    depth_conf = _to_np(preds["depth_conf"])
    extr_np    = _to_np(extr)
    intr_np    = _to_np(intr)
    images_np  = _to_np(preds["images"])

    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
        depth_conf = depth_conf[..., 0]
    if extr_np.shape[-2:] == (3, 4):
        pad = np.zeros((*extr_np.shape[:-2], 4, 4), dtype=np.float32)
        pad[..., :3, :4] = extr_np
        pad[..., 3, 3] = 1.0
        extr_np = pad

    F = len(image_paths)
    if depth.shape[0] != F:
        raise RuntimeError(f"VGGT-Omega returned {depth.shape[0]} frames; expected {F}.")

    world_points = _unproject_depth_to_world(depth, extr_np, intr_np)

    packed: dict[str, np.ndarray] = {
        "depth":                   depth.astype(np.float32),
        "depth_conf":              depth_conf.astype(np.float32),
        "intrinsics":              intr_np.astype(np.float32),
        "extrinsics":              extr_np.astype(np.float32),
        "images":                  images_np.astype(np.float32),
        "world_points_from_depth": world_points.astype(np.float32),
        "source_image_hw":         np.asarray(source_hw, dtype=np.int32),
        "source_image_names":      np.asarray([p.name for p in image_paths]),
    }

    if save:
        out = run_dir / "predictions.npz"
        ensure_dir(out.parent)
        np.savez_compressed(out, **packed)
        print(f"[Predict] Wrote {out}")
    return packed


def load_predictions(run_dir_or_path: str | Path) -> dict[str, np.ndarray]:
    p = Path(run_dir_or_path)
    if p.is_dir():
        p = p / "predictions.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}