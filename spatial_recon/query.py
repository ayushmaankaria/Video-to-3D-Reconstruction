from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for query.py") from exc

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_): return it

from .utils import ensure_dir, list_images
from .video import load_frames_meta
from .fusion import load_fused_cloud
from .export import write_ply


_SAM3_CACHE: dict[str, Any] = {}


def _load_sam3(device: str):
    if device in _SAM3_CACHE:
        return _SAM3_CACHE[device]
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        raise ImportError("SAM 3 not installed (see semantics.py for install steps).") from exc
    # Build SAM 3 in its native (mixed) dtype. Don't force-cast the whole
    # model to fp32 or bf16 -- some sublayers keep fp32 weights while the
    # image processor produces bf16 activations. We reconcile the two with a
    # bf16 autocast block at forward time (see _sam3_score_maps).
    model = build_sam3_image_model()
    try:
        model = model.to(device)
    except Exception:
        pass
    model.eval()
    processor = Sam3Processor(model)
    _SAM3_CACHE[device] = (model, processor)
    return model, processor


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        # NumPy can't cast bf16/fp16 directly -- promote first.
        if x.dtype in (torch.bfloat16, torch.float16):
            x = x.float()
        return x.detach().cpu().numpy()
    return np.asarray(x)


@torch.inference_mode()
def _sam3_score_maps(
    frames: np.ndarray,        # [F, H, W, 3] uint8
    text: str,
    *,
    device: str,
    score_threshold: float = 0.5,
) -> np.ndarray:
    """For each frame return a (H, W) float32 map where each pixel holds the
    *highest* SAM 3 mask score covering it (0 if no mask above threshold)."""
    _, processor = _load_sam3(device)
    F, H, W, _ = frames.shape
    out = np.zeros((F, H, W), dtype=np.float32)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )

    for i in tqdm(range(F), desc=f"SAM 3 query '{text}'"):
        img = Image.fromarray(frames[i])
        with autocast_ctx:
            state = processor.set_image(img)
            try:
                res = processor.set_text_prompt(state=state, prompt=text)
            except Exception as exc:
                print(f"[Query] frame {i} failed: {exc}")
                continue
        masks = res.get("masks")
        scores = res.get("scores")
        if masks is None or (hasattr(masks, "__len__") and len(masks) == 0):
            continue
        masks = _to_np(masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        scores = (_to_np(scores).reshape(-1).astype(np.float32)
                  if scores is not None else np.ones(len(masks), dtype=np.float32))
        for m, s in zip(masks, scores):
            s = float(s)
            if s < score_threshold:
                continue
            if m.dtype != bool:
                m = (m > 0.5) if float(m.max()) <= 1.0 else (m > 127)
            if m.shape != (H, W):
                m = np.asarray(
                    Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.Resampling.NEAREST)
                ) > 0
            np.maximum(out[i], np.where(m, s, 0.0), out=out[i])
    return out


def query_scene(
    run_dir: str | Path,
    text: str,
    *,
    topk_percent: float = 10.0,
    score_threshold: float = 0.5,
    device: str | None = None,
    save: bool = True,
) -> dict[str, np.ndarray]:
    """Open-vocabulary 3D query: highlight the top K% of fused points that
    fall inside SAM 3's mask for ``text`` in their source frame."""
    run_dir = Path(run_dir)
    cloud = load_fused_cloud(run_dir / "exports" / "fused_points.npz")
    meta = load_frames_meta(run_dir)
    image_paths = list_images(run_dir / "frames")
    if len(image_paths) != meta.num_frames:
        raise RuntimeError(
            f"frames_meta.json says {meta.num_frames} frames, found {len(image_paths)}."
        )

    H, W = meta.height, meta.width
    frames = np.empty((meta.num_frames, H, W, 3), dtype=np.uint8)
    for i, p in enumerate(image_paths):
        img = Image.open(p).convert("RGB")
        if img.size != (W, H):
            img = img.resize((W, H), Image.Resampling.BILINEAR)
        frames[i] = np.asarray(img, dtype=np.uint8)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    score_maps = _sam3_score_maps(frames, text, device=device, score_threshold=score_threshold)
    n_hit_px = int((score_maps > 0).sum())
    print(f"[Query] SAM 3 produced {n_hit_px:,} matching pixels across {len(frames)} frames")

    # Per-point score from each point's source frame + saved-frame pixel.
    scores = score_maps[cloud.source_frame, cloud.source_y, cloud.source_x]

    if topk_percent and 0 < topk_percent < 100 and (scores > 0).any():
        k = max(1, int(round(len(scores) * topk_percent / 100.0)))
        order = np.argsort(-scores)
        top = order[:k]
        selected = np.zeros(len(scores), dtype=bool)
        selected[top[scores[top] > 0]] = True
    else:
        selected = scores > 0

    print(f"[Query] Selected {int(selected.sum()):,} / {len(scores):,} points "
          f"({100.0 * selected.mean():.1f}%)")

    if save:
        exports = ensure_dir(run_dir / "exports")
        slug = "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")[:60] or "query"
        ply_path = exports / f"query_{slug}.ply"
        # Selected -> red; rest -> muted gray, so the highlight stays readable.
        colors = np.full((len(cloud.points), 3), 80, dtype=np.uint8)
        colors[selected] = (220, 30, 30)
        write_ply(ply_path, {"points": cloud.points, "colors": colors})
        print(f"[Query] Wrote {ply_path}")

    return {
        "text": text,
        "selected_mask": selected,
        "scores": scores,
        "points": cloud.points[selected],
        "colors": cloud.colors[selected],
    }