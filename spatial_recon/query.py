from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from .utils import ensure_dir, list_images


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "query"


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_clipseg(model_name: str, device: str):
    try:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    except ImportError as exc:
        raise ImportError("Open-vocabulary querying needs transformers. Install requirements first.") from exc

    processor = CLIPSegProcessor.from_pretrained(model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def _frame_heatmap(image: Image.Image, text: str, processor, model, device: str) -> np.ndarray:
    import torch

    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        heat = torch.sigmoid(logits).detach().cpu().numpy().squeeze()
    heat_img = Image.fromarray(heat.astype(np.float32)).resize(image.size, Image.Resampling.BILINEAR)
    return np.asarray(heat_img, dtype=np.float32)


def _write_query_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    semantic_ids: np.ndarray,
    scores: np.ndarray,
    selected: np.ndarray,
) -> None:
    ensure_dir(path.parent)
    out_colors = (colors.astype(np.float32) * 0.25 + 105).clip(0, 180).astype(np.uint8)
    out_colors[selected] = np.asarray([255, 40, 35], dtype=np.uint8)
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int semantic_id\n")
        f.write("property float query_score\n")
        f.write("end_header\n")
        for point, color, label_id, score in zip(points, out_colors, semantic_ids, scores):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {int(label_id)} {float(score):.6f}\n"
            )


def query_cloud(
    run_dir: str | Path,
    text: str,
    out_ply: str | Path | None = None,
    topk_percent: float = 10.0,
    device: str = "auto",
    model_name: str = "CIDAS/clipseg-rd64-refined",
) -> tuple[Path, Path]:
    """Highlight 3D points whose source pixels match an open-vocabulary text query."""
    run_dir = Path(run_dir)
    npz_path = run_dir / "exports" / "fused_points.npz"
    frames_dir = run_dir / "frames"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing fused point export: {npz_path}")
    frame_paths = list_images(frames_dir)
    if not frame_paths:
        raise FileNotFoundError(f"Missing frame images: {frames_dir}")

    data = np.load(npz_path, allow_pickle=True)
    points = data["points"]
    colors = data["colors_rgb"]
    semantic_ids = data["semantic_ids"]
    source_frame = data["source_frame"].astype(np.int32)
    source_y = data["source_y"].astype(np.int32)
    source_x = data["source_x"].astype(np.int32)

    resolved_device = _resolve_device(device)
    print(f"[Query] Using {resolved_device} with {model_name}")
    processor, model = _load_clipseg(model_name, resolved_device)

    scores = np.zeros(len(points), dtype=np.float32)
    for frame_idx, frame_path in enumerate(frame_paths):
        point_mask = source_frame == frame_idx
        if not np.any(point_mask):
            continue
        image = Image.open(frame_path).convert("RGB")
        heatmap = _frame_heatmap(image, text, processor, model, resolved_device)
        y = np.clip(source_y[point_mask], 0, heatmap.shape[0] - 1)
        x = np.clip(source_x[point_mask], 0, heatmap.shape[1] - 1)
        scores[point_mask] = heatmap[y, x]

    fraction = max(0.0, min(100.0, topk_percent)) / 100.0
    k = max(1, int(np.ceil(len(scores) * fraction)))
    top_idx = np.argpartition(scores, -k)[-k:]
    selected = np.zeros(len(scores), dtype=bool)
    selected[top_idx] = True
    cutoff = float(scores[top_idx].min())
    if out_ply is None:
        out_ply = run_dir / "exports" / f"query_{slugify(text)}.ply"
    out_ply = Path(out_ply)
    _write_query_ply(out_ply, points, colors, semantic_ids, scores, selected)

    score_path = out_ply.with_suffix(".npz")
    np.savez_compressed(score_path, query=text, scores=scores, selected=selected, threshold=cutoff)
    print(f"[Query] Highlighted {int(selected.sum())}/{len(points)} points for query '{text}'")
    print(f"[Query] Wrote {out_ply}")
    return out_ply, score_path
