from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_): return it

from .utils import ensure_dir, list_images, semantic_color, write_json


DEFAULT_CONCEPTS = (
    "wall", "floor", "ceiling",
    "desk", "table", "chair", "sofa", "bed",
    "monitor", "screen", "laptop", "keyboard", "mouse",
    "cup", "mug", "bottle", "book", "phone",
    "lamp", "window", "door", "shelf", "cabinet",
    "plant", "person", "picture", "curtain",
    "rug", "pillow", "box",
)


def _resolve_device(device: str | None) -> str:
    if device in (None, "auto"):
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


def _to_numpy(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _binarize_mask(mask, size_hw: tuple[int, int]) -> np.ndarray:
    arr = _to_numpy(mask)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected SAM 3 mask shape: {arr.shape}")
    if arr.dtype != bool:
        arr = (arr > 127) if arr.max(initial=0) > 1 else (arr > 0.5)
    H, W = size_hw
    if arr.shape != (H, W):
        arr = np.asarray(
            Image.fromarray(arr.astype(np.uint8) * 255).resize((W, H), Image.Resampling.NEAREST)
        ) > 0
    return arr.astype(bool)


def _paint_label_map(label_map: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for lid in np.unique(label_map):
        rgb[label_map == lid] = semantic_color(int(lid))
    return rgb


def run_semantics(
    image_dir: str | Path,
    out_dir: str | Path,
    concepts: Iterable[str] | None = None,
    score_threshold: float = 0.5,
    device: str | None = "auto",
    model_name: str = "sam3",
) -> dict:
    """Run SAM 3 with a fixed list of text concepts and save per-frame label maps.

    Per-pixel label = the concept whose highest-scoring mask covering that pixel
    has the largest score. Pixels with no mask above ``score_threshold`` stay
    at id 0 ("unknown"). Output schema matches what fusion.py expects:
        out_dir/label_maps/<frame>.npy   int16, shape (H, W) at source-frame res
        out_dir/previews/<frame>.png     color preview
        out_dir/labels.json              {labels: {"0": "unknown", "1": ...}, frames: [...]}
    """
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        raise ImportError(
            "SAM 3 is not installed. Run:\n"
            "  git clone https://github.com/facebookresearch/sam3\n"
            "  cd sam3 && pip install -e .\n"
            "And `hf auth login` with an account that has access to facebook/sam3."
        ) from exc

    image_paths = list_images(image_dir)
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    concepts = [c for c in (concepts if concepts is not None else DEFAULT_CONCEPTS) if c]
    resolved_device = _resolve_device(device)
    print(f"[Semantics] SAM 3 on {resolved_device}; {len(concepts)} concepts")

    model = build_sam3_image_model()
    try:
        model = model.to(resolved_device)
    except Exception:
        pass  # some builders place the model themselves

    processor = Sam3Processor(model)

    out_dir = ensure_dir(out_dir)
    masks_dir = ensure_dir(out_dir / "label_maps")
    previews_dir = ensure_dir(out_dir / "previews")

    label_to_id: dict[str, int] = {"unknown": 0}
    for c in concepts:
        key = c.strip().lower().replace(" ", "_")
        if key and key not in label_to_id:
            label_to_id[key] = len(label_to_id)

    frames_meta = []
    for fi, image_path in enumerate(tqdm(image_paths, desc="SAM 3 segmentation")):
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        label_map = np.zeros((H, W), dtype=np.int32)
        score_map = np.zeros((H, W), dtype=np.float32)

        state = processor.set_image(image)
        for concept in concepts:
            lid = label_to_id[concept.strip().lower().replace(" ", "_")]
            try:
                out = processor.set_text_prompt(state=state, prompt=concept)
            except Exception as exc:
                print(f"[Semantics] '{concept}' failed on frame {fi}: {exc}")
                continue

            masks = out.get("masks")
            scores = out.get("scores")
            if masks is None or (hasattr(masks, "__len__") and len(masks) == 0):
                continue

            masks_np = _to_numpy(masks)
            if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                masks_np = masks_np[:, 0]

            if scores is not None:
                scores_np = _to_numpy(scores).reshape(-1).astype(np.float32)
            else:
                scores_np = np.ones(len(masks_np), dtype=np.float32)

            for m, s in zip(masks_np, scores_np):
                s = float(s)
                if s < score_threshold:
                    continue
                bm = _binarize_mask(m, (H, W))
                better = bm & (s > score_map)
                label_map[better] = lid
                score_map[better] = s

        np.save(masks_dir / f"{fi:04d}.npy", label_map.astype(np.int16))
        Image.fromarray(_paint_label_map(label_map)).save(previews_dir / f"{fi:04d}.png")
        frames_meta.append({"image": image_path.name, "label_map": f"label_maps/{fi:04d}.npy"})

    metadata = {
        "model": "sam3",
        "concepts": list(concepts),
        "score_threshold": float(score_threshold),
        "labels": {str(v): k for k, v in label_to_id.items()},
        "frames": frames_meta,
    }
    write_json(out_dir / "labels.json", metadata)
    return metadata


def load_label_maps(semantics_dir: str | Path) -> tuple[list[np.ndarray], dict[int, str]]:
    semantics_dir = Path(semantics_dir)
    meta_path = semantics_dir / "labels.json"
    if not meta_path.exists():
        return [], {0: "unknown"}
    import json
    meta = json.loads(meta_path.read_text())
    labels = {int(k): v for k, v in meta.get("labels", {}).items()}
    maps = [np.load(semantics_dir / f["label_map"]) for f in meta.get("frames", [])]
    return maps, labels