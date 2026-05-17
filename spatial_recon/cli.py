from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .export import write_cloud_npz, write_glb, write_html_viewer, write_legend, write_ply
from .fusion import fuse_predictions
from .report import write_report
from .semantics import load_label_maps, run_semantics
from .utils import command_string, ensure_dir, list_images
from .vggt_runner import load_predictions, run_vggt
from .video import extract_frames


def _prepare_images(args: argparse.Namespace, out_dir: Path) -> list[Path]:
    frames_dir = ensure_dir(out_dir / "frames")
    if args.video:
        return extract_frames(
            args.video,
            frames_dir,
            max_frames=args.max_frames,
            mode=args.frame_mode,
            image_size=args.image_size,
            overwrite=args.overwrite_frames,
        )

    if not args.images_dir:
        raise ValueError("Provide either --video or --images-dir.")
    images = list_images(args.images_dir)
    if not images:
        raise ValueError(f"No images found in {args.images_dir}")
    for idx, image in enumerate(images):
        dst = frames_dir / f"{idx:04d}{image.suffix.lower()}"
        if args.overwrite_frames or not dst.exists():
            shutil.copy2(image, dst)
    return list_images(frames_dir)


def run_pipeline(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.out)
    image_paths = _prepare_images(args, out_dir)

    predictions_path = out_dir / "predictions_vggt.npz"
    if predictions_path.exists() and not args.overwrite_predictions:
        predictions = load_predictions(predictions_path)
    else:
        predictions = run_vggt(
            out_dir / "frames",
            predictions_path,
            checkpoint=args.checkpoint,
            device=args.device,
        )

    semantics_dir = out_dir / "semantics"
    semantic_model = None
    if not args.no_semantics:
        semantic_model = args.semantic_model
        labels_path = semantics_dir / "labels.json"
        if not labels_path.exists() or args.overwrite_semantics:
            run_semantics(out_dir / "frames", semantics_dir, model_name=args.semantic_model, device=args.device)
        label_maps, labels = load_label_maps(semantics_dir)
    else:
        label_maps, labels = [], {0: "unknown"}

    cloud = fuse_predictions(
        predictions,
        label_maps=label_maps,
        labels=labels,
        conf_percentile=args.conf_percentile,
        sample_stride=args.sample_stride,
        voxel_size=args.voxel_size,
        max_points=args.max_points,
        use_point_map=args.use_point_map,
    )

    exports = ensure_dir(out_dir / "exports")
    write_ply(exports / "reconstruction_rgb.ply", cloud)
    write_ply(exports / "reconstruction_semantic.ply", cloud, semantic_colors=True)
    write_glb(exports / "reconstruction_semantic.glb", cloud, semantic_colors=True)
    write_html_viewer(exports / "viewer.html", cloud, extrinsic=predictions.get("extrinsic"))
    write_legend(exports / "semantic_legend.json", cloud)
    write_cloud_npz(exports / "fused_points.npz", cloud)
    write_report(
        out_dir / "REPORT.md",
        cloud,
        frame_count=len(image_paths),
        video_path=args.video,
        checkpoint=args.checkpoint,
        semantic_model=semantic_model,
        command=command_string(args),
    )

    print(f"Done. Open {exports / 'viewer.html'}")
    print(f"Report: {out_dir / 'REPORT.md'}")


def run_query(args: argparse.Namespace) -> None:
    from .query import query_cloud

    query_cloud(
        args.run,
        args.text,
        out_ply=args.out,
        topk_percent=args.topk_percent,
        device=args.device,
        model_name=args.query_model,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VGGT video-to-3D reconstruction with semantic point projection.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the complete video/images -> VGGT -> semantics -> exports pipeline.")
    run.add_argument("--video", type=str, default=None, help="Input phone video.")
    run.add_argument("--images-dir", type=str, default=None, help="Alternative input directory of frames/images.")
    run.add_argument("--out", type=str, required=True, help="Output run directory.")
    run.add_argument("--max-frames", type=int, default=24)
    run.add_argument("--frame-mode", choices=["hybrid", "uniform"], default="hybrid")
    run.add_argument("--image-size", type=int, default=1280, help="Resize long video edge before VGGT; set 0 to disable.")
    run.add_argument("--checkpoint", type=str, default="facebook/VGGT-1B")
    run.add_argument("--semantic-model", type=str, default="facebook/mask2former-swin-large-ade-semantic")
    run.add_argument("--device", type=str, default="auto")
    run.add_argument("--conf-percentile", type=float, default=35.0)
    run.add_argument("--sample-stride", type=int, default=2)
    run.add_argument("--voxel-size", type=float, default=0.015)
    run.add_argument("--max-points", type=int, default=450_000)
    run.add_argument("--use-point-map", action="store_true", help="Use VGGT point-map branch instead of depth unprojection.")
    run.add_argument("--no-semantics", action="store_true")
    run.add_argument("--overwrite-frames", action="store_true")
    run.add_argument("--overwrite-predictions", action="store_true")
    run.add_argument("--overwrite-semantics", action="store_true")
    run.set_defaults(func=run_pipeline)

    query = sub.add_parser("query", help="Open-vocabulary query over a fused 3D scene.")
    query.add_argument("--run", type=str, required=True, help="Run directory containing frames/ and exports/fused_points.npz.")
    query.add_argument("--text", type=str, required=True, help="Text query, e.g. 'monitor' or 'blue mug'.")
    query.add_argument("--out", type=str, default=None, help="Output highlighted PLY path.")
    query.add_argument("--topk-percent", type=float, default=10.0, help="Percentage of points to highlight.")
    query.add_argument("--device", type=str, default="auto")
    query.add_argument("--query-model", type=str, default="CIDAS/clipseg-rd64-refined")
    query.set_defaults(func=run_query)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "image_size", None) == 0:
        args.image_size = None
    args.func(args)


if __name__ == "__main__":
    main()
