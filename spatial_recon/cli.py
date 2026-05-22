from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline
from .query import query_scene


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, help="Run output directory, e.g. runs/desk")
    p.add_argument("--checkpoint", required=True, help="Local VGGT-Omega .pt checkpoint")
    p.add_argument("--image-resolution", type=int, default=512, choices=[256, 512])
    p.add_argument("--max-frames", type=int, default=24)
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--concepts", type=str, default=None,
                   help="Comma-separated SAM 3 text concepts")
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--conf-percentile", type=float, default=35.0)
    p.add_argument("--sample-stride", type=int, default=2)
    p.add_argument("--voxel-size", type=float, default=0.015)
    p.add_argument("--max-points", type=int, default=450_000)
    p.add_argument("--no-semantics", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite-frames", action="store_true")
    p.add_argument("--overwrite-predictions", action="store_true")
    p.add_argument("--overwrite-semantics", action="store_true")


def _add_query_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--topk-percent", type=float, default=10.0)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--device", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatial_recon",
        description="Video → VGGT-Omega → SAM 3 → fused 3D point cloud + text queries.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_run_args(sub.add_parser("run", help="Full pipeline"))
    _add_query_args(sub.add_parser("query", help="Open-vocabulary 3D query"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        concepts = [c.strip() for c in args.concepts.split(",")] if args.concepts else None
        run_pipeline(
            video_path=args.video,
            run_dir=args.out,
            checkpoint=args.checkpoint,
            image_resolution=args.image_resolution,
            max_frames=args.max_frames,
            target_fps=args.fps,
            concepts=concepts,
            score_threshold=args.score_threshold,
            conf_percentile=args.conf_percentile,
            sample_stride=args.sample_stride,
            voxel_size=args.voxel_size,
            max_points=args.max_points,
            no_semantics=args.no_semantics,
            device=args.device,
            overwrite_frames=args.overwrite_frames,
            overwrite_predictions=args.overwrite_predictions,
            overwrite_semantics=args.overwrite_semantics,
        )
    elif args.cmd == "query":
        query_scene(
            args.run, args.text,
            topk_percent=args.topk_percent,
            score_threshold=args.score_threshold,
            device=args.device,
        )


if __name__ == "__main__":
    main()