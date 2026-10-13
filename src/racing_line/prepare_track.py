"""Command-line entry point for the first verified project slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .geometry import load_track_csv, preprocess_track
from .plotting import save_track_plot
from .source import SILVERSTONE_SOURCE_URL, download_silverstone, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and validate Silverstone geometry.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the public source CSV before preprocessing.",
    )
    parser.add_argument("--spacing", type=float, default=5.0, help="Uniform sample spacing in metres.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root containing data/ and output/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    raw_path = root / "data" / "raw" / "Silverstone.csv"
    processed_directory = root / "data" / "processed"
    output_path = root / "output" / "silverstone_geometry.png"

    if args.download:
        digest = download_silverstone(raw_path)
        print(f"Downloaded {raw_path} (SHA-256: {digest})")
    elif not raw_path.exists():
        raise SystemExit(f"Missing {raw_path}. Run again with --download.")

    points, right_width, left_width = load_track_csv(raw_path)
    track = preprocess_track(points, right_width, left_width, spacing_m=args.spacing)
    processed_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed_directory / "silverstone_geometry.npz",
        centreline=track.centreline,
        right_boundary=track.right_boundary,
        left_boundary=track.left_boundary,
        right_width=track.right_width,
        left_width=track.left_width,
        arc_length=track.arc_length,
        segment_length=track.segment_length,
        heading=track.heading,
        curvature=track.curvature,
    )
    metadata = {
        "track": "Silverstone Grand Prix Circuit",
        "source_url": SILVERSTONE_SOURCE_URL,
        "source_sha256": file_sha256(raw_path),
        "spacing_m": args.spacing,
        "sample_count": track.sample_count,
        "total_length_m": round(track.total_length, 3),
        "minimum_left_width_m": round(float(np.min(track.left_width)), 3),
        "minimum_right_width_m": round(float(np.min(track.right_width)), 3),
    }
    (processed_directory / "silverstone_geometry.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    save_track_plot(track, output_path)
    print(
        f"Validated {track.sample_count} samples over {track.total_length:.1f} m. "
        f"Plot: {output_path}"
    )


if __name__ == "__main__":
    main()
