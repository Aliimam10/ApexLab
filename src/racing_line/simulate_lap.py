"""Run the verified centreline-lap simulation for Silverstone."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .dynamics import default_car, simulate_centreline_lap
from .geometry import load_track_csv, preprocess_track
from .plotting import save_dynamics_plot
from .source import download_silverstone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a feasible Silverstone centreline lap.")
    parser.add_argument("--download", action="store_true", help="Download source data before simulating.")
    parser.add_argument("--spacing", type=float, default=5.0, help="Track sample spacing in metres.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root containing data/ and output/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    raw_path = root / "data" / "raw" / "Silverstone.csv"
    if args.download:
        download_silverstone(raw_path)
    elif not raw_path.exists():
        raise SystemExit(f"Missing {raw_path}. Run again with --download.")

    points, right_width, left_width = load_track_csv(raw_path)
    track = preprocess_track(points, right_width, left_width, spacing_m=args.spacing)
    simulation = simulate_centreline_lap(track, default_car())

    processed_directory = root / "data" / "processed"
    processed_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed_directory / "silverstone_centreline_lap.npz",
        arc_length=track.arc_length,
        curvature=track.curvature,
        speed_mps=simulation.speed_mps,
        corner_speed_limit_mps=simulation.corner_speed_limit_mps,
        longitudinal_acceleration_mps2=simulation.longitudinal_acceleration_mps2,
        lateral_force_n=simulation.lateral_force_n,
        longitudinal_tyre_force_n=simulation.longitudinal_tyre_force_n,
        tyre_force_limit_n=simulation.tyre_force_limit_n,
        tyre_utilisation=simulation.tyre_utilisation,
        segment_time_s=simulation.segment_time_s,
    )
    metrics = {
        "lap_time_s": round(simulation.lap_time_s, 3),
        "max_speed_kph": round(simulation.max_speed_mps * 3.6, 2),
        "max_tyre_utilisation": round(simulation.max_tyre_utilisation, 4),
        "solver_iterations": simulation.iterations,
        "car": asdict(simulation.car),
    }
    (processed_directory / "silverstone_centreline_lap.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    output_path = root / "output" / "silverstone_centreline_dynamics.png"
    save_dynamics_plot(track, simulation, output_path)
    print(
        f"Centreline lap: {simulation.lap_time_s:.3f} s; "
        f"max speed: {simulation.max_speed_mps * 3.6:.1f} km/h; "
        f"max tyre use: {simulation.max_tyre_utilisation:.3f}."
    )


if __name__ == "__main__":
    main()
