"""Run the spline-controlled minimum-time optimisation for Silverstone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dynamics import default_car
from .geometry import load_track_csv, preprocess_track
from .optimisation import optimise_racing_line
from .plotting import save_optimisation_plot
from .source import download_silverstone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimise a feasible Silverstone racing line.")
    parser.add_argument("--download", action="store_true", help="Download source data before optimising.")
    parser.add_argument("--spacing", type=float, default=5.0, help="Track sample spacing in metres.")
    parser.add_argument("--control-points", type=int, default=80, help="Final spline control-point count.")
    parser.add_argument("--coarse-control-points", type=int, default=20, help="Broad first-stage controls.")
    parser.add_argument("--maxiter", type=int, default=500, help="Final-stage SLSQP safety cap.")
    parser.add_argument("--coarse-maxiter", type=int, default=200, help="First-stage SLSQP safety cap.")
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
    result = optimise_racing_line(
        track,
        default_car(),
        control_count=args.control_points,
        coarse_control_count=args.coarse_control_points,
        max_iterations=args.maxiter,
        coarse_max_iterations=args.coarse_maxiter,
    )
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed / "silverstone_optimised_line.npz",
        centreline=track.centreline,
        optimised_line=result.track.centreline,
        offset_m=result.offset_m,
        control_distance_m=result.control_distance_m,
        control_offset_m=result.control_offset_m,
        speed_mps=result.simulation.speed_mps,
    )
    metrics = {
        "baseline_lap_time_s": round(result.baseline.lap_time_s, 3),
        "optimised_lap_time_s": round(result.simulation.lap_time_s, 3),
        "improvement_s": round(result.lap_time_improvement_s, 3),
        "improvement_percent": round(result.lap_time_improvement_percent, 3),
        "minimum_legal_wheel_clearance_m": round(result.minimum_clearance_m, 3),
        "curb_wheel_fraction": round(result.curb_wheel_fraction, 4),
        "maximum_tyre_utilisation": round(result.simulation.max_tyre_utilisation, 4),
        "control_points": args.control_points,
        "coarse_control_points": result.coarse_control_count,
        "solver_success": result.solver_success,
        "solver_message": result.solver_message,
        "solver_iterations": result.solver_iterations,
    }
    (processed / "silverstone_optimised_line.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    save_optimisation_plot(track, result, root / "output" / "silverstone_optimised_line.png")
    print(
        f"Baseline: {result.baseline.lap_time_s:.3f} s; optimised: {result.simulation.lap_time_s:.3f} s; "
        f"improvement: {result.lap_time_improvement_s:.3f} s ({result.lap_time_improvement_percent:.2f}%)."
    )


if __name__ == "__main__":
    main()
