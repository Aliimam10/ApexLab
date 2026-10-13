"""Generate fixed-line time-energy Pareto points for Silverstone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dynamics import default_car
from .energy_pareto import generate_energy_pareto_frontier
from .geometry import load_track_csv, preprocess_track
from .optimisation import CurbModel, VehicleFootprint, footprint_report, optimise_racing_line
from .plotting import save_energy_pareto_plot
from .source import download_silverstone


def parse_allowances(value: str) -> tuple[float, ...]:
    try:
        percentages = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Allowances must be comma-separated percentages.") from error
    if not percentages or any(percent < 0.0 for percent in percentages):
        raise argparse.ArgumentTypeError("Allowances must contain non-negative percentages.")
    return tuple(percent / 100.0 for percent in percentages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Silverstone's fixed-line time-energy Pareto frontier.")
    parser.add_argument("--download", action="store_true", help="Download source data before planning.")
    parser.add_argument("--spacing", type=float, default=5.0, help="Track sample spacing in metres.")
    parser.add_argument("--allowances", type=parse_allowances, default=parse_allowances("0,1,3,5"), help="Comma-separated lap-time allowances in percent.")
    parser.add_argument("--control-points", type=int, default=24, help="Smooth speed-ceiling control count.")
    parser.add_argument("--maxiter", type=int, default=250, help="Per-point SLSQP safety cap.")
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
    source_track = preprocess_track(points, right_width, left_width, spacing_m=args.spacing)
    fastest = optimise_racing_line(source_track, default_car())
    surface = footprint_report(source_track, fastest.track, VehicleFootprint(), CurbModel())
    frontier = generate_energy_pareto_frontier(
        fastest.track,
        default_car(),
        fastest.simulation,
        grip_scale=surface.grip_scale,
        time_allowances=args.allowances,
        control_count=args.control_points,
        max_iterations=args.maxiter,
    )

    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed / "silverstone_energy_pareto.npz",
        lap_time_s=np.asarray([point.lap_time_s for point in frontier.points]),
        net_battery_energy_j=np.asarray([point.energy.net_battery_energy_j for point in frontier.points]),
        allowances=np.asarray([point.time_allowance_fraction for point in frontier.points]),
        speed_mps=np.vstack([point.simulation.speed_mps for point in frontier.points]),
    )
    payload = {
        "track_name": "Silverstone Grand Prix Circuit",
        "points": [
            {
                "label": point.label,
                "time_allowance_percent": round(point.time_allowance_fraction * 100.0, 3),
                "lap_time_s": round(point.lap_time_s, 3),
                "lap_time_cost_s": round(point.lap_time_s - frontier.fastest_point.lap_time_s, 3),
                "net_battery_energy_kwh": round(point.net_battery_energy_kwh, 5),
                "energy_saved_kwh": round(
                    frontier.fastest_point.energy.net_battery_energy_kwh - point.net_battery_energy_kwh, 5
                ),
                "solver_success": point.plan is None or point.plan.solver_success,
                "solver_iterations": 0 if point.plan is None else point.plan.solver_iterations,
            }
            for point in frontier.points
        ],
    }
    (processed / "silverstone_energy_pareto.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    save_energy_pareto_plot(frontier, root / "output" / "silverstone_energy_pareto.png")
    print(f"Generated {len(frontier.points)} nondominated time-energy points.")
    for point in frontier.points:
        print(f"  {point.label}: {point.lap_time_s:.3f} s, {point.net_battery_energy_kwh:.4f} kWh")


if __name__ == "__main__":
    main()
