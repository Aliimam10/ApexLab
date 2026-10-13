"""Find a time-constrained, lower-energy speed profile on the fastest line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dynamics import default_car
from .energy_planning import plan_energy_efficient_speed
from .geometry import load_track_csv, preprocess_track
from .optimisation import CurbModel, VehicleFootprint, footprint_report, optimise_racing_line
from .plotting import save_energy_plan_plot
from .source import download_silverstone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan an energy-aware speed profile on Silverstone's fastest line.")
    parser.add_argument("--download", action="store_true", help="Download source data before planning.")
    parser.add_argument("--spacing", type=float, default=5.0, help="Track sample spacing in metres.")
    parser.add_argument("--time-allowance-percent", type=float, default=3.0, help="Allowed lap-time cost versus fastest line.")
    parser.add_argument("--control-points", type=int, default=24, help="Smooth speed-ceiling control count.")
    parser.add_argument("--maxiter", type=int, default=250, help="Energy-planner SLSQP safety cap.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root containing data/ and output/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.time_allowance_percent < 0.0:
        raise SystemExit("--time-allowance-percent cannot be negative.")
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
    plan = plan_energy_efficient_speed(
        fastest.track,
        default_car(),
        fastest.simulation,
        grip_scale=surface.grip_scale,
        time_allowance_fraction=args.time_allowance_percent / 100.0,
        control_count=args.control_points,
        max_iterations=args.maxiter,
    )

    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed / "silverstone_energy_plan.npz",
        reference_speed_mps=plan.reference_simulation.speed_mps,
        energy_speed_mps=plan.simulation.speed_mps,
        speed_limit_mps=plan.speed_limit_mps,
        control_distance_m=plan.control_distance_m,
        control_speed_factor=plan.control_speed_factor,
        reference_net_battery_energy_j=plan.reference_energy.net_battery_energy_j,
        energy_net_battery_energy_j=plan.energy.net_battery_energy_j,
    )
    metrics = {
        "fastest_lap_time_s": round(plan.reference_simulation.lap_time_s, 3),
        "energy_lap_time_s": round(plan.simulation.lap_time_s, 3),
        "target_lap_time_s": round(plan.target_lap_time_s, 3),
        "lap_time_cost_s": round(plan.lap_time_cost_s, 3),
        "fastest_net_battery_energy_kwh": round(plan.reference_energy.net_battery_energy_kwh, 5),
        "energy_net_battery_energy_kwh": round(plan.energy.net_battery_energy_kwh, 5),
        "energy_saved_kwh": round(plan.energy_saved_j / 3_600_000.0, 5),
        "energy_saved_percent": round(plan.energy_saved_percent, 3),
        "time_allowance_percent": args.time_allowance_percent,
        "control_points": args.control_points,
        "solver_success": plan.solver_success,
        "solver_message": plan.solver_message,
        "solver_iterations": plan.solver_iterations,
    }
    (processed / "silverstone_energy_plan.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    save_energy_plan_plot(fastest.track, plan, root / "output" / "silverstone_energy_plan.png")
    print(
        f"Fastest: {plan.reference_simulation.lap_time_s:.3f} s, {plan.reference_energy.net_battery_energy_kwh:.4f} kWh; "
        f"energy plan: {plan.simulation.lap_time_s:.3f} s, {plan.energy.net_battery_energy_kwh:.4f} kWh; "
        f"saved: {plan.energy_saved_percent:.2f}%."
    )


if __name__ == "__main__":
    main()
