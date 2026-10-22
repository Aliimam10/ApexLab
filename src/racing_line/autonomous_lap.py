"""Run a feedback-controlled autonomous lap on ApexLab's fastest line."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .autonomy import (
    AutonomousLapResult,
    AutonomousSimulationConfig,
    BicycleModel,
    PIDConfig,
    PurePursuitConfig,
    ReferenceTrajectory,
    simulate_autonomous_lap,
)
from .dynamics import default_car
from .geometry import load_track_csv, preprocess_track
from .optimisation import optimise_racing_line
from .plotting import save_autonomous_tracking_plot
from .source import download_silverstone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track ApexLab's fastest line with a feedback-controlled bicycle model.")
    parser.add_argument("--download", action="store_true", help="Download source track data before simulating.")
    parser.add_argument("--spacing", type=float, default=5.0, help="Reference-track sample spacing in metres.")
    parser.add_argument("--time-step", type=float, default=0.02, help="Bicycle integration time step in seconds.")
    parser.add_argument("--max-time", type=float, default=180.0, help="Autonomous run time limit in seconds.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="ApexLab project root.")
    return parser.parse_args()


def write_telemetry_csv(destination: Path, result: AutonomousLapResult) -> None:
    """Write timestamped autonomous telemetry in a tool-neutral CSV format."""

    telemetry = result.telemetry
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "time_s", "x_m", "y_m", "heading_rad", "actual_speed_mps", "target_speed_mps",
        "steering_command_rad", "actual_steering_rad", "acceleration_command_mps2",
        "lateral_error_m", "boundary_clearance_m", "lateral_acceleration_mps2",
        "controller_compute_time_s", "reference_index",
    )
    arrays = tuple(getattr(telemetry, field) for field in fields)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(zip(*arrays, strict=True))


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
    reference = ReferenceTrajectory.from_apexlab(fastest.track, fastest.simulation)
    result = simulate_autonomous_lap(
        reference,
        default_car(),
        bicycle=BicycleModel(wheelbase_m=3.6, maximum_speed_mps=default_car().max_speed_mps),
        pure_pursuit=PurePursuitConfig(base_lookahead_m=12.0, speed_lookahead_s=0.20, maximum_lookahead_m=40.0),
        pid=PIDConfig(proportional_gain=1.2, integral_gain=0.05),
        config=AutonomousSimulationConfig(time_step_s=args.time_step, maximum_time_s=args.max_time),
    )

    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed / "silverstone_autonomous_lap.npz",
        time_s=result.telemetry.time_s,
        x_m=result.telemetry.x_m,
        y_m=result.telemetry.y_m,
        heading_rad=result.telemetry.heading_rad,
        actual_speed_mps=result.telemetry.actual_speed_mps,
        target_speed_mps=result.telemetry.target_speed_mps,
        steering_command_rad=result.telemetry.steering_command_rad,
        actual_steering_rad=result.telemetry.actual_steering_rad,
        acceleration_command_mps2=result.telemetry.acceleration_command_mps2,
        lateral_error_m=result.telemetry.lateral_error_m,
        boundary_clearance_m=result.telemetry.boundary_clearance_m,
        lateral_acceleration_mps2=result.telemetry.lateral_acceleration_mps2,
        reference_x_m=reference.points_m[:, 0],
        reference_y_m=reference.points_m[:, 1],
        reference_speed_mps=reference.target_speed_mps,
    )
    write_telemetry_csv(processed / "silverstone_autonomous_lap.csv", result)
    save_autonomous_tracking_plot(reference, result, root / "output" / "silverstone_autonomous_lap.png")
    summary = {
        "completed_lap": result.completed_lap,
        "failure_reason": result.failure_reason,
        "completion_time_s": round(result.completion_time_s, 3),
        "reference_lap_time_s": round(fastest.simulation.lap_time_s, 3),
        "progress_m": round(result.progress_m, 2),
        "reference_length_m": round(reference.total_length_m, 2),
        "mean_absolute_lateral_error_m": round(result.mean_absolute_lateral_error_m, 4),
        "rms_lateral_error_m": round(result.rms_lateral_error_m, 4),
        "maximum_lateral_error_m": round(result.maximum_lateral_error_m, 4),
        "boundary_violations": result.boundary_violations,
        "friction_violations": result.friction_violations,
        "mean_controller_compute_time_ms": round(float(np.mean(result.telemetry.controller_compute_time_s) * 1_000.0), 5),
    }
    destination = processed / "silverstone_autonomous_lap.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    status = "completed" if result.completed_lap else f"failed ({result.failure_reason})"
    print(
        f"Autonomous lap {status}: {result.completion_time_s:.3f} s, "
        f"MAE {result.mean_absolute_lateral_error_m:.3f} m, "
        f"boundary violations {result.boundary_violations}."
    )


if __name__ == "__main__":
    main()
