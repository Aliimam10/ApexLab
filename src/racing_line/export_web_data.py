"""Export a verified Silverstone simulation as browser-consumable JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dynamics import default_car
from .energy import analyse_lap_energy
from .geometry import load_track_csv, preprocess_track
from .optimisation import optimise_racing_line
from .source import download_silverstone


def _array(values: np.ndarray) -> list[float] | list[list[float]]:
    return np.round(values, 5).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ApexLab data for the local 2D viewer.")
    parser.add_argument("--download", action="store_true", help="Download Silverstone source data first.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="ApexLab project root.")
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
    track = preprocess_track(points, right_width, left_width)
    result = optimise_racing_line(track, default_car())
    baseline_energy = analyse_lap_energy(track, result.baseline)
    optimised_energy = analyse_lap_energy(result.track, result.simulation)
    payload = {
        "track_name": "Silverstone Grand Prix Circuit",
        "track": {
            "centreline": _array(track.centreline),
            "left_boundary": _array(track.left_boundary),
            "right_boundary": _array(track.right_boundary),
        },
        "baseline": {
            "lap_time_s": result.baseline.lap_time_s,
            "speed_mps": _array(result.baseline.speed_mps),
            "acceleration_mps2": _array(result.baseline.longitudinal_acceleration_mps2),
            "segment_time_s": _array(result.baseline.segment_time_s),
            "energy": {
                "net_battery_energy_kwh": baseline_energy.net_battery_energy_kwh,
                "wheel_energy_kwh": baseline_energy.wheel_energy_j / 3_600_000.0,
                "recovered_energy_kwh": baseline_energy.recovered_energy_j / 3_600_000.0,
            },
        },
        "optimised": {
            "lap_time_s": result.simulation.lap_time_s,
            "line": _array(result.track.centreline),
            "speed_mps": _array(result.simulation.speed_mps),
            "acceleration_mps2": _array(result.simulation.longitudinal_acceleration_mps2),
            "segment_time_s": _array(result.simulation.segment_time_s),
            "energy": {
                "net_battery_energy_kwh": optimised_energy.net_battery_energy_kwh,
                "wheel_energy_kwh": optimised_energy.wheel_energy_j / 3_600_000.0,
                "recovered_energy_kwh": optimised_energy.recovered_energy_j / 3_600_000.0,
            },
        },
        "metrics": {
            "improvement_s": result.lap_time_improvement_s,
            "improvement_percent": result.lap_time_improvement_percent,
            "minimum_clearance_m": result.minimum_clearance_m,
            "curb_wheel_fraction": result.curb_wheel_fraction,
            "maximum_tyre_utilisation": result.simulation.max_tyre_utilisation,
            "control_points": len(result.control_offset_m),
            "coarse_control_points": result.coarse_control_count,
            "boundary_validation_spacing_m": 1.0,
            "solver_success": result.solver_success,
            "solver_status": result.solver_message,
        },
    }
    destination = root / "web" / "data" / "silverstone.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
