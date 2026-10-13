"""Export a verified Silverstone simulation as browser-consumable JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dynamics import LapSimulation, default_car
from .energy import LapEnergy, analyse_lap_energy
from .energy_pareto import generate_energy_pareto_frontier
from .geometry import load_track_csv, preprocess_track
from .optimisation import CurbModel, VehicleFootprint, footprint_report, optimise_racing_line
from .source import download_silverstone


def _array(values: np.ndarray) -> list[float] | list[list[float]]:
    return np.round(values, 5).tolist()


def _energy_payload(energy: LapEnergy) -> dict[str, float | list[float]]:
    """Return the energy fields the browser needs for a live replay."""

    # ``LapEnergy`` is intentionally kept out of the public JSON schema: this
    # compact payload is stable and contains both whole-lap and live segments.
    return {
        "net_battery_energy_kwh": energy.net_battery_energy_kwh,
        "wheel_energy_kwh": energy.wheel_energy_j / 3_600_000.0,
        "recovered_energy_kwh": energy.recovered_energy_j / 3_600_000.0,
        "battery_draw_segment_j": _array(energy.battery_draw_energy_j),
        "recovered_segment_j": _array(energy.recovered_segment_energy_j),
        "friction_brake_segment_j": _array(energy.friction_brake_energy_j),
    }


def _simulation_payload(
    simulation: LapSimulation, energy: LapEnergy
) -> dict[str, float | list[float] | dict[str, float | list[float]]]:
    """Serialise one already feasible speed profile."""

    return {
        "lap_time_s": simulation.lap_time_s,
        "speed_mps": _array(simulation.speed_mps),
        "acceleration_mps2": _array(simulation.longitudinal_acceleration_mps2),
        "segment_time_s": _array(simulation.segment_time_s),
        "energy": _energy_payload(energy),
    }


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
    surface = footprint_report(track, result.track, VehicleFootprint(), CurbModel())
    frontier = generate_energy_pareto_frontier(
        result.track,
        default_car(),
        result.simulation,
        grip_scale=surface.grip_scale,
    )
    energy_profiles = {}
    pareto_points = []
    for point in frontier.points:
        profile_id = "optimised" if point.time_allowance_fraction == 0.0 else f"efficient_{point.time_allowance_fraction * 100:.0f}"
        profile = _simulation_payload(point.simulation, point.energy)
        profile.update(
            {
                "id": profile_id,
                "label": point.label,
                "time_allowance_percent": point.time_allowance_fraction * 100.0,
                "energy_saved_kwh": frontier.fastest_point.net_battery_energy_kwh - point.net_battery_energy_kwh,
                "energy_saved_percent": 100.0
                * (frontier.fastest_point.net_battery_energy_kwh - point.net_battery_energy_kwh)
                / frontier.fastest_point.net_battery_energy_kwh,
                "solver_success": point.plan is None or point.plan.solver_success,
                "solver_iterations": 0 if point.plan is None else point.plan.solver_iterations,
            }
        )
        pareto_points.append(profile)
        if profile_id != "optimised":
            energy_profiles[profile_id] = profile
    payload = {
        "track_name": "Silverstone Grand Prix Circuit",
        "track": {
            "centreline": _array(track.centreline),
            "left_boundary": _array(track.left_boundary),
            "right_boundary": _array(track.right_boundary),
        },
        "baseline": _simulation_payload(result.baseline, baseline_energy),
        "optimised": {
            "line": _array(result.track.centreline),
            **_simulation_payload(result.simulation, optimised_energy),
        },
        "energy_profiles": energy_profiles,
        "pareto": {"points": pareto_points},
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
