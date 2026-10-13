from __future__ import annotations

import numpy as np

from racing_line.dynamics import CarModel
from racing_line.geometry import preprocess_track
from racing_line.optimisation import boundary_clearance_m, optimise_racing_line


def low_speed_car() -> CarModel:
    return CarModel(
        mass_kg=1_000.0, max_power_w=120_000.0, max_drive_force_n=5_000.0,
        max_brake_force_n=12_000.0, tyre_friction_coefficient=1.1,
        drag_area_m2=0.0, downforce_area_m2=0.0,
        rolling_resistance_coefficient=0.01, max_speed_mps=60.0,
    )


def test_optimised_circle_is_faster_and_within_boundaries() -> None:
    radius_m, count = 60.0, 120
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    points = np.column_stack((radius_m * np.cos(angles), radius_m * np.sin(angles)))
    track = preprocess_track(points, np.full(count, 8.0), np.full(count, 8.0), spacing_m=5.0)

    result = optimise_racing_line(track, low_speed_car(), control_count=8, max_iterations=25)

    assert result.simulation.lap_time_s <= result.baseline.lap_time_s
    assert result.lap_time_improvement_s > 0.1
    assert np.min(boundary_clearance_m(track, result.offset_m)) >= result.minimum_clearance_m - 1e-7
    assert result.simulation.max_tyre_utilisation <= 1.015
