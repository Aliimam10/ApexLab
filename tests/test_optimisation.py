from __future__ import annotations

import numpy as np

from racing_line.dynamics import CarModel
from racing_line.geometry import preprocess_track
from racing_line.optimisation import (
    CurbModel,
    VehicleFootprint,
    footprint_report,
    optimise_racing_line,
    racing_line_track,
    validate_racing_line,
)


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

    result = optimise_racing_line(track, low_speed_car(), control_count=8, max_iterations=200)
    footprint = validate_racing_line(track, result.offset_m, 1.2)

    assert result.solver_success
    assert result.simulation.lap_time_s <= result.baseline.lap_time_s
    assert result.lap_time_improvement_s > 0.1
    assert not footprint.all_four_wheels_out
    assert footprint.minimum_clearance_m >= 1.2 - 1e-7
    assert result.simulation.max_tyre_utilisation <= 1.015


def test_wheel_footprint_allows_curb_use_but_detects_four_wheels_out() -> None:
    radius_m, count = 60.0, 120
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    points = np.column_stack((radius_m * np.cos(angles), radius_m * np.sin(angles)))
    track = preprocess_track(points, np.full(count, 7.0), np.full(count, 7.0), spacing_m=5.0)
    vehicle, curb = VehicleFootprint(), CurbModel()

    curb_line = racing_line_track(track, np.full(track.sample_count, 6.2))
    curb_report = footprint_report(track, curb_line, vehicle, curb)
    assert curb_report.curb_wheel_fraction > 0.0
    assert not curb_report.all_four_wheels_out
    assert validate_racing_line(track, np.full(track.sample_count, 6.2), 1.2).minimum_clearance_m >= 1.2

    off_track_line = racing_line_track(track, np.full(track.sample_count, 10.0))
    assert footprint_report(track, off_track_line, vehicle, curb).all_four_wheels_out
