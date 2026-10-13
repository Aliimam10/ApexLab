from __future__ import annotations

import numpy as np
import pytest

from racing_line.dynamics import CarModel, corner_speed_limit_mps, simulate_centreline_lap
from racing_line.geometry import ProcessedTrack


def no_aero_car() -> CarModel:
    return CarModel(
        mass_kg=1_000.0,
        max_power_w=200_000.0,
        max_drive_force_n=10_000.0,
        max_brake_force_n=15_000.0,
        tyre_friction_coefficient=1.2,
        drag_area_m2=0.0,
        downforce_area_m2=0.0,
        rolling_resistance_coefficient=0.0,
        max_speed_mps=120.0,
    )


def artificial_track(curvature: np.ndarray, segment_length_m: float = 5.0) -> ProcessedTrack:
    count = len(curvature)
    points = np.column_stack((np.arange(count, dtype=float), np.zeros(count)))
    widths = np.full(count, 8.0)
    return ProcessedTrack(
        centreline=points,
        right_boundary=points + np.array([0.0, -8.0]),
        left_boundary=points + np.array([0.0, 8.0]),
        right_width=widths,
        left_width=widths,
        arc_length=np.arange(count, dtype=float) * segment_length_m,
        segment_length=np.full(count, segment_length_m),
        heading=np.zeros(count),
        curvature=curvature.astype(float),
    )


def test_constant_radius_corner_matches_analytic_no_aero_limit() -> None:
    car = no_aero_car()
    radius_m = 100.0
    expected = np.sqrt(car.tyre_friction_coefficient * 9.80665 * radius_m)

    calculated = corner_speed_limit_mps(1.0 / radius_m, car)

    assert calculated == pytest.approx(expected, rel=0.002)


def test_straight_after_corner_accelerates_and_respects_terminal_speed() -> None:
    car = no_aero_car()
    curvature = np.zeros(240)
    curvature[:3] = 1.0 / 25.0
    track = artificial_track(curvature)

    simulation = simulate_centreline_lap(track, car)

    assert simulation.max_speed_mps <= car.max_speed_mps
    assert np.any(simulation.longitudinal_acceleration_mps2 > 0.0)
    assert simulation.max_tyre_utilisation <= 1.015


def test_braking_is_propagated_before_a_tight_corner() -> None:
    car = no_aero_car()
    curvature = np.zeros(160)
    corner_index = 90
    curvature[corner_index : corner_index + 3] = 1.0 / 20.0
    track = artificial_track(curvature)

    simulation = simulate_centreline_lap(track, car)

    assert simulation.speed_mps[corner_index - 3] > simulation.speed_mps[corner_index - 1]
    assert simulation.longitudinal_acceleration_mps2[corner_index - 3] < 0.0
    assert simulation.speed_mps[corner_index] <= simulation.corner_speed_limit_mps[corner_index] + 1e-6
