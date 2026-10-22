from __future__ import annotations

import numpy as np
import pytest

from racing_line.autonomy import (
    AutonomousSimulationConfig,
    BicycleModel,
    PIDConfig,
    PIDSpeedController,
    PurePursuitConfig,
    PurePursuitController,
    ReferenceTrajectory,
    VehicleState,
    simulate_autonomous_lap,
)
from racing_line.dynamics import CarModel
from racing_line.geometry import preprocess_track
from racing_line.autonomous_lap import write_telemetry_csv
from racing_line.plotting import save_autonomous_tracking_plot


def autonomy_car() -> CarModel:
    return CarModel(
        mass_kg=1_000.0,
        max_power_w=200_000.0,
        max_drive_force_n=8_000.0,
        max_brake_force_n=12_000.0,
        tyre_friction_coefficient=1.4,
        drag_area_m2=0.0,
        downforce_area_m2=0.0,
        rolling_resistance_coefficient=0.0,
        max_speed_mps=40.0,
    )


def circle_reference(radius_m: float = 50.0, speed_mps: float = 8.0) -> ReferenceTrajectory:
    angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    points = np.column_stack((radius_m * np.cos(angles), radius_m * np.sin(angles)))
    track = preprocess_track(points, np.full(len(points), 7.0), np.full(len(points), 7.0), spacing_m=2.0)
    return ReferenceTrajectory(
        points_m=track.centreline,
        heading_rad=track.heading,
        target_speed_mps=np.full(track.sample_count, speed_mps),
        arc_length_m=track.arc_length,
        segment_length_m=track.segment_length,
        right_width_m=track.right_width,
        left_width_m=track.left_width,
    )


def test_bicycle_zero_steering_travels_straight() -> None:
    bicycle = BicycleModel(wheelbase_m=2.5, maximum_speed_mps=30.0)
    state = VehicleState(x_m=0.0, y_m=0.0, heading_rad=0.0, speed_mps=10.0)

    next_state = bicycle.step(state, steering_command_rad=0.0, acceleration_mps2=0.0, time_step_s=0.5)

    assert next_state.x_m == pytest.approx(5.0)
    assert next_state.y_m == pytest.approx(0.0)
    assert next_state.heading_rad == pytest.approx(0.0)


def test_pure_pursuit_steers_towards_forward_left_target() -> None:
    reference = circle_reference()
    controller = PurePursuitController(
        reference, BicycleModel(wheelbase_m=2.5), PurePursuitConfig(base_lookahead_m=8.0, maximum_lookahead_m=8.0)
    )
    state = VehicleState(
        x_m=float(reference.points_m[0, 0]),
        y_m=float(reference.points_m[0, 1]),
        heading_rad=float(reference.heading_rad[0]),
        speed_mps=8.0,
    )

    steering, nearest, target, _ = controller.command(state)

    assert nearest == 0
    assert target != nearest
    assert steering > 0.0


def test_pid_output_is_limited_without_integral_windup() -> None:
    controller = PIDSpeedController(PIDConfig(proportional_gain=1.0, integral_gain=1.0))

    for _ in range(30):
        command = controller.command(100.0, 0.0, 0.1, minimum_acceleration_mps2=-3.0, maximum_acceleration_mps2=2.0)
        assert command == pytest.approx(2.0)
    braking = controller.command(0.0, 20.0, 0.1, minimum_acceleration_mps2=-3.0, maximum_acceleration_mps2=2.0)

    assert braking < 0.0


def test_closed_loop_simulation_completes_a_simple_track() -> None:
    reference = circle_reference()
    result = simulate_autonomous_lap(
        reference,
        autonomy_car(),
        bicycle=BicycleModel(wheelbase_m=2.8, maximum_speed_mps=40.0),
        pure_pursuit=PurePursuitConfig(base_lookahead_m=7.0, speed_lookahead_s=0.0, maximum_lookahead_m=7.0),
        config=AutonomousSimulationConfig(time_step_s=0.02, maximum_time_s=55.0),
    )

    assert result.completed_lap
    assert result.failure_reason is None
    assert result.boundary_violations == 0
    assert result.friction_violations == 0
    assert result.maximum_lateral_error_m < 1.0
    assert result.telemetry.time_s.size > 10


def test_autonomy_telemetry_exports_csv_and_plot(tmp_path) -> None:
    reference = circle_reference()
    result = simulate_autonomous_lap(
        reference,
        autonomy_car(),
        bicycle=BicycleModel(wheelbase_m=2.8, maximum_speed_mps=40.0),
        pure_pursuit=PurePursuitConfig(base_lookahead_m=7.0, speed_lookahead_s=0.0, maximum_lookahead_m=7.0),
        config=AutonomousSimulationConfig(time_step_s=0.04, maximum_time_s=55.0),
    )
    csv_path = tmp_path / "telemetry.csv"
    image_path = tmp_path / "tracking.png"

    write_telemetry_csv(csv_path, result)
    save_autonomous_tracking_plot(reference, result, image_path)

    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("time_s,x_m,y_m")
    assert image_path.exists()
    assert image_path.stat().st_size > 1_000
