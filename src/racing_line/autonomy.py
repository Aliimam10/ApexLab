"""Closed-loop autonomous tracking for an ApexLab reference trajectory.

This module deliberately complements rather than replaces the quasi-static
lap-time solver.  ``LapSimulation`` supplies a feasible reference speed and
``ProcessedTrack`` supplies a path and boundaries; the kinematic bicycle model
then answers whether a feedback controller can actually follow that reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from .dynamics import (
    CarModel,
    LapSimulation,
    brake_deceleration_mps2,
    corner_speed_limit_mps,
    drive_acceleration_mps2,
    tyre_force_limit_n,
)
from .geometry import ProcessedTrack

FloatArray = NDArray[np.float64]


class AutonomyError(ValueError):
    """Raised when an autonomous-driving configuration is not meaningful."""


def _wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class VehicleState:
    """Planar kinematic state used by the tracking simulator."""

    x_m: float
    y_m: float
    heading_rad: float
    speed_mps: float
    steering_rad: float = 0.0

    def __post_init__(self) -> None:
        values = (self.x_m, self.y_m, self.heading_rad, self.speed_mps, self.steering_rad)
        if not all(np.isfinite(values)) or self.speed_mps < 0.0:
            raise AutonomyError("Vehicle state must be finite with non-negative speed.")

    @property
    def position_m(self) -> FloatArray:
        return np.asarray((self.x_m, self.y_m), dtype=np.float64)


@dataclass(frozen=True)
class BicycleModel:
    """Kinematic bicycle model with steering magnitude and response limits."""

    wheelbase_m: float = 3.6
    maximum_steering_rad: float = 0.50
    maximum_steering_rate_radps: float = 1.5
    maximum_speed_mps: float = 100.0

    def __post_init__(self) -> None:
        values = (
            self.wheelbase_m,
            self.maximum_steering_rad,
            self.maximum_steering_rate_radps,
            self.maximum_speed_mps,
        )
        if not all(np.isfinite(values)) or any(value <= 0.0 for value in values):
            raise AutonomyError("Bicycle parameters must be finite and strictly positive.")

    def step(
        self,
        state: VehicleState,
        steering_command_rad: float,
        acceleration_mps2: float,
        time_step_s: float,
    ) -> VehicleState:
        """Advance one step using bounded steering and semi-implicit Euler integration."""

        if not np.isfinite(steering_command_rad) or not np.isfinite(acceleration_mps2):
            raise AutonomyError("Control commands must be finite.")
        if not np.isfinite(time_step_s) or time_step_s <= 0.0:
            raise AutonomyError("Time step must be finite and positive.")
        desired_steering = float(
            np.clip(steering_command_rad, -self.maximum_steering_rad, self.maximum_steering_rad)
        )
        maximum_change = self.maximum_steering_rate_radps * time_step_s
        steering = state.steering_rad + float(
            np.clip(desired_steering - state.steering_rad, -maximum_change, maximum_change)
        )
        speed = float(np.clip(state.speed_mps + acceleration_mps2 * time_step_s, 0.0, self.maximum_speed_mps))
        yaw_rate = speed * np.tan(steering) / self.wheelbase_m
        heading = _wrap_angle(state.heading_rad + yaw_rate * time_step_s)
        return VehicleState(
            x_m=state.x_m + speed * np.cos(heading) * time_step_s,
            y_m=state.y_m + speed * np.sin(heading) * time_step_s,
            heading_rad=heading,
            speed_mps=speed,
            steering_rad=steering,
        )


@dataclass(frozen=True)
class ReferenceTrajectory:
    """Closed ApexLab path, speed profile, headings, and track limits."""

    points_m: FloatArray
    heading_rad: FloatArray
    target_speed_mps: FloatArray
    arc_length_m: FloatArray
    segment_length_m: FloatArray
    right_width_m: FloatArray
    left_width_m: FloatArray

    def __post_init__(self) -> None:
        count = self.points_m.shape[0]
        if self.points_m.ndim != 2 or self.points_m.shape[1] != 2 or count < 3:
            raise AutonomyError("Reference trajectory needs at least three planar points.")
        fields = (
            self.heading_rad,
            self.target_speed_mps,
            self.arc_length_m,
            self.segment_length_m,
            self.right_width_m,
            self.left_width_m,
        )
        if any(np.asarray(field).shape != (count,) for field in fields):
            raise AutonomyError("Reference trajectory arrays must have matching lengths.")
        if not all(np.all(np.isfinite(field)) for field in (self.points_m, *fields)):
            raise AutonomyError("Reference trajectory must contain finite values.")
        if np.any(self.segment_length_m <= 0.0) or np.any(self.target_speed_mps < 0.0):
            raise AutonomyError("Reference segments must be positive and target speeds non-negative.")
        if np.any(self.right_width_m <= 0.0) or np.any(self.left_width_m <= 0.0):
            raise AutonomyError("Reference boundaries must be positive.")

    @classmethod
    def from_apexlab(
        cls, track: ProcessedTrack, lap: LapSimulation, *, legal_curb_width_m: float = 1.5
    ) -> "ReferenceTrajectory":
        """Adapt an existing ApexLab path and feasible speed profile.

        ``legal_curb_width_m`` matches ApexLab's documented default legal curb
        strip. It keeps feedback boundary validation consistent with the
        optimiser, whose fastest line may deliberately use that legal surface.
        """

        if lap.speed_mps.shape != (track.sample_count,):
            raise AutonomyError("Lap speed profile must match the reference track.")
        if not np.isfinite(legal_curb_width_m) or legal_curb_width_m < 0.0:
            raise AutonomyError("Legal curb width must be finite and non-negative.")
        # An optimised racing line retains the original physical boundaries but
        # moves its centreline. Project those physical boundaries onto the new
        # line normal; using the source widths directly would be incorrect once
        # the reference path is offset from the source centreline.
        right_normal = np.column_stack((np.sin(track.heading), -np.cos(track.heading)))
        right_width = np.einsum("ij,ij->i", track.right_boundary - track.centreline, right_normal)
        left_width = -np.einsum("ij,ij->i", track.left_boundary - track.centreline, right_normal)
        if np.any(right_width <= 0.0) or np.any(left_width <= 0.0):
            raise AutonomyError("Reference path does not have usable projected track boundaries.")
        return cls(
            points_m=np.asarray(track.centreline, dtype=np.float64),
            heading_rad=np.asarray(track.heading, dtype=np.float64),
            target_speed_mps=np.asarray(lap.speed_mps, dtype=np.float64),
            arc_length_m=np.asarray(track.arc_length, dtype=np.float64),
            segment_length_m=np.asarray(track.segment_length, dtype=np.float64),
            right_width_m=np.asarray(right_width + legal_curb_width_m, dtype=np.float64),
            left_width_m=np.asarray(left_width + legal_curb_width_m, dtype=np.float64),
        )

    @property
    def sample_count(self) -> int:
        return int(self.points_m.shape[0])

    @property
    def total_length_m(self) -> float:
        return float(np.sum(self.segment_length_m))

    def right_normal(self, index: int) -> FloatArray:
        heading = self.heading_rad[index % self.sample_count]
        return np.asarray((np.sin(heading), -np.cos(heading)), dtype=np.float64)

    def nearest_index(
        self,
        position_m: FloatArray,
        *,
        previous_index: int | None = None,
        forward_window: int = 100,
    ) -> int:
        """Find a relevant nearest point without losing closed-loop progress."""

        point = np.asarray(position_m, dtype=np.float64)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise AutonomyError("Vehicle position must be a finite two-dimensional point.")
        if previous_index is None:
            candidates = np.arange(self.sample_count)
        else:
            if forward_window < 1:
                raise AutonomyError("Forward waypoint window must be positive.")
            # A small backwards allowance handles controller overshoot; limiting
            # the forward search avoids a wrong branch at a close track crossing.
            offsets = np.arange(-3, forward_window + 1)
            candidates = (int(previous_index) + offsets) % self.sample_count
        squared_distance = np.sum(np.square(self.points_m[candidates] - point), axis=1)
        return int(candidates[int(np.argmin(squared_distance))])

    def advance_index(self, start_index: int, distance_m: float) -> int:
        """Advance around the periodic trajectory by at least ``distance_m``."""

        if not np.isfinite(distance_m) or distance_m < 0.0:
            raise AutonomyError("Lookahead distance must be finite and non-negative.")
        travelled = 0.0
        index = int(start_index) % self.sample_count
        for _ in range(self.sample_count):
            if travelled >= distance_m:
                return index
            travelled += self.segment_length_m[index]
            index = (index + 1) % self.sample_count
        return index

    def minimum_speed_ahead(self, start_index: int, distance_m: float) -> float:
        """Return the lowest reference speed within a forward preview distance."""

        if not np.isfinite(distance_m) or distance_m < 0.0:
            raise AutonomyError("Preview distance must be finite and non-negative.")
        index = int(start_index) % self.sample_count
        minimum = float(self.target_speed_mps[index])
        travelled = 0.0
        for _ in range(self.sample_count):
            if travelled >= distance_m:
                break
            travelled += self.segment_length_m[index]
            index = (index + 1) % self.sample_count
            minimum = min(minimum, float(self.target_speed_mps[index]))
        return minimum

    def signed_lateral_error_m(self, position_m: FloatArray, index: int) -> float:
        displacement = np.asarray(position_m, dtype=np.float64) - self.points_m[index % self.sample_count]
        return float(np.dot(displacement, self.right_normal(index)))

    def boundary_clearance_m(self, position_m: FloatArray, index: int) -> float:
        lateral = self.signed_lateral_error_m(position_m, index)
        if lateral >= 0.0:
            return float(self.right_width_m[index % self.sample_count] - lateral)
        return float(self.left_width_m[index % self.sample_count] + lateral)


@dataclass(frozen=True)
class PurePursuitConfig:
    """Pure Pursuit steering settings in SI units."""

    base_lookahead_m: float = 12.0
    speed_lookahead_s: float = 0.20
    minimum_lookahead_m: float = 5.0
    maximum_lookahead_m: float = 40.0

    def __post_init__(self) -> None:
        values = (
            self.base_lookahead_m,
            self.speed_lookahead_s,
            self.minimum_lookahead_m,
            self.maximum_lookahead_m,
        )
        if not all(np.isfinite(values)) or self.minimum_lookahead_m <= 0.0:
            raise AutonomyError("Pure Pursuit settings must be finite with positive minimum lookahead.")
        if self.maximum_lookahead_m < self.minimum_lookahead_m or self.base_lookahead_m < 0.0:
            raise AutonomyError("Pure Pursuit lookahead limits are inconsistent.")


@dataclass(frozen=True)
class PIDConfig:
    """PID gains for longitudinal acceleration control."""

    proportional_gain: float = 1.2
    integral_gain: float = 0.05
    derivative_gain: float = 0.0

    def __post_init__(self) -> None:
        values = (self.proportional_gain, self.integral_gain, self.derivative_gain)
        if not all(np.isfinite(values)) or any(value < 0.0 for value in values):
            raise AutonomyError("PID gains must be finite and non-negative.")


class PurePursuitController:
    """Closed-loop geometric steering controller for a periodic reference path."""

    def __init__(self, trajectory: ReferenceTrajectory, bicycle: BicycleModel, config: PurePursuitConfig = PurePursuitConfig()) -> None:
        self.trajectory = trajectory
        self.bicycle = bicycle
        self.config = config
        self._previous_index: int | None = None

    def reset(self) -> None:
        self._previous_index = None

    def command(self, state: VehicleState) -> tuple[float, int, int, float]:
        """Return steering command, nearest index, target index, and lateral error."""

        nearest = self.trajectory.nearest_index(state.position_m, previous_index=self._previous_index)
        self._previous_index = nearest
        lookahead = float(
            np.clip(
                self.config.base_lookahead_m + self.config.speed_lookahead_s * state.speed_mps,
                self.config.minimum_lookahead_m,
                self.config.maximum_lookahead_m,
            )
        )
        target = self.trajectory.advance_index(nearest, lookahead)
        delta = self.trajectory.points_m[target] - state.position_m
        distance = float(np.linalg.norm(delta))
        if distance < 1e-8:
            target = (target + 1) % self.trajectory.sample_count
            delta = self.trajectory.points_m[target] - state.position_m
            distance = float(np.linalg.norm(delta))
        bearing = float(np.arctan2(delta[1], delta[0]))
        alpha = _wrap_angle(bearing - state.heading_rad)
        steering = float(np.arctan2(2.0 * self.bicycle.wheelbase_m * np.sin(alpha), distance))
        steering = float(np.clip(steering, -self.bicycle.maximum_steering_rad, self.bicycle.maximum_steering_rad))
        return steering, nearest, target, self.trajectory.signed_lateral_error_m(state.position_m, nearest)


class PIDSpeedController:
    """Bounded PID controller with conditional-integration anti-windup."""

    def __init__(self, config: PIDConfig = PIDConfig()) -> None:
        self.config = config
        self._integral = 0.0
        self._previous_error: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._previous_error = None

    def command(
        self,
        target_speed_mps: float,
        actual_speed_mps: float,
        time_step_s: float,
        *,
        minimum_acceleration_mps2: float,
        maximum_acceleration_mps2: float,
    ) -> float:
        """Return a bounded acceleration command in m/s²."""

        if not np.isfinite(time_step_s) or time_step_s <= 0.0:
            raise AutonomyError("PID time step must be finite and positive.")
        if minimum_acceleration_mps2 >= maximum_acceleration_mps2:
            raise AutonomyError("PID acceleration bounds are inconsistent.")
        error = float(target_speed_mps - actual_speed_mps)
        derivative = 0.0 if self._previous_error is None else (error - self._previous_error) / time_step_s
        candidate_integral = self._integral + error * time_step_s
        candidate = (
            self.config.proportional_gain * error
            + self.config.integral_gain * candidate_integral
            + self.config.derivative_gain * derivative
        )
        bounded = float(np.clip(candidate, minimum_acceleration_mps2, maximum_acceleration_mps2))
        # Integrate only if unsaturated or if the error drives the command back
        # towards the allowed interval.
        if bounded == candidate or (bounded == maximum_acceleration_mps2 and error < 0.0) or (
            bounded == minimum_acceleration_mps2 and error > 0.0
        ):
            self._integral = candidate_integral
        self._previous_error = error
        return bounded


@dataclass(frozen=True)
class AutonomousSimulationConfig:
    """Numerical and safety limits for one feedback-controlled lap."""

    time_step_s: float = 0.02
    maximum_time_s: float = 180.0
    boundary_margin_m: float = 0.0
    terminate_on_boundary_violation: bool = True
    braking_preview_margin_m: float = 10.0
    lateral_speed_safety_factor: float = 0.97

    def __post_init__(self) -> None:
        if not np.isfinite(self.time_step_s) or self.time_step_s <= 0.0:
            raise AutonomyError("Autonomy time step must be finite and positive.")
        if not np.isfinite(self.maximum_time_s) or self.maximum_time_s <= self.time_step_s:
            raise AutonomyError("Maximum simulation time must exceed one time step.")
        if not np.isfinite(self.boundary_margin_m) or self.boundary_margin_m < 0.0:
            raise AutonomyError("Boundary margin must be finite and non-negative.")
        if not np.isfinite(self.braking_preview_margin_m) or self.braking_preview_margin_m < 0.0:
            raise AutonomyError("Braking preview margin must be finite and non-negative.")
        if not np.isfinite(self.lateral_speed_safety_factor) or not 0.0 < self.lateral_speed_safety_factor <= 1.0:
            raise AutonomyError("Lateral speed safety factor must lie in (0, 1].")


@dataclass(frozen=True)
class AutonomousTelemetry:
    """Timestamped true-state, controller-command, and safety traces."""

    time_s: FloatArray
    x_m: FloatArray
    y_m: FloatArray
    heading_rad: FloatArray
    actual_speed_mps: FloatArray
    target_speed_mps: FloatArray
    steering_command_rad: FloatArray
    actual_steering_rad: FloatArray
    acceleration_command_mps2: FloatArray
    lateral_error_m: FloatArray
    boundary_clearance_m: FloatArray
    lateral_acceleration_mps2: FloatArray
    controller_compute_time_s: FloatArray
    reference_index: NDArray[np.int64]


@dataclass(frozen=True)
class AutonomousLapResult:
    """Outcome and telemetry from a closed-loop autonomous attempt."""

    telemetry: AutonomousTelemetry
    completed_lap: bool
    failure_reason: str | None
    progress_m: float
    boundary_violations: int
    friction_violations: int

    @property
    def completion_time_s(self) -> float:
        return float(self.telemetry.time_s[-1])

    @property
    def mean_absolute_lateral_error_m(self) -> float:
        return float(np.mean(np.abs(self.telemetry.lateral_error_m)))

    @property
    def rms_lateral_error_m(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.telemetry.lateral_error_m))))

    @property
    def maximum_lateral_error_m(self) -> float:
        return float(np.max(np.abs(self.telemetry.lateral_error_m)))


def simulate_autonomous_lap(
    trajectory: ReferenceTrajectory,
    car: CarModel,
    *,
    bicycle: BicycleModel | None = None,
    pure_pursuit: PurePursuitConfig = PurePursuitConfig(),
    pid: PIDConfig = PIDConfig(),
    config: AutonomousSimulationConfig = AutonomousSimulationConfig(),
    initial_state: VehicleState | None = None,
) -> AutonomousLapResult:
    """Track an ApexLab reference path with feedback-based steering and speed.

    The target profile comes from ApexLab's existing force-feasible speed
    planner.  Each PID output is clipped again by the existing drive/brake and
    tyre-force functions using the bicycle's instantaneous curvature.
    """

    bicycle = bicycle or BicycleModel(maximum_speed_mps=car.max_speed_mps)
    steering = PurePursuitController(trajectory, bicycle, pure_pursuit)
    speed = PIDSpeedController(pid)
    start = trajectory.points_m[0]
    state = initial_state or VehicleState(
        x_m=float(start[0]),
        y_m=float(start[1]),
        heading_rad=float(trajectory.heading_rad[0]),
        speed_mps=float(trajectory.target_speed_mps[0]),
    )
    steering.reset()
    speed.reset()

    history: dict[str, list[float] | list[int]] = {
        "time": [], "x": [], "y": [], "heading": [], "actual_speed": [], "target_speed": [],
        "steering_command": [], "actual_steering": [], "acceleration_command": [], "lateral_error": [],
        "boundary_clearance": [], "lateral_acceleration": [], "controller_time": [], "reference_index": [],
    }
    progress = 0.0
    previous_arc = float(trajectory.arc_length_m[0])
    boundary_violations = 0
    friction_violations = 0
    completed = False
    failure_reason: str | None = None
    steps = int(np.ceil(config.maximum_time_s / config.time_step_s))

    for step in range(steps):
        now = step * config.time_step_s
        controller_start = perf_counter()
        steering_command, nearest, _, lateral_error = steering.command(state)
        curvature = float(np.tan(state.steering_rad) / bicycle.wheelbase_m)
        maximum_drive = drive_acceleration_mps2(state.speed_mps, curvature, car)
        maximum_brake = brake_deceleration_mps2(state.speed_mps, curvature, car)
        braking_preview_m = state.speed_mps**2 / (2.0 * max(maximum_brake, 1e-6)) + config.braking_preview_margin_m
        target_speed = trajectory.minimum_speed_ahead(nearest, braking_preview_m)
        maximum_steering_change = bicycle.maximum_steering_rate_radps * config.time_step_s
        predicted_steering = state.steering_rad + float(
            np.clip(steering_command - state.steering_rad, -maximum_steering_change, maximum_steering_change)
        )
        predicted_curvature = float(np.tan(predicted_steering) / bicycle.wheelbase_m)
        target_speed = min(
            target_speed,
            config.lateral_speed_safety_factor * corner_speed_limit_mps(predicted_curvature, car),
        )
        acceleration_command = speed.command(
            target_speed,
            state.speed_mps,
            config.time_step_s,
            minimum_acceleration_mps2=-maximum_brake,
            maximum_acceleration_mps2=maximum_drive,
        )
        controller_time = perf_counter() - controller_start
        state = bicycle.step(state, steering_command, acceleration_command, config.time_step_s)
        nearest_after = trajectory.nearest_index(state.position_m, previous_index=nearest)
        clearance = trajectory.boundary_clearance_m(state.position_m, nearest_after)
        actual_curvature = float(np.tan(state.steering_rad) / bicycle.wheelbase_m)
        lateral_acceleration = abs(state.speed_mps**2 * actual_curvature)
        lateral_limit = float(tyre_force_limit_n(state.speed_mps, car) / car.mass_kg)

        current_arc = float(trajectory.arc_length_m[nearest_after])
        forward_delta = (current_arc - previous_arc) % trajectory.total_length_m
        if forward_delta <= 0.5 * trajectory.total_length_m:
            progress += forward_delta
            previous_arc = current_arc

        history["time"].append(now + config.time_step_s)
        history["x"].append(state.x_m)
        history["y"].append(state.y_m)
        history["heading"].append(state.heading_rad)
        history["actual_speed"].append(state.speed_mps)
        history["target_speed"].append(target_speed)
        history["steering_command"].append(steering_command)
        history["actual_steering"].append(state.steering_rad)
        history["acceleration_command"].append(acceleration_command)
        history["lateral_error"].append(trajectory.signed_lateral_error_m(state.position_m, nearest_after))
        history["boundary_clearance"].append(clearance)
        history["lateral_acceleration"].append(lateral_acceleration)
        history["controller_time"].append(controller_time)
        history["reference_index"].append(nearest_after)

        if clearance < config.boundary_margin_m:
            boundary_violations += 1
            if config.terminate_on_boundary_violation:
                failure_reason = "track_boundary_violation"
                break
        if lateral_acceleration > lateral_limit + 1e-6:
            friction_violations += 1
            failure_reason = "lateral_friction_violation"
            break
        if progress >= trajectory.total_length_m - float(np.mean(trajectory.segment_length_m)):
            completed = True
            break
    else:
        failure_reason = "timeout"

    telemetry = AutonomousTelemetry(
        time_s=np.asarray(history["time"], dtype=np.float64),
        x_m=np.asarray(history["x"], dtype=np.float64),
        y_m=np.asarray(history["y"], dtype=np.float64),
        heading_rad=np.asarray(history["heading"], dtype=np.float64),
        actual_speed_mps=np.asarray(history["actual_speed"], dtype=np.float64),
        target_speed_mps=np.asarray(history["target_speed"], dtype=np.float64),
        steering_command_rad=np.asarray(history["steering_command"], dtype=np.float64),
        actual_steering_rad=np.asarray(history["actual_steering"], dtype=np.float64),
        acceleration_command_mps2=np.asarray(history["acceleration_command"], dtype=np.float64),
        lateral_error_m=np.asarray(history["lateral_error"], dtype=np.float64),
        boundary_clearance_m=np.asarray(history["boundary_clearance"], dtype=np.float64),
        lateral_acceleration_mps2=np.asarray(history["lateral_acceleration"], dtype=np.float64),
        controller_compute_time_s=np.asarray(history["controller_time"], dtype=np.float64),
        reference_index=np.asarray(history["reference_index"], dtype=np.int64),
    )
    return AutonomousLapResult(
        telemetry=telemetry,
        completed_lap=completed,
        failure_reason=failure_reason,
        progress_m=progress,
        boundary_violations=boundary_violations,
        friction_violations=friction_violations,
    )
