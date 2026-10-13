"""A deliberately simple, auditable centreline lap-time model.

The model is quasi-static and point-mass based.  It captures the effects that
are needed for the next optimisation commit: curvature-limited corner speed,
combined tyre force, power/drag-limited acceleration, and braking propagated
backwards from the next corner.  It does not attempt to represent suspension,
load transfer, tyre temperature, or F1 hybrid deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .geometry import ProcessedTrack

FloatArray = NDArray[np.float64]
GRAVITY_MPS2 = 9.80665


class DynamicsError(ValueError):
    """Raised when car parameters or a computed lap are physically invalid."""


@dataclass(frozen=True)
class CarModel:
    """Configurable parameters for the simplified point-mass vehicle model."""

    mass_kg: float
    max_power_w: float
    max_drive_force_n: float
    max_brake_force_n: float
    tyre_friction_coefficient: float
    drag_area_m2: float
    downforce_area_m2: float
    rolling_resistance_coefficient: float
    max_speed_mps: float
    air_density_kg_m3: float = 1.225

    def __post_init__(self) -> None:
        strictly_positive = (
            self.mass_kg,
            self.max_power_w,
            self.max_drive_force_n,
            self.max_brake_force_n,
            self.tyre_friction_coefficient,
            self.max_speed_mps,
            self.air_density_kg_m3,
        )
        non_negative = (
            self.drag_area_m2,
            self.downforce_area_m2,
            self.rolling_resistance_coefficient,
        )
        if (
            not all(np.isfinite(strictly_positive + non_negative))
            or any(value <= 0.0 for value in strictly_positive)
            or any(value < 0.0 for value in non_negative)
        ):
            raise DynamicsError(
                "Mass, forces, power, grip, speed, and air density must be positive; "
                "aerodynamic and rolling coefficients cannot be negative."
            )


def default_car() -> CarModel:
    """Return the one configurable, formula-style car used in the MVP."""

    return CarModel(
        mass_kg=850.0,
        max_power_w=600_000.0,
        max_drive_force_n=11_000.0,
        max_brake_force_n=18_000.0,
        tyre_friction_coefficient=1.70,
        drag_area_m2=1.00,
        downforce_area_m2=3.00,
        rolling_resistance_coefficient=0.015,
        max_speed_mps=100.0,
    )


@dataclass(frozen=True)
class LapSimulation:
    """Full centreline speed profile and force checks for one closed lap."""

    car: CarModel
    corner_speed_limit_mps: FloatArray
    speed_mps: FloatArray
    longitudinal_acceleration_mps2: FloatArray
    lateral_force_n: FloatArray
    longitudinal_tyre_force_n: FloatArray
    tyre_force_limit_n: FloatArray
    tyre_utilisation: FloatArray
    segment_time_s: FloatArray
    lap_time_s: float
    iterations: int

    @property
    def max_speed_mps(self) -> float:
        return float(np.max(self.speed_mps))

    @property
    def max_tyre_utilisation(self) -> float:
        return float(np.max(self.tyre_utilisation))


def downforce_n(speed_mps: float | FloatArray, car: CarModel) -> float | FloatArray:
    """Return aerodynamic downforce, positive towards the track surface."""

    return 0.5 * car.air_density_kg_m3 * car.downforce_area_m2 * np.square(speed_mps)


def drag_n(speed_mps: float | FloatArray, car: CarModel) -> float | FloatArray:
    """Return aerodynamic drag opposing travel."""

    return 0.5 * car.air_density_kg_m3 * car.drag_area_m2 * np.square(speed_mps)


def normal_load_n(speed_mps: float | FloatArray, car: CarModel) -> float | FloatArray:
    return car.mass_kg * GRAVITY_MPS2 + downforce_n(speed_mps, car)


def rolling_resistance_n(speed_mps: float | FloatArray, car: CarModel) -> float | FloatArray:
    return car.rolling_resistance_coefficient * normal_load_n(speed_mps, car)


def tyre_force_limit_n(
    speed_mps: float | FloatArray, car: CarModel, grip_scale: float | FloatArray = 1.0
) -> float | FloatArray:
    """Return friction-circle capacity, optionally reduced by surface grip."""

    return car.tyre_friction_coefficient * np.asarray(grip_scale) * normal_load_n(speed_mps, car)


def lateral_force_n(speed_mps: float | FloatArray, curvature_per_m: float | FloatArray, car: CarModel) -> float | FloatArray:
    return car.mass_kg * np.square(speed_mps) * np.abs(curvature_per_m)


def available_longitudinal_tyre_force_n(
    speed_mps: float, curvature_per_m: float, car: CarModel, grip_scale: float = 1.0
) -> float:
    """Remaining friction-circle capacity after supplying lateral force."""

    total_limit = float(tyre_force_limit_n(speed_mps, car, grip_scale))
    lateral = float(lateral_force_n(speed_mps, curvature_per_m, car))
    return float(np.sqrt(max(total_limit**2 - lateral**2, 0.0)))


def corner_speed_limit_mps(
    curvature_per_m: float, car: CarModel, grip_scale: float = 1.0
) -> float:
    """Solve `m v² |k| <= mu (m g + downforce(v))` by bisection."""

    if abs(curvature_per_m) < 1e-10:
        return car.max_speed_mps

    def lateral_margin(speed: float) -> float:
        return float(tyre_force_limit_n(speed, car, grip_scale) - lateral_force_n(speed, curvature_per_m, car))

    if lateral_margin(car.max_speed_mps) >= 0.0:
        return car.max_speed_mps

    low, high = 0.0, car.max_speed_mps
    for _ in range(60):
        midpoint = (low + high) / 2.0
        if lateral_margin(midpoint) >= 0.0:
            low = midpoint
        else:
            high = midpoint
    return low


def drive_acceleration_mps2(
    speed_mps: float, curvature_per_m: float, car: CarModel, grip_scale: float = 1.0
) -> float:
    """Maximum longitudinal acceleration at a speed and path curvature."""

    remaining_tyre_force = available_longitudinal_tyre_force_n(
        speed_mps, curvature_per_m, car, grip_scale
    )
    power_force = car.max_power_w / max(speed_mps, 1.0)
    drive_force = min(car.max_drive_force_n, power_force, remaining_tyre_force)
    resistance = float(drag_n(speed_mps, car) + rolling_resistance_n(speed_mps, car))
    return max((drive_force - resistance) / car.mass_kg, 0.0)


def brake_deceleration_mps2(
    speed_mps: float, curvature_per_m: float, car: CarModel, grip_scale: float = 1.0
) -> float:
    """Maximum positive deceleration magnitude at a speed and curvature."""

    remaining_tyre_force = available_longitudinal_tyre_force_n(
        speed_mps, curvature_per_m, car, grip_scale
    )
    braking_force = min(car.max_brake_force_n, remaining_tyre_force)
    resistance = float(drag_n(speed_mps, car) + rolling_resistance_n(speed_mps, car))
    return (braking_force + resistance) / car.mass_kg


def _forward_reachable_speed(
    speed_mps: float, distance_m: float, curvature_per_m: float, car: CarModel, grip_scale: float
) -> float:
    acceleration = drive_acceleration_mps2(speed_mps, curvature_per_m, car, grip_scale)
    return min(car.max_speed_mps, float(np.sqrt(speed_mps**2 + 2.0 * acceleration * distance_m)))


def _maximum_brakeable_entry_speed(
    exit_speed_mps: float,
    distance_m: float,
    curvature_per_m: float,
    car: CarModel,
    upper_bound_mps: float,
    grip_scale: float,
) -> float:
    """Find the greatest speed that can reach `exit_speed_mps` in one segment.

    For a trial entry speed, the segment uses its maximum available braking
    deceleration. If this brings the car below the requested exit speed, the
    driver can simply use less brake, so that entry speed is feasible.
    """

    def is_feasible(entry_speed: float) -> bool:
        deceleration = brake_deceleration_mps2(entry_speed, curvature_per_m, car, grip_scale)
        minimum_exit_squared = max(entry_speed**2 - 2.0 * deceleration * distance_m, 0.0)
        return minimum_exit_squared <= exit_speed_mps**2 + 1e-9

    low, high = exit_speed_mps, upper_bound_mps
    if is_feasible(high):
        return high
    for _ in range(60):
        midpoint = (low + high) / 2.0
        if is_feasible(midpoint):
            low = midpoint
        else:
            high = midpoint
    return low


def _force_trace(
    track: ProcessedTrack, speed_mps: FloatArray, car: CarModel, grip_scale: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    next_speed = np.roll(speed_mps, -1)
    acceleration = (np.square(next_speed) - np.square(speed_mps)) / (2.0 * track.segment_length)
    lateral = np.asarray(lateral_force_n(speed_mps, track.curvature, car), dtype=np.float64)
    tyre_limit = np.asarray(tyre_force_limit_n(speed_mps, car, grip_scale), dtype=np.float64)
    resistance = np.asarray(drag_n(speed_mps, car) + rolling_resistance_n(speed_mps, car))
    longitudinal_tyre = car.mass_kg * acceleration + resistance
    utilisation = np.sqrt(np.square(longitudinal_tyre) + np.square(lateral)) / tyre_limit
    return acceleration, lateral, longitudinal_tyre, utilisation


def simulate_centreline_lap(
    track: ProcessedTrack,
    car: CarModel,
    *,
    convergence_tolerance_mps: float = 1e-6,
    max_iterations: int = 2_000,
    grip_scale: FloatArray | None = None,
) -> LapSimulation:
    """Calculate the fastest feasible closed-lap speed profile on a centreline.

    The profile starts at the local curvature speed limits. Alternating forward
    acceleration and backward braking sweeps only lower those values until all
    segment-to-segment transitions are feasible. Closed-loop sweeps repeat
    until the start/finish speed has also converged.
    """

    if convergence_tolerance_mps <= 0.0 or max_iterations <= 0:
        raise DynamicsError("Convergence tolerance and iteration count must be positive.")

    if grip_scale is None:
        grip = np.ones(track.sample_count, dtype=np.float64)
    else:
        grip = np.asarray(grip_scale, dtype=np.float64)
        if grip.shape != (track.sample_count,) or not np.all(np.isfinite(grip)) or np.any(grip <= 0.0):
            raise DynamicsError("Surface grip scale must be finite, positive, and match the track samples.")
    corner_limits = np.asarray(
        [corner_speed_limit_mps(curvature, car, scale) for curvature, scale in zip(track.curvature, grip, strict=True)],
        dtype=np.float64,
    )
    speed = corner_limits.copy()
    count = track.sample_count

    for iteration in range(1, max_iterations + 1):
        previous = speed.copy()
        for index in range(count):
            next_index = (index + 1) % count
            reachable = _forward_reachable_speed(
                speed[index], track.segment_length[index], track.curvature[index], car, grip[index]
            )
            speed[next_index] = min(speed[next_index], reachable)

        for index in range(count - 1, -1, -1):
            next_index = (index + 1) % count
            speed[index] = _maximum_brakeable_entry_speed(
                speed[next_index],
                track.segment_length[index],
                track.curvature[index],
                car,
                upper_bound_mps=speed[index],
                grip_scale=grip[index],
            )

        if float(np.max(np.abs(speed - previous))) < convergence_tolerance_mps:
            break
    else:
        raise DynamicsError("Centreline speed profile did not converge.")

    acceleration, lateral, longitudinal_tyre, utilisation = _force_trace(track, speed, car, grip)
    segment_time = 2.0 * track.segment_length / (speed + np.roll(speed, -1))
    simulation = LapSimulation(
        car=car,
        corner_speed_limit_mps=corner_limits,
        speed_mps=speed,
        longitudinal_acceleration_mps2=acceleration,
        lateral_force_n=lateral,
        longitudinal_tyre_force_n=longitudinal_tyre,
        tyre_force_limit_n=np.asarray(tyre_force_limit_n(speed, car, grip), dtype=np.float64),
        tyre_utilisation=utilisation,
        segment_time_s=segment_time,
        lap_time_s=float(np.sum(segment_time)),
        iterations=iteration,
    )
    validate_lap_simulation(track, simulation)
    return simulation


def validate_lap_simulation(
    track: ProcessedTrack, simulation: LapSimulation, *, tolerance: float = 0.015
) -> None:
    """Raise when an output violates the stated point-mass constraints."""

    if tolerance < 0.0:
        raise DynamicsError("Validation tolerance cannot be negative.")
    if not np.all(np.isfinite(simulation.speed_mps)) or np.any(simulation.speed_mps <= 0.0):
        raise DynamicsError("Lap contains a non-positive or non-finite speed.")
    if np.any(simulation.speed_mps > simulation.corner_speed_limit_mps + tolerance):
        raise DynamicsError("Lap exceeds a curvature speed limit.")
    if np.max(simulation.tyre_utilisation) > 1.0 + tolerance:
        raise DynamicsError(
            f"Lap exceeds the friction-circle limit: {np.max(simulation.tyre_utilisation):.3f}."
        )
    expected_time = 2.0 * track.segment_length / (
        simulation.speed_mps + np.roll(simulation.speed_mps, -1)
    )
    if not np.allclose(simulation.segment_time_s, expected_time, atol=1e-10):
        raise DynamicsError("Lap segment times are inconsistent with the speed profile.")
