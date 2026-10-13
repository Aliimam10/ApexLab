"""Auditable energy accounting for an already simulated lap.

This module measures energy; it deliberately does not alter the racing line,
speed profile, throttle, or braking strategy. Those become optimisation
variables in the next energy-focused slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .dynamics import LapSimulation, drag_n, rolling_resistance_n
from .geometry import ProcessedTrack

FloatArray = NDArray[np.float64]
JOULES_PER_KWH = 3_600_000.0


class EnergyError(ValueError):
    """Raised when energy inputs or the accounting result are invalid."""


@dataclass(frozen=True)
class EnergyModel:
    """Configurable drivetrain and regenerative-braking assumptions.

    These are intentionally generic values for the Formula-style MVP car, not
    a calibration of a particular Formula 1 power unit.  Regeneration is
    capped at motor mechanical power before conversion to stored electricity.
    """

    drive_efficiency: float = 0.94
    regen_efficiency: float = 0.65
    maximum_regen_power_w: float = 150_000.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.drive_efficiency) or not 0.0 < self.drive_efficiency <= 1.0:
            raise EnergyError("Drive efficiency must lie in (0, 1].")
        if not np.isfinite(self.regen_efficiency) or not 0.0 <= self.regen_efficiency <= 1.0:
            raise EnergyError("Regeneration efficiency must lie in [0, 1].")
        if not np.isfinite(self.maximum_regen_power_w) or self.maximum_regen_power_w < 0.0:
            raise EnergyError("Maximum regeneration power must be finite and non-negative.")


@dataclass(frozen=True)
class LapEnergy:
    """Per-segment and whole-lap wheel and battery energy accounting."""

    model: EnergyModel
    tractive_wheel_energy_j: FloatArray
    braking_wheel_energy_j: FloatArray
    battery_draw_energy_j: FloatArray
    recovered_segment_energy_j: FloatArray
    friction_brake_energy_j: FloatArray
    drag_energy_j: FloatArray
    rolling_energy_j: FloatArray
    kinetic_energy_change_j: FloatArray

    @property
    def wheel_energy_j(self) -> float:
        """Positive mechanical energy delivered at the driven wheels."""

        return float(np.sum(self.tractive_wheel_energy_j))

    @property
    def braking_energy_j(self) -> float:
        """Mechanical energy removed by wheel braking, before recovery losses."""

        return float(np.sum(self.braking_wheel_energy_j))

    @property
    def battery_draw_j(self) -> float:
        """Electrical energy drawn from the battery/inverter for propulsion."""

        return float(np.sum(self.battery_draw_energy_j))

    @property
    def recovered_energy_j(self) -> float:
        """Electrical energy returned to the battery by capped regeneration."""

        return float(np.sum(self.recovered_segment_energy_j))

    @property
    def net_battery_energy_j(self) -> float:
        return self.battery_draw_j - self.recovered_energy_j

    @property
    def net_battery_energy_kwh(self) -> float:
        return self.net_battery_energy_j / JOULES_PER_KWH

    @property
    def drag_energy_j_total(self) -> float:
        return float(np.sum(self.drag_energy_j))

    @property
    def rolling_energy_j_total(self) -> float:
        return float(np.sum(self.rolling_energy_j))

    @property
    def friction_brake_energy_j_total(self) -> float:
        return float(np.sum(self.friction_brake_energy_j))

    @property
    def energy_balance_residual_j(self) -> float:
        """Closed-lap mechanical-work residual; near zero indicates consistency."""

        return float(
            self.wheel_energy_j
            - self.braking_energy_j
            - self.drag_energy_j_total
            - self.rolling_energy_j_total
            - np.sum(self.kinetic_energy_change_j)
        )


def analyse_lap_energy(
    track: ProcessedTrack,
    simulation: LapSimulation,
    model: EnergyModel = EnergyModel(),
) -> LapEnergy:
    """Integrate wheel and battery energy over a feasible closed lap.

    For each segment, wheel work is ``F_x ds``.  Positive longitudinal tyre
    force is propulsion; negative force is braking.  Regeneration can recover
    only braking work and is limited by the configured maximum motor power:

    ``E_recovered = eta_regen * min(E_brake, P_regen,max * dt)``.
    """

    count = track.sample_count
    required = (
        simulation.speed_mps,
        simulation.longitudinal_tyre_force_n,
        simulation.segment_time_s,
    )
    if any(np.asarray(values).shape != (count,) for values in required):
        raise EnergyError("Energy analysis requires simulation arrays matching the track samples.")
    if np.any(simulation.segment_time_s <= 0.0) or not np.all(np.isfinite(simulation.segment_time_s)):
        raise EnergyError("Lap segment times must be finite and positive.")

    force = np.asarray(simulation.longitudinal_tyre_force_n, dtype=np.float64)
    distance = np.asarray(track.segment_length, dtype=np.float64)
    segment_time = np.asarray(simulation.segment_time_s, dtype=np.float64)
    tractive_work = np.maximum(force, 0.0) * distance
    braking_work = np.maximum(-force, 0.0) * distance
    max_regen_work = model.maximum_regen_power_w * segment_time
    regenerated_mechanical_work = np.minimum(braking_work, max_regen_work)
    recovered = regenerated_mechanical_work * model.regen_efficiency
    battery_draw = tractive_work / model.drive_efficiency
    friction_brake = braking_work - regenerated_mechanical_work
    drag_work = np.asarray(drag_n(simulation.speed_mps, simulation.car), dtype=np.float64) * distance
    rolling_work = np.asarray(rolling_resistance_n(simulation.speed_mps, simulation.car), dtype=np.float64) * distance
    next_speed = np.roll(simulation.speed_mps, -1)
    kinetic_change = 0.5 * simulation.car.mass_kg * (np.square(next_speed) - np.square(simulation.speed_mps))

    values = (
        tractive_work, braking_work, battery_draw, recovered, friction_brake,
        drag_work, rolling_work, kinetic_change,
    )
    if not all(np.all(np.isfinite(value)) for value in values):
        raise EnergyError("Energy analysis produced non-finite values.")
    if np.any(recovered > battery_draw + braking_work + 1e-8):
        raise EnergyError("Recovered energy exceeds available braking work.")
    return LapEnergy(
        model=model,
        tractive_wheel_energy_j=tractive_work,
        braking_wheel_energy_j=braking_work,
        battery_draw_energy_j=battery_draw,
        recovered_segment_energy_j=recovered,
        friction_brake_energy_j=friction_brake,
        drag_energy_j=drag_work,
        rolling_energy_j=rolling_work,
        kinetic_energy_change_j=kinetic_change,
    )
