"""Generate a nondominated time-versus-energy frontier on a fixed racing line."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .dynamics import CarModel, LapSimulation
from .energy import EnergyModel, LapEnergy, analyse_lap_energy
from .energy_planning import EnergySpeedPlan, plan_energy_efficient_speed
from .geometry import ProcessedTrack

FloatArray = NDArray[np.float64]


class ParetoError(RuntimeError):
    """Raised when requested trade-off points cannot form a valid frontier."""


@dataclass(frozen=True)
class ParetoPoint:
    """One nondominated time-energy operating point."""

    time_allowance_fraction: float
    simulation: LapSimulation
    energy: LapEnergy
    plan: EnergySpeedPlan | None

    @property
    def lap_time_s(self) -> float:
        return self.simulation.lap_time_s

    @property
    def net_battery_energy_kwh(self) -> float:
        return self.energy.net_battery_energy_kwh

    @property
    def label(self) -> str:
        return "Fastest" if self.time_allowance_fraction == 0.0 else f"+{self.time_allowance_fraction * 100:.0f}% time"


@dataclass(frozen=True)
class EnergyParetoFrontier:
    """Ordered nondominated time-energy operating points for a fixed line."""

    points: tuple[ParetoPoint, ...]
    reference_simulation: LapSimulation
    reference_energy: LapEnergy

    @property
    def fastest_point(self) -> ParetoPoint:
        return self.points[0]

    @property
    def most_efficient_point(self) -> ParetoPoint:
        return min(self.points, key=lambda point: point.energy.net_battery_energy_j)


def generate_energy_pareto_frontier(
    track: ProcessedTrack,
    car: CarModel,
    reference_simulation: LapSimulation,
    *,
    grip_scale: FloatArray | None = None,
    time_allowances: Iterable[float] = (0.0, 0.01, 0.03, 0.05),
    energy_model: EnergyModel = EnergyModel(),
    control_count: int = 24,
    max_iterations: int = 250,
) -> EnergyParetoFrontier:
    """Solve a monotonic, nondominated time-energy trade-off frontier.

    Each non-zero allowance independently minimises battery energy subject to
    ``T <= T_fastest * (1 + allowance)`` from the fastest profile. Every point
    must have its own solver convergence and time-constraint checks; this
    avoids allowing a warm start to choose a weaker local minimum.
    """

    allowances = tuple(sorted(set(float(value) for value in time_allowances)))
    if not allowances or allowances[0] < 0.0:
        raise ParetoError("Time allowances must be a non-empty set of non-negative values.")
    if allowances[0] != 0.0:
        allowances = (0.0, *allowances)
    if not all(np.isfinite(value) for value in allowances):
        raise ParetoError("Time allowances must be finite.")

    reference_energy = analyse_lap_energy(track, reference_simulation, energy_model)
    points: list[ParetoPoint] = [ParetoPoint(0.0, reference_simulation, reference_energy, None)]
    previous_energy = reference_energy.net_battery_energy_j

    for allowance in allowances[1:]:
        plan = plan_energy_efficient_speed(
            track,
            car,
            reference_simulation,
            grip_scale=grip_scale,
            energy_model=energy_model,
            time_allowance_fraction=allowance,
            control_count=control_count,
            max_iterations=max_iterations,
        )
        if plan.energy.net_battery_energy_j >= previous_energy - 1e-4:
            # A slower point with equal-or-higher energy is dominated by the
            # previous point, so it is intentionally excluded from the front.
            continue
        points.append(ParetoPoint(allowance, plan.simulation, plan.energy, plan))
        previous_energy = plan.energy.net_battery_energy_j

    if len(points) < 2:
        raise ParetoError("No energy-saving point was found for the requested allowances.")
    return EnergyParetoFrontier(tuple(points), reference_simulation, reference_energy)
