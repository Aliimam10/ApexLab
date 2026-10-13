"""Fixed-line, time-constrained battery-energy speed planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.optimize import OptimizeResult, minimize

from .dynamics import CarModel, LapSimulation, simulate_centreline_lap
from .energy import EnergyModel, LapEnergy, analyse_lap_energy
from .geometry import ProcessedTrack

FloatArray = NDArray[np.float64]


class EnergyPlanningError(RuntimeError):
    """Raised when no converged, time-feasible energy plan is found."""


@dataclass(frozen=True)
class EnergySpeedPlan:
    """An energy-minimising speed profile on an unchanged racing line."""

    reference_simulation: LapSimulation
    reference_energy: LapEnergy
    simulation: LapSimulation
    energy: LapEnergy
    speed_limit_mps: FloatArray
    control_distance_m: FloatArray
    control_speed_factor: FloatArray
    time_allowance_fraction: float
    solver_success: bool
    solver_message: str
    solver_iterations: int

    @property
    def target_lap_time_s(self) -> float:
        return self.reference_simulation.lap_time_s * (1.0 + self.time_allowance_fraction)

    @property
    def energy_saved_j(self) -> float:
        return self.reference_energy.net_battery_energy_j - self.energy.net_battery_energy_j

    @property
    def energy_saved_percent(self) -> float:
        return 100.0 * self.energy_saved_j / self.reference_energy.net_battery_energy_j

    @property
    def lap_time_cost_s(self) -> float:
        return self.simulation.lap_time_s - self.reference_simulation.lap_time_s


def _control_distances(track: ProcessedTrack, control_count: int) -> FloatArray:
    if control_count < 4:
        raise EnergyPlanningError("Periodic speed planning needs at least four controls.")
    if control_count >= track.sample_count:
        raise EnergyPlanningError("Control count must be smaller than the track sample count.")
    return np.linspace(0.0, track.total_length, control_count, endpoint=False)


def _speed_factors(
    track: ProcessedTrack,
    control_distance: FloatArray,
    controls: FloatArray,
    minimum_speed_factor: float,
) -> FloatArray:
    spline = CubicSpline(
        np.append(control_distance, track.total_length), np.append(controls, controls[0]), bc_type="periodic"
    )
    # Individual controls are bounded, but a cubic spline can overshoot between
    # them. Clipping preserves the physical promise that this planner never
    # requests a speed above the validated minimum-time reference profile.
    return np.clip(np.asarray(spline(track.arc_length), dtype=np.float64), minimum_speed_factor, 1.0)


def plan_energy_efficient_speed(
    track: ProcessedTrack,
    car: CarModel,
    reference_simulation: LapSimulation,
    *,
    grip_scale: FloatArray | None = None,
    energy_model: EnergyModel = EnergyModel(),
    time_allowance_fraction: float = 0.03,
    control_count: int = 24,
    minimum_speed_factor: float = 0.55,
    max_iterations: int = 250,
    initial_speed_factor: FloatArray | None = None,
) -> EnergySpeedPlan:
    """Minimise battery energy while keeping lap time within an allowance.

    The racing line and physical corner/tyre constraints are unchanged. The
    decision variables are sparse speed-ceiling factors. They are expanded by
    a periodic spline and multiplied by the already feasible fastest profile.
    The existing forward/backward dynamics solver then finds the fastest
    physically reachable response beneath that requested ceiling.
    """

    if not np.isfinite(time_allowance_fraction) or time_allowance_fraction < 0.0:
        raise EnergyPlanningError("Time allowance must be finite and non-negative.")
    if not np.isfinite(minimum_speed_factor) or not 0.0 < minimum_speed_factor <= 1.0:
        raise EnergyPlanningError("Minimum speed factor must lie in (0, 1].")
    if max_iterations <= 0:
        raise EnergyPlanningError("Maximum iteration count must be positive.")
    if reference_simulation.speed_mps.shape != (track.sample_count,):
        raise EnergyPlanningError("Reference speed profile must match the fixed line.")

    controls_distance = _control_distances(track, control_count)
    target_time = reference_simulation.lap_time_s * (1.0 + time_allowance_fraction)
    reference_energy = analyse_lap_energy(track, reference_simulation, energy_model)
    cache: dict[bytes, tuple[LapSimulation, LapEnergy, FloatArray]] = {}

    def evaluate(controls: FloatArray) -> tuple[LapSimulation, LapEnergy, FloatArray]:
        factors = np.asarray(controls, dtype=np.float64)
        key = factors.tobytes()
        if key not in cache:
            dense_factors = _speed_factors(track, controls_distance, factors, minimum_speed_factor)
            speed_limit = reference_simulation.speed_mps * dense_factors
            simulation = simulate_centreline_lap(
                track, car, grip_scale=grip_scale, speed_limit_mps=speed_limit
            )
            cache[key] = (simulation, analyse_lap_energy(track, simulation, energy_model), speed_limit)
        return cache[key]

    if initial_speed_factor is None:
        initial = np.ones(control_count, dtype=np.float64)
    else:
        initial = np.asarray(initial_speed_factor, dtype=np.float64)
        if initial.shape != (control_count,):
            raise EnergyPlanningError("Initial speed factors must match the control-point count.")
        initial = np.clip(initial, minimum_speed_factor, 1.0)

    def objective(controls: FloatArray) -> float:
        return evaluate(controls)[1].net_battery_energy_kwh

    def remaining_time(controls: FloatArray) -> float:
        return target_time - evaluate(controls)[0].lap_time_s

    result: OptimizeResult = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(minimum_speed_factor, 1.0)] * control_count,
        constraints=[{"type": "ineq", "fun": remaining_time}],
        options={"maxiter": max_iterations, "ftol": 1e-7, "disp": False},
    )
    if not result.success:
        raise EnergyPlanningError(f"Energy speed planner did not converge: {result.message}")
    simulation, energy, speed_limit = evaluate(np.asarray(result.x, dtype=np.float64))
    if simulation.lap_time_s > target_time + 1e-6:
        raise EnergyPlanningError("Converged plan violates its lap-time allowance.")
    if energy.net_battery_energy_j > reference_energy.net_battery_energy_j + 1e-5:
        raise EnergyPlanningError("Converged energy plan consumes more battery energy than its reference.")
    return EnergySpeedPlan(
        reference_simulation=reference_simulation,
        reference_energy=reference_energy,
        simulation=simulation,
        energy=energy,
        speed_limit_mps=speed_limit,
        control_distance_m=controls_distance,
        control_speed_factor=np.asarray(result.x, dtype=np.float64),
        time_allowance_fraction=time_allowance_fraction,
        solver_success=True,
        solver_message=str(result.message),
        solver_iterations=int(result.nit),
    )
