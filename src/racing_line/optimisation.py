"""Spline-controlled minimum-time racing-line optimisation."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.optimize import OptimizeResult, minimize

from .dynamics import CarModel, LapSimulation, simulate_centreline_lap
from .geometry import ProcessedTrack, TrackGeometryError, closed_path_kinematics

FloatArray = NDArray[np.float64]


class OptimisationError(RuntimeError):
    """Raised when no valid minimum-time racing-line solution is found."""


@dataclass(frozen=True)
class RacingLineResult:
    """A valid spline-controlled line and its baseline comparison."""

    baseline: LapSimulation
    simulation: LapSimulation
    track: ProcessedTrack
    control_distance_m: FloatArray
    control_offset_m: FloatArray
    offset_m: FloatArray
    minimum_clearance_m: float
    solver_success: bool
    solver_message: str
    solver_iterations: int

    @property
    def lap_time_improvement_s(self) -> float:
        return self.baseline.lap_time_s - self.simulation.lap_time_s

    @property
    def lap_time_improvement_percent(self) -> float:
        return 100.0 * self.lap_time_improvement_s / self.baseline.lap_time_s


def _right_normal(track: ProcessedTrack) -> FloatArray:
    return (track.right_boundary - track.centreline) / track.right_width[:, None]


def _control_distances(track: ProcessedTrack, control_count: int) -> FloatArray:
    if control_count < 4:
        raise OptimisationError("Periodic cubic interpolation needs at least four control points.")
    if control_count >= track.sample_count:
        raise OptimisationError("Control-point count must be smaller than the track sample count.")
    return np.linspace(0.0, track.total_length, control_count, endpoint=False)


def _interpolate_offsets(track: ProcessedTrack, control_distance_m: FloatArray, control_offset_m: FloatArray) -> FloatArray:
    closed_distance = np.append(control_distance_m, track.total_length)
    closed_offset = np.append(control_offset_m, control_offset_m[0])
    spline = CubicSpline(closed_distance, closed_offset, bc_type="periodic")
    return np.asarray(spline(track.arc_length), dtype=np.float64)


def racing_line_track(track: ProcessedTrack, offset_m: FloatArray) -> ProcessedTrack:
    """Construct a path from lateral offsets and recompute its kinematics."""

    offset_m = np.asarray(offset_m, dtype=np.float64)
    if offset_m.shape != (track.sample_count,):
        raise OptimisationError("A racing line needs exactly one offset for every track sample.")
    centreline = track.centreline + offset_m[:, None] * _right_normal(track)
    segment_length, arc_length, heading, curvature = closed_path_kinematics(centreline)
    return replace(
        track,
        centreline=centreline,
        segment_length=segment_length,
        arc_length=arc_length,
        heading=heading,
        curvature=curvature,
    )


def boundary_clearance_m(track: ProcessedTrack, offset_m: FloatArray) -> FloatArray:
    """Return clearance from the nearest boundary for every centre-point sample."""

    offset_m = np.asarray(offset_m, dtype=np.float64)
    return np.minimum(track.right_width - offset_m, track.left_width + offset_m)


def validate_racing_line(track: ProcessedTrack, offset_m: FloatArray, minimum_clearance_m: float) -> None:
    """Check densely evaluated spline offsets against both track boundaries."""

    if minimum_clearance_m < 0.0:
        raise OptimisationError("Minimum clearance cannot be negative.")
    clearance = boundary_clearance_m(track, offset_m)
    if not np.all(np.isfinite(clearance)):
        raise OptimisationError("Racing line contains non-finite offsets.")
    if np.min(clearance) < minimum_clearance_m - 1e-8:
        raise OptimisationError(
            f"Racing line leaves the usable track width by {minimum_clearance_m - np.min(clearance):.3f} m."
        )


def optimise_racing_line(
    track: ProcessedTrack,
    car: CarModel,
    *,
    control_count: int = 12,
    minimum_clearance_m: float = 1.2,
    max_iterations: int = 25,
    initial_offsets_m: FloatArray | None = None,
) -> RacingLineResult:
    """Minimise simulated lap time using bounded periodic spline offsets.

    The only free variables are the control-point lateral offsets. The cubic
    spline is evaluated at every track sample, so boundary checks apply to the
    complete path rather than merely to the sparse control points.
    """

    if max_iterations <= 0:
        raise OptimisationError("Maximum iteration count must be positive.")
    control_distance = _control_distances(track, control_count)
    control_index = np.searchsorted(track.arc_length, control_distance, side="left")
    lower = -track.right_width[control_index] + minimum_clearance_m
    upper = track.left_width[control_index] - minimum_clearance_m
    if np.any(lower >= upper):
        raise OptimisationError("Requested clearance leaves no usable track width.")
    bounds = list(zip(lower, upper, strict=True))
    baseline = simulate_centreline_lap(track, car)

    if initial_offsets_m is None:
        initial = np.zeros(control_count, dtype=np.float64)
    else:
        initial = np.asarray(initial_offsets_m, dtype=np.float64)
        if initial.shape != (control_count,):
            raise OptimisationError("Initial offsets must match the control-point count.")
        initial = np.clip(initial, lower, upper)

    def objective(control_offset: FloatArray) -> float:
        offsets = _interpolate_offsets(track, control_distance, control_offset)
        clearance = boundary_clearance_m(track, offsets)
        violation = np.maximum(minimum_clearance_m - clearance, 0.0)
        if np.any(violation > 0.0):
            # The penalty is only active if cubic interpolation overshoots a
            # valid control point. It keeps dense boundary checks meaningful.
            return baseline.lap_time_s + 1_000_000.0 * float(np.mean(np.square(violation)))
        try:
            candidate = racing_line_track(track, offsets)
            return simulate_centreline_lap(candidate, car).lap_time_s
        except (TrackGeometryError, ValueError):
            return baseline.lap_time_s + 1_000_000.0

    result: OptimizeResult = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": max_iterations, "ftol": 1e-4, "disp": False},
    )
    offset = _interpolate_offsets(track, control_distance, np.asarray(result.x, dtype=np.float64))
    validate_racing_line(track, offset, minimum_clearance_m)
    optimised_track = racing_line_track(track, offset)
    simulation = simulate_centreline_lap(optimised_track, car)
    if simulation.lap_time_s > baseline.lap_time_s + 1e-5:
        raise OptimisationError("Optimiser returned a line slower than the centreline baseline.")
    return RacingLineResult(
        baseline=baseline,
        simulation=simulation,
        track=optimised_track,
        control_distance_m=control_distance,
        control_offset_m=np.asarray(result.x, dtype=np.float64),
        offset_m=offset,
        minimum_clearance_m=minimum_clearance_m,
        solver_success=bool(result.success),
        solver_message=str(result.message),
        solver_iterations=int(result.nit),
    )
