"""Two-stage, footprint-aware minimum-time racing-line optimisation."""

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
    """Raised when a converged, legal minimum-time solution is unavailable."""


@dataclass(frozen=True)
class VehicleFootprint:
    """Plan-view wheel geometry used for track-limit validation, in metres."""

    wheelbase_m: float = 3.6
    track_width_m: float = 1.8

    def __post_init__(self) -> None:
        if not np.isfinite(self.wheelbase_m) or self.wheelbase_m <= 0.0:
            raise OptimisationError("Wheelbase must be finite and positive.")
        if not np.isfinite(self.track_width_m) or self.track_width_m <= 0.0:
            raise OptimisationError("Track width must be finite and positive.")


@dataclass(frozen=True)
class CurbModel:
    """Declared legal curb envelope outside the source asphalt boundaries.

    The source geometry does not label individual Silverstone kerbs. The MVP
    therefore uses a documented uniform legal strip on both sides; it is a
    modelling assumption, not surveyed curb geometry.
    """

    right_width_m: float = 1.5
    left_width_m: float = 1.5
    grip_scale: float = 0.90

    def __post_init__(self) -> None:
        if not np.isfinite(self.right_width_m) or self.right_width_m < 0.0:
            raise OptimisationError("Right curb width must be finite and non-negative.")
        if not np.isfinite(self.left_width_m) or self.left_width_m < 0.0:
            raise OptimisationError("Left curb width must be finite and non-negative.")
        if not np.isfinite(self.grip_scale) or not 0.0 < self.grip_scale <= 1.0:
            raise OptimisationError("Curb grip scale must lie in (0, 1].")


@dataclass(frozen=True)
class FootprintReport:
    """Wheel-level legality and surface report for a sampled racing line."""

    wheel_lateral_m: FloatArray
    wheel_on_curb: NDArray[np.bool_]
    wheel_outside: NDArray[np.bool_]
    legal_clearance_m: FloatArray
    grip_scale: FloatArray

    @property
    def all_four_wheels_out(self) -> bool:
        return bool(np.any(np.all(self.wheel_outside, axis=1)))

    @property
    def minimum_clearance_m(self) -> float:
        return float(np.min(self.legal_clearance_m))

    @property
    def curb_wheel_fraction(self) -> float:
        return float(np.mean(self.wheel_on_curb))


@dataclass(frozen=True)
class RacingLineResult:
    """A converged, legal spline-controlled line and baseline comparison."""

    baseline: LapSimulation
    simulation: LapSimulation
    track: ProcessedTrack
    control_distance_m: FloatArray
    control_offset_m: FloatArray
    offset_m: FloatArray
    minimum_clearance_m: float
    curb_wheel_fraction: float
    solver_success: bool
    solver_message: str
    solver_iterations: int
    coarse_control_count: int

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


def _periodic_spline(track: ProcessedTrack, distance_m: FloatArray, values: FloatArray) -> CubicSpline:
    return CubicSpline(
        np.append(distance_m, track.total_length), np.append(values, values[0]), bc_type="periodic"
    )


def _interpolate_offsets(
    track: ProcessedTrack, control_distance_m: FloatArray, control_offset_m: FloatArray
) -> FloatArray:
    return np.asarray(_periodic_spline(track, control_distance_m, control_offset_m)(track.arc_length), dtype=np.float64)


def _transfer_controls(
    track: ProcessedTrack,
    source_distance_m: FloatArray,
    source_offset_m: FloatArray,
    target_distance_m: FloatArray,
) -> FloatArray:
    """Carry the broad first-stage line into the finer spline basis."""

    return np.asarray(
        _periodic_spline(track, source_distance_m, source_offset_m)(target_distance_m), dtype=np.float64
    )


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
    """Return centre-point clearance to the source asphalt boundaries."""

    offset_m = np.asarray(offset_m, dtype=np.float64)
    return np.minimum(track.right_width - offset_m, track.left_width + offset_m)


def footprint_report(
    track: ProcessedTrack,
    line: ProcessedTrack,
    footprint: VehicleFootprint,
    curb: CurbModel,
) -> FootprintReport:
    """Classify all four wheel centres against asphalt, curb, and legal limits."""

    if line.sample_count != track.sample_count:
        raise OptimisationError("Source track and racing line must use identical sample counts.")
    source_normal = _right_normal(track)
    line_tangent = np.column_stack((np.cos(line.heading), np.sin(line.heading)))
    line_right = np.column_stack((np.sin(line.heading), -np.cos(line.heading)))
    wheel_points = np.stack(
        [
            line.centreline + longitudinal * 0.5 * footprint.wheelbase_m * line_tangent
            + lateral * 0.5 * footprint.track_width_m * line_right
            for longitudinal in (-1.0, 1.0)
            for lateral in (-1.0, 1.0)
        ],
        axis=1,
    )
    relative = wheel_points - track.centreline[:, None, :]
    wheel_lateral = np.einsum("nkj,nj->nk", relative, source_normal)
    right_limit = (track.right_width + curb.right_width_m)[:, None]
    left_limit = (track.left_width + curb.left_width_m)[:, None]
    outside = (wheel_lateral > right_limit) | (wheel_lateral < -left_limit)
    on_right_curb = (wheel_lateral > track.right_width[:, None]) & ~outside
    on_left_curb = (wheel_lateral < -track.left_width[:, None]) & ~outside
    on_curb = on_right_curb | on_left_curb
    clearance = np.where(wheel_lateral >= 0.0, right_limit - wheel_lateral, left_limit + wheel_lateral)
    curb_fraction = np.mean(on_curb, axis=1)
    grip = 1.0 - (1.0 - curb.grip_scale) * curb_fraction
    return FootprintReport(
        wheel_lateral_m=np.asarray(wheel_lateral, dtype=np.float64),
        wheel_on_curb=on_curb,
        wheel_outside=outside,
        legal_clearance_m=np.asarray(clearance, dtype=np.float64),
        grip_scale=np.asarray(grip, dtype=np.float64),
    )


def _dense_footprint_report(
    track: ProcessedTrack,
    offset_m: FloatArray,
    footprint: VehicleFootprint,
    curb: CurbModel,
    spacing_m: float,
) -> FootprintReport:
    """Rebuild the line at a fine spacing for independent legality validation."""

    if spacing_m <= 0.0:
        raise OptimisationError("Boundary-validation spacing must be positive.")
    count = max(3, int(np.ceil(track.total_length / spacing_m)))
    distance = np.linspace(0.0, track.total_length, count, endpoint=False)
    source_distance = np.append(track.arc_length, track.total_length)

    def interpolate(values: FloatArray) -> FloatArray:
        values = np.asarray(values, dtype=np.float64)
        return np.interp(distance, source_distance, np.append(values, values[0]))

    source_centreline = np.column_stack((interpolate(track.centreline[:, 0]), interpolate(track.centreline[:, 1])))
    source_normal = np.column_stack(
        (interpolate(_right_normal(track)[:, 0]), interpolate(_right_normal(track)[:, 1]))
    )
    source_normal /= np.linalg.norm(source_normal, axis=1)[:, None]
    dense_offset = np.asarray(_periodic_spline(track, track.arc_length, offset_m)(distance), dtype=np.float64)
    line_centreline = source_centreline + dense_offset[:, None] * source_normal
    segment_length, arc_length, heading, curvature = closed_path_kinematics(line_centreline)
    dense_track = ProcessedTrack(
        centreline=source_centreline,
        right_boundary=source_centreline + interpolate(track.right_width)[:, None] * source_normal,
        left_boundary=source_centreline - interpolate(track.left_width)[:, None] * source_normal,
        right_width=interpolate(track.right_width),
        left_width=interpolate(track.left_width),
        arc_length=arc_length,
        segment_length=segment_length,
        heading=interpolate(track.heading),
        curvature=interpolate(track.curvature),
    )
    dense_line = replace(dense_track, centreline=line_centreline, heading=heading, curvature=curvature)
    return footprint_report(dense_track, dense_line, footprint, curb)


def validate_racing_line(
    track: ProcessedTrack,
    offset_m: FloatArray,
    minimum_clearance_m: float,
    *,
    footprint: VehicleFootprint = VehicleFootprint(),
    curb: CurbModel = CurbModel(),
    validation_spacing_m: float = 1.0,
) -> FootprintReport:
    """Verify wheel-level limits every metre, not just at spline controls."""

    if minimum_clearance_m < 0.0:
        raise OptimisationError("Minimum clearance cannot be negative.")
    report = _dense_footprint_report(track, offset_m, footprint, curb, validation_spacing_m)
    if not np.all(np.isfinite(report.legal_clearance_m)):
        raise OptimisationError("Racing line contains non-finite wheel positions.")
    if report.all_four_wheels_out:
        raise OptimisationError("Racing line puts all four wheels outside the legal surface.")
    if report.minimum_clearance_m < minimum_clearance_m - 1e-8:
        raise OptimisationError(
            "Racing line violates the wheel safety margin by "
            f"{minimum_clearance_m - report.minimum_clearance_m:.3f} m."
        )
    return report


def _stage_bounds(
    track: ProcessedTrack,
    control_distance: FloatArray,
    footprint: VehicleFootprint,
    curb: CurbModel,
    minimum_clearance_m: float,
) -> tuple[FloatArray, FloatArray]:
    control_index = np.searchsorted(track.arc_length, control_distance, side="left")
    edge_allowance = 0.5 * footprint.track_width_m + minimum_clearance_m
    lower = -track.right_width[control_index] - curb.right_width_m + edge_allowance
    upper = track.left_width[control_index] + curb.left_width_m - edge_allowance
    if np.any(lower >= upper):
        raise OptimisationError("Requested vehicle margin leaves no usable track width.")
    return lower, upper


@dataclass(frozen=True)
class _StageResult:
    control_distance_m: FloatArray
    control_offset_m: FloatArray
    offset_m: FloatArray
    simulation: LapSimulation
    footprint: FootprintReport
    result: OptimizeResult


def _optimise_stage(
    track: ProcessedTrack,
    car: CarModel,
    baseline: LapSimulation,
    *,
    control_count: int,
    minimum_clearance_m: float,
    max_iterations: int,
    footprint: VehicleFootprint,
    curb: CurbModel,
    initial_offsets_m: FloatArray | None,
) -> _StageResult:
    control_distance = _control_distances(track, control_count)
    lower, upper = _stage_bounds(track, control_distance, footprint, curb, minimum_clearance_m)
    bounds = list(zip(lower, upper, strict=True))
    initial = np.zeros(control_count, dtype=np.float64) if initial_offsets_m is None else np.asarray(initial_offsets_m, dtype=np.float64)
    if initial.shape != (control_count,):
        raise OptimisationError("Initial offsets must match the control-point count.")
    initial = np.clip(initial, lower, upper)

    def objective(control_offset: FloatArray) -> float:
        offsets = _interpolate_offsets(track, control_distance, control_offset)
        try:
            validate_racing_line(track, offsets, minimum_clearance_m, footprint=footprint, curb=curb)
        except (OptimisationError, TrackGeometryError, ValueError):
            centre_clearance = float(np.min(boundary_clearance_m(track, offsets)))
            return baseline.lap_time_s + 1_000_000.0 * max(1e-6, minimum_clearance_m - centre_clearance) ** 2
        try:
            candidate = racing_line_track(track, offsets)
            surface = footprint_report(track, candidate, footprint, curb)
            return simulate_centreline_lap(candidate, car, grip_scale=surface.grip_scale).lap_time_s
        except (TrackGeometryError, ValueError):
            return baseline.lap_time_s + 1_000_000.0

    result: OptimizeResult = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": max_iterations, "ftol": 1e-4, "disp": False},
    )
    if not result.success:
        raise OptimisationError(f"Optimisation stage did not converge: {result.message}")
    offset = _interpolate_offsets(track, control_distance, np.asarray(result.x, dtype=np.float64))
    dense = validate_racing_line(track, offset, minimum_clearance_m, footprint=footprint, curb=curb)
    candidate = racing_line_track(track, offset)
    surface = footprint_report(track, candidate, footprint, curb)
    simulation = simulate_centreline_lap(candidate, car, grip_scale=surface.grip_scale)
    return _StageResult(control_distance, np.asarray(result.x, dtype=np.float64), offset, simulation, dense, result)


def optimise_racing_line(
    track: ProcessedTrack,
    car: CarModel,
    *,
    control_count: int = 80,
    coarse_control_count: int = 20,
    minimum_clearance_m: float = 1.2,
    max_iterations: int = 500,
    coarse_max_iterations: int = 200,
    footprint: VehicleFootprint = VehicleFootprint(),
    curb: CurbModel = CurbModel(),
    initial_offsets_m: FloatArray | None = None,
) -> RacingLineResult:
    """Find a converged, footprint-legal minimum-time line.

    A broad 20-control solve provides a stable racing-line shape. Its periodic
    spline is transferred into the 80-control basis, allowing individual entry,
    apex, and exit adjustments. Both stages must converge. Every candidate and
    the returned solution are checked at one-metre spacing against all four
    wheel centres, including the declared curb envelope.
    """

    if max_iterations <= 0 or coarse_max_iterations <= 0:
        raise OptimisationError("Maximum iteration counts must be positive.")
    if control_count < 4:
        raise OptimisationError("Final control count must be at least four.")
    baseline = simulate_centreline_lap(track, car)

    if initial_offsets_m is None and 4 <= coarse_control_count < control_count:
        coarse = _optimise_stage(
            track, car, baseline, control_count=coarse_control_count,
            minimum_clearance_m=minimum_clearance_m, max_iterations=coarse_max_iterations,
            footprint=footprint, curb=curb, initial_offsets_m=None,
        )
        final_distance = _control_distances(track, control_count)
        final_initial = _transfer_controls(track, coarse.control_distance_m, coarse.control_offset_m, final_distance)
        coarse_count = coarse_control_count
        coarse_iterations = int(coarse.result.nit)
    else:
        final_initial = initial_offsets_m
        coarse_count = 0
        coarse_iterations = 0

    final = _optimise_stage(
        track, car, baseline, control_count=control_count,
        minimum_clearance_m=minimum_clearance_m, max_iterations=max_iterations,
        footprint=footprint, curb=curb, initial_offsets_m=final_initial,
    )
    if final.simulation.lap_time_s > baseline.lap_time_s + 1e-5:
        raise OptimisationError("Converged optimiser returned a line slower than the centreline baseline.")
    return RacingLineResult(
        baseline=baseline, simulation=final.simulation, track=racing_line_track(track, final.offset_m),
        control_distance_m=final.control_distance_m, control_offset_m=final.control_offset_m,
        offset_m=final.offset_m, minimum_clearance_m=final.footprint.minimum_clearance_m,
        curb_wheel_fraction=final.footprint.curb_wheel_fraction, solver_success=True,
        solver_message=str(final.result.message), solver_iterations=coarse_iterations + int(final.result.nit),
        coarse_control_count=coarse_count,
    )
