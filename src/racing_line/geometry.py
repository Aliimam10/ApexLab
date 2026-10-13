"""Closed-track preprocessing and geometry validation.

This module intentionally has no vehicle model. It provides the geometric
objects that the later minimum-time solver will consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class TrackGeometryError(ValueError):
    """Raised when a track cannot form a valid closed driving surface."""


@dataclass(frozen=True)
class ProcessedTrack:
    """Uniformly sampled closed circuit in local Cartesian metres."""

    centreline: FloatArray
    right_boundary: FloatArray
    left_boundary: FloatArray
    right_width: FloatArray
    left_width: FloatArray
    arc_length: FloatArray
    segment_length: FloatArray
    heading: FloatArray
    curvature: FloatArray

    @property
    def total_length(self) -> float:
        return float(np.sum(self.segment_length))

    @property
    def sample_count(self) -> int:
        return int(self.centreline.shape[0])


def load_track_csv(path: Path) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Load `[x_m, y_m, right_width_m, left_width_m]` source CSV data."""

    try:
        values = np.loadtxt(path, delimiter=";", comments="#", dtype=np.float64)
    except ValueError:
        # The upstream repository currently uses semicolons; retain a clear
        # fallback for comma-separated compatible files.
        values = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64)

    if values.ndim != 2 or values.shape[1] != 4:
        raise TrackGeometryError("Track CSV must contain x, y, right width, and left width.")
    if not np.all(np.isfinite(values)):
        raise TrackGeometryError("Track CSV contains non-finite values.")
    if np.any(values[:, 2:] <= 0.0):
        raise TrackGeometryError("Track widths must be strictly positive.")

    return values[:, :2], values[:, 2], values[:, 3]


def _remove_duplicate_endpoint(
    points: FloatArray, right_width: FloatArray, left_width: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if np.linalg.norm(points[0] - points[-1]) < 1e-9:
        return points[:-1], right_width[:-1], left_width[:-1]
    return points, right_width, left_width


def _periodic_segment_lengths(points: FloatArray) -> FloatArray:
    segments = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 1e-8):
        raise TrackGeometryError("Track contains duplicate or zero-length consecutive points.")
    return lengths


def _resample_closed_polyline(
    points: FloatArray,
    right_width: FloatArray,
    left_width: FloatArray,
    spacing_m: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if spacing_m <= 0:
        raise TrackGeometryError("Resampling spacing must be positive.")

    lengths = _periodic_segment_lengths(points)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total_length = cumulative[-1]
    sample_count = max(3, int(np.ceil(total_length / spacing_m)))
    targets = np.linspace(0.0, total_length, sample_count, endpoint=False)

    closed_points = np.vstack((points, points[0]))
    closed_right = np.append(right_width, right_width[0])
    closed_left = np.append(left_width, left_width[0])
    resampled_points = np.column_stack(
        (
            np.interp(targets, cumulative, closed_points[:, 0]),
            np.interp(targets, cumulative, closed_points[:, 1]),
        )
    )
    return (
        resampled_points,
        np.interp(targets, cumulative, closed_right),
        np.interp(targets, cumulative, closed_left),
    )


def _periodic_differential_geometry(points: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return right normals, headings, and signed three-point curvatures."""

    previous = np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0)
    tangent = following - previous
    tangent_norm = np.linalg.norm(tangent, axis=1)
    if np.any(tangent_norm <= 1e-8):
        raise TrackGeometryError("Track has an undefined tangent.")
    tangent = tangent / tangent_norm[:, None]
    right_normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    heading = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))

    a = points - previous
    b = following - points
    c = following - previous
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1)
    if np.any(denominator <= 1e-10):
        raise TrackGeometryError("Track has an undefined curvature sample.")
    cross_product = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    curvature = 2.0 * cross_product / denominator
    return right_normal, heading, curvature


def preprocess_track(
    points: FloatArray,
    right_width: FloatArray,
    left_width: FloatArray,
    *,
    spacing_m: float = 5.0,
    minimum_clearance_m: float = 0.5,
) -> ProcessedTrack:
    """Resample a source centreline and construct/validate its boundaries."""

    points = np.asarray(points, dtype=np.float64)
    right_width = np.asarray(right_width, dtype=np.float64)
    left_width = np.asarray(left_width, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise TrackGeometryError("Track needs at least three two-dimensional points.")
    if right_width.shape != (points.shape[0],) or left_width.shape != (points.shape[0],):
        raise TrackGeometryError("Each centreline point needs a left and right width.")
    if not np.all(np.isfinite(points)) or np.any(right_width <= 0) or np.any(left_width <= 0):
        raise TrackGeometryError("Track points must be finite and widths must be positive.")
    if minimum_clearance_m < 0:
        raise TrackGeometryError("Minimum clearance cannot be negative.")
    # Check source samples before interpolation. Otherwise an unsafe narrow
    # section located between resampling targets could be hidden by smoothing.
    if np.any(right_width <= minimum_clearance_m) or np.any(left_width <= minimum_clearance_m):
        raise TrackGeometryError("Track width is not larger than the requested safety margin.")

    points, right_width, left_width = _remove_duplicate_endpoint(points, right_width, left_width)
    points, right_width, left_width = _resample_closed_polyline(
        points, right_width, left_width, spacing_m
    )
    segment_length = _periodic_segment_lengths(points)
    right_normal, heading, curvature = _periodic_differential_geometry(points)
    right_boundary = points + right_width[:, None] * right_normal
    left_boundary = points - left_width[:, None] * right_normal
    arc_length = np.concatenate(([0.0], np.cumsum(segment_length[:-1])))

    track = ProcessedTrack(
        centreline=points,
        right_boundary=right_boundary,
        left_boundary=left_boundary,
        right_width=right_width,
        left_width=left_width,
        arc_length=arc_length,
        segment_length=segment_length,
        heading=heading,
        curvature=curvature,
    )
    validate_track(track, minimum_clearance_m=minimum_clearance_m)
    return track


def validate_track(track: ProcessedTrack, *, minimum_clearance_m: float = 0.5) -> None:
    """Validate the closed geometry used as constraints by later solvers."""

    if minimum_clearance_m < 0:
        raise TrackGeometryError("Minimum clearance cannot be negative.")
    if track.sample_count < 3:
        raise TrackGeometryError("Track has too few samples.")
    if not np.all(np.isfinite(track.centreline)) or not np.all(np.isfinite(track.curvature)):
        raise TrackGeometryError("Processed track contains non-finite values.")
    if np.any(track.segment_length <= 0):
        raise TrackGeometryError("Processed track contains a zero-length segment.")
    if np.any(track.right_width <= minimum_clearance_m) or np.any(
        track.left_width <= minimum_clearance_m
    ):
        raise TrackGeometryError("Track width is not larger than the requested safety margin.")

    right_distance = np.linalg.norm(track.right_boundary - track.centreline, axis=1)
    left_distance = np.linalg.norm(track.left_boundary - track.centreline, axis=1)
    if not np.allclose(right_distance, track.right_width, atol=1e-7):
        raise TrackGeometryError("Right boundary construction failed.")
    if not np.allclose(left_distance, track.left_width, atol=1e-7):
        raise TrackGeometryError("Left boundary construction failed.")
