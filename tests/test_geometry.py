from __future__ import annotations

import numpy as np
import pytest

from racing_line.geometry import TrackGeometryError, preprocess_track


def circle_track(radius: float = 80.0, samples: int = 240):
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    return points, np.full(samples, 7.0), np.full(samples, 7.0)


def test_circle_curvature_matches_analytic_solution() -> None:
    radius = 80.0
    points, right_width, left_width = circle_track(radius=radius)
    track = preprocess_track(points, right_width, left_width, spacing_m=2.0)

    assert np.median(np.abs(track.curvature)) == pytest.approx(1.0 / radius, rel=0.01)
    assert track.total_length == pytest.approx(2.0 * np.pi * radius, rel=0.002)


def test_resampled_closed_track_has_valid_boundaries_and_lengths() -> None:
    points, right_width, left_width = circle_track()
    track = preprocess_track(points, right_width, left_width, spacing_m=5.0)

    assert track.sample_count > 3
    assert np.all(track.segment_length > 0.0)
    assert np.allclose(
        np.linalg.norm(track.right_boundary - track.centreline, axis=1), track.right_width
    )
    assert np.allclose(
        np.linalg.norm(track.left_boundary - track.centreline, axis=1), track.left_width
    )


def test_width_smaller_than_clearance_is_rejected() -> None:
    points, right_width, left_width = circle_track()
    right_width[20] = 0.4

    with pytest.raises(TrackGeometryError, match="safety margin"):
        preprocess_track(points, right_width, left_width, minimum_clearance_m=0.5)
