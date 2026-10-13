"""Static visualisation for the verified track-geometry slice."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .geometry import ProcessedTrack


def save_track_plot(track: ProcessedTrack, destination: Path, *, title: str = "Silverstone") -> None:
    """Render boundaries, centreline, direction, and start/finish line."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 7.5), layout="constrained")
    right = np.vstack((track.right_boundary, track.right_boundary[0]))
    left = np.vstack((track.left_boundary, track.left_boundary[0]))
    centre = np.vstack((track.centreline, track.centreline[0]))

    axis.fill(
        np.concatenate((right[:, 0], left[::-1, 0])),
        np.concatenate((right[:, 1], left[::-1, 1])),
        color="#172554",
        alpha=0.2,
        label="Drivable surface",
    )
    axis.plot(right[:, 0], right[:, 1], color="#2563eb", linewidth=1.4, label="Right boundary")
    axis.plot(left[:, 0], left[:, 1], color="#eab308", linewidth=1.4, label="Left boundary")
    axis.plot(centre[:, 0], centre[:, 1], color="#e2e8f0", linestyle="--", linewidth=1.2, label="Centreline")

    start = track.centreline[0]
    normal = track.right_boundary[0] - track.left_boundary[0]
    axis.plot(
        [start[0] - normal[0] / 2, start[0] + normal[0] / 2],
        [start[1] - normal[1] / 2, start[1] + normal[1] / 2],
        color="#f8fafc",
        linewidth=2.2,
        label="Start / finish",
    )
    direction_index = track.sample_count // 12
    tangent = track.centreline[(direction_index + 1) % track.sample_count] - track.centreline[direction_index]
    axis.annotate(
        "Direction of travel",
        xy=track.centreline[direction_index] + tangent * 8,
        xytext=track.centreline[direction_index] + tangent * 45,
        arrowprops={"arrowstyle": "->", "color": "#0f172a", "lw": 1.6},
        color="#0f172a",
        fontsize=10,
    )
    axis.set_title(f"{title} — validated track geometry")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(destination, dpi=180)
    plt.close(figure)
