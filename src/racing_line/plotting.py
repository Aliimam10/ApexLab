"""Static visualisation for the verified track-geometry slice."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from .geometry import ProcessedTrack
from .dynamics import LapSimulation
from .optimisation import RacingLineResult


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


def save_optimisation_plot(
    track: ProcessedTrack, result: RacingLineResult, destination: Path
) -> None:
    """Render boundaries, centreline baseline, and a speed-coloured racing line."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 7.5), layout="constrained")
    right = np.vstack((track.right_boundary, track.right_boundary[0]))
    left = np.vstack((track.left_boundary, track.left_boundary[0]))
    axis.fill(
        np.concatenate((right[:, 0], left[::-1, 0])),
        np.concatenate((right[:, 1], left[::-1, 1])),
        color="#172554",
        alpha=0.15,
        label="Drivable surface",
    )
    axis.plot(right[:, 0], right[:, 1], color="#2563eb", linewidth=1.1, label="Right boundary")
    axis.plot(left[:, 0], left[:, 1], color="#eab308", linewidth=1.1, label="Left boundary")
    baseline = np.vstack((track.centreline, track.centreline[0]))
    axis.plot(baseline[:, 0], baseline[:, 1], color="#94a3b8", linewidth=1.2, linestyle="--", label="Centreline baseline")
    path = result.track.centreline
    line = LineCollection(np.stack((path, np.roll(path, -1, axis=0)), axis=1), cmap="viridis", linewidth=3.0)
    line.set_array(result.simulation.speed_mps * 3.6)
    axis.add_collection(line)
    colourbar = figure.colorbar(line, ax=axis, pad=0.02)
    colourbar.set_label("Optimised-line speed [km/h]")
    axis.set_title(f"Silverstone fastest feasible line — {result.lap_time_improvement_s:.3f} s faster than centreline")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def save_dynamics_plot(track: ProcessedTrack, simulation: LapSimulation, destination: Path) -> None:
    """Save speed, longitudinal acceleration, curvature, and tyre-use charts."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    distance = track.arc_length
    figure, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, layout="constrained")

    axes[0].plot(distance, simulation.speed_mps * 3.6, color="#0369a1", linewidth=1.8)
    axes[0].plot(
        distance,
        simulation.corner_speed_limit_mps * 3.6,
        color="#94a3b8",
        linewidth=1.0,
        linestyle="--",
        label="Curvature limit",
    )
    axes[0].set_ylabel("Speed [km/h]")
    axes[0].legend(loc="upper right")

    acceleration = simulation.longitudinal_acceleration_mps2
    axes[1].fill_between(distance, acceleration, 0.0, where=acceleration >= 0.0, color="#16a34a", alpha=0.72, label="Accelerating")
    axes[1].fill_between(distance, acceleration, 0.0, where=acceleration < 0.0, color="#dc2626", alpha=0.72, label="Braking")
    axes[1].axhline(0.0, color="#334155", linewidth=0.8)
    axes[1].set_ylabel("Long. accel. [m/s²]")
    axes[1].legend(loc="upper right")

    axes[2].plot(distance, track.curvature, color="#7c3aed", linewidth=1.4)
    axes[2].axhline(0.0, color="#334155", linewidth=0.8)
    axes[2].set_ylabel("Curvature [1/m]")

    axes[3].plot(distance, simulation.tyre_utilisation, color="#d97706", linewidth=1.5)
    axes[3].axhline(1.0, color="#dc2626", linewidth=1.0, linestyle="--", label="Friction-circle limit")
    axes[3].set_ylabel("Tyre utilisation [-]")
    axes[3].set_xlabel("Distance around lap, s [m]")
    axes[3].legend(loc="upper right")

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xlim(0.0, track.total_length)
    figure.suptitle(f"Silverstone centreline — {simulation.lap_time_s:.3f} s feasible lap")
    figure.savefig(destination, dpi=180)
    plt.close(figure)
