from __future__ import annotations

import numpy as np
import pytest

from racing_line.dynamics import CarModel, LapSimulation
from racing_line.energy import EnergyError, EnergyModel, analyse_lap_energy
from racing_line.geometry import ProcessedTrack


def manual_lap(force_n: np.ndarray) -> tuple:
    """Return a constant-speed trace with deliberately known segment work."""

    car = CarModel(
        mass_kg=1_000.0, max_power_w=200_000.0, max_drive_force_n=10_000.0,
        max_brake_force_n=15_000.0, tyre_friction_coefficient=1.2,
        drag_area_m2=0.0, downforce_area_m2=0.0,
        rolling_resistance_coefficient=0.0, max_speed_mps=120.0,
    )
    count = len(force_n)
    points = np.column_stack((np.arange(count, dtype=float), np.zeros(count)))
    widths = np.full(count, 8.0)
    track = ProcessedTrack(
        centreline=points,
        right_boundary=points + np.array([0.0, -8.0]),
        left_boundary=points + np.array([0.0, 8.0]),
        right_width=widths,
        left_width=widths,
        arc_length=np.arange(count, dtype=float) * 10.0,
        segment_length=np.full(count, 10.0),
        heading=np.zeros(count),
        curvature=np.zeros(count),
    )
    speed = np.full(len(force_n), 10.0)
    simulation = LapSimulation(
        car=car,
        corner_speed_limit_mps=np.full(len(force_n), 20.0),
        speed_mps=speed,
        longitudinal_acceleration_mps2=np.zeros(len(force_n)),
        lateral_force_n=np.zeros(len(force_n)),
        longitudinal_tyre_force_n=force_n.astype(float),
        tyre_force_limit_n=np.full(len(force_n), 10_000.0),
        tyre_utilisation=np.zeros(len(force_n)),
        segment_time_s=np.ones(len(force_n)),
        lap_time_s=float(len(force_n)),
        iterations=1,
    )
    return track, simulation


def test_energy_accounting_respects_drive_and_regen_efficiency_and_cap() -> None:
    track, simulation = manual_lap(np.array([100.0, -80.0, 0.0]))
    energy = analyse_lap_energy(
        track,
        simulation,
        EnergyModel(drive_efficiency=0.8, regen_efficiency=0.5, maximum_regen_power_w=500.0),
    )

    assert energy.wheel_energy_j == pytest.approx(1_000.0)
    assert energy.braking_energy_j == pytest.approx(800.0)
    assert energy.battery_draw_j == pytest.approx(1_250.0)
    # One second at 500 W can mechanically capture only 500 J of the 800 J brake event.
    assert energy.recovered_energy_j == pytest.approx(250.0)
    assert energy.friction_brake_energy_j_total == pytest.approx(300.0)
    assert energy.net_battery_energy_j == pytest.approx(1_000.0)


def test_energy_model_rejects_unphysical_efficiencies() -> None:
    with pytest.raises(EnergyError, match="Drive efficiency"):
        EnergyModel(drive_efficiency=1.1)
    with pytest.raises(EnergyError, match="Regeneration efficiency"):
        EnergyModel(regen_efficiency=-0.1)
