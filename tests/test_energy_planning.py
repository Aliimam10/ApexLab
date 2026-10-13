from __future__ import annotations

import numpy as np

from racing_line.dynamics import CarModel, simulate_centreline_lap
from racing_line.energy_planning import plan_energy_efficient_speed
from racing_line.geometry import preprocess_track


def test_fixed_line_energy_plan_saves_energy_within_time_budget() -> None:
    radius_m, count = 70.0, 120
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    points = np.column_stack((radius_m * np.cos(angles), radius_m * np.sin(angles)))
    track = preprocess_track(points, np.full(count, 8.0), np.full(count, 8.0), spacing_m=5.0)
    car = CarModel(
        mass_kg=1_000.0, max_power_w=180_000.0, max_drive_force_n=7_000.0,
        max_brake_force_n=12_000.0, tyre_friction_coefficient=1.2,
        drag_area_m2=0.8, downforce_area_m2=0.0,
        rolling_resistance_coefficient=0.01, max_speed_mps=70.0,
    )
    fastest = simulate_centreline_lap(track, car)

    plan = plan_energy_efficient_speed(
        track, car, fastest, time_allowance_fraction=0.10, control_count=4, max_iterations=100
    )

    assert plan.solver_success
    assert plan.simulation.lap_time_s <= plan.target_lap_time_s + 1e-6
    assert plan.energy.net_battery_energy_j < plan.reference_energy.net_battery_energy_j
    assert plan.lap_time_cost_s > 0.0
