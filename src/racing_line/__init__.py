"""Track geometry foundations for the racing-line simulator."""

from .geometry import ProcessedTrack, TrackGeometryError, preprocess_track
from .source import SILVERSTONE_SOURCE_URL, download_silverstone
from .dynamics import CarModel, LapSimulation, default_car, simulate_centreline_lap
from .energy import EnergyModel, LapEnergy, analyse_lap_energy
from .energy_planning import EnergySpeedPlan, plan_energy_efficient_speed
from .energy_pareto import EnergyParetoFrontier, ParetoPoint, generate_energy_pareto_frontier
from .optimisation import RacingLineResult, optimise_racing_line
from .autonomy import (
    AutonomousLapResult,
    AutonomousSimulationConfig,
    BicycleModel,
    PIDConfig,
    PurePursuitConfig,
    ReferenceTrajectory,
    VehicleState,
    simulate_autonomous_lap,
)

__all__ = [
    "ProcessedTrack",
    "CarModel",
    "BicycleModel",
    "EnergyModel",
    "EnergySpeedPlan",
    "EnergyParetoFrontier",
    "LapEnergy",
    "ParetoPoint",
    "LapSimulation",
    "AutonomousLapResult",
    "AutonomousSimulationConfig",
    "PIDConfig",
    "PurePursuitConfig",
    "ReferenceTrajectory",
    "RacingLineResult",
    "SILVERSTONE_SOURCE_URL",
    "TrackGeometryError",
    "VehicleState",
    "download_silverstone",
    "default_car",
    "analyse_lap_energy",
    "plan_energy_efficient_speed",
    "generate_energy_pareto_frontier",
    "preprocess_track",
    "simulate_centreline_lap",
    "simulate_autonomous_lap",
    "optimise_racing_line",
]
