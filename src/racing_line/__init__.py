"""Track geometry foundations for the racing-line simulator."""

from .geometry import ProcessedTrack, TrackGeometryError, preprocess_track
from .source import SILVERSTONE_SOURCE_URL, download_silverstone
from .dynamics import CarModel, LapSimulation, default_car, simulate_centreline_lap
from .energy import EnergyModel, LapEnergy, analyse_lap_energy
from .energy_planning import EnergySpeedPlan, plan_energy_efficient_speed
from .optimisation import RacingLineResult, optimise_racing_line

__all__ = [
    "ProcessedTrack",
    "CarModel",
    "EnergyModel",
    "EnergySpeedPlan",
    "LapEnergy",
    "LapSimulation",
    "RacingLineResult",
    "SILVERSTONE_SOURCE_URL",
    "TrackGeometryError",
    "download_silverstone",
    "default_car",
    "analyse_lap_energy",
    "plan_energy_efficient_speed",
    "preprocess_track",
    "simulate_centreline_lap",
    "optimise_racing_line",
]
