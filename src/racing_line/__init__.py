"""Track geometry foundations for the racing-line simulator."""

from .geometry import ProcessedTrack, TrackGeometryError, preprocess_track
from .source import SILVERSTONE_SOURCE_URL, download_silverstone

__all__ = [
    "ProcessedTrack",
    "SILVERSTONE_SOURCE_URL",
    "TrackGeometryError",
    "download_silverstone",
    "preprocess_track",
]
