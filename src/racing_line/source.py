"""Download and identify the public source data used by this project."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen

SILVERSTONE_SOURCE_URL = (
    "https://raw.githubusercontent.com/TUMFTM/racetrack-database/"
    "master/tracks/Silverstone.csv"
)


def download_silverstone(destination: Path) -> str:
    """Download Silverstone source geometry and return its SHA-256 digest.

    The response is checked before it replaces a local file, so a temporary
    GitHub error page cannot silently become track data.
    """

    request = Request(SILVERSTONE_SOURCE_URL, headers={"User-Agent": "racing-line/0.1"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS URL
        payload = response.read()

    if not payload.startswith(b"# x_m"):
        raise RuntimeError("The Silverstone download did not have the expected CSV header.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a local source file."""

    return sha256(path.read_bytes()).hexdigest()
