# Racing line simulator

This directory is a self-contained, physics-based racing-line project. The
first slice deliberately implements geometry only: it downloads a public
Silverstone centreline/width data set, converts it into a uniformly sampled
closed track, validates the geometry, and renders a static track plot.

No lap-time, vehicle-dynamics, or racing-line optimisation code is included
yet. Those pieces belong to later commits so this track foundation can be
verified independently.

## Run the first slice

From this directory:

```bash
uv sync --dev
uv run python -m racing_line.prepare_track --download
uv run pytest
```

The first command that includes `--download` saves the upstream data locally
under `data/raw/`; this local cache and all generated artefacts are ignored by
Git. The preprocessing command creates:

- `data/processed/silverstone_geometry.npz` — NumPy arrays for later commits;
- `data/processed/silverstone_geometry.json` — reproducibility metadata;
- `output/silverstone_geometry.png` — static geometry visualisation.

Run without `--download` after the initial download:

```bash
uv run python -m racing_line.prepare_track
```

## Geometry convention

Each centreline sample is represented in local Cartesian metres as
`(x, y, right_width, left_width)`. The source track is resampled at 5 m
intervals. Its start point is deliberately not repeated at the end; all
derivatives and distances use periodic (closed-loop) indexing.

The right boundary at distance `s` is `C(s) + w_right(s) n_right(s)` and the
left boundary is `C(s) - w_left(s) n_right(s)`, where `n_right` is the
clockwise normal to the direction of travel. The next commit will use these
same boundary arrays as hard constraints for the vehicle centre.

## Data provenance

Silverstone data is fetched from the
[TUMFTM racetrack database](https://github.com/TUMFTM/racetrack-database),
which supplies smoothed centreline coordinates and left/right track widths.
Its README states that centreline GPS points originate from OpenStreetMap and
the widths were derived from satellite imagery. The upstream project is
licensed under LGPL-3.0; see its
[licence](https://github.com/TUMFTM/racetrack-database/blob/master/LICENSE).

The downloaded file's SHA-256 digest and source URL are stored with each
processed geometry artefact. This makes a result reproducible even if the
upstream file changes later.
