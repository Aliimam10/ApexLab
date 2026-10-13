# ApexLab

This directory is a self-contained, physics-based racing-line project. The
first slice deliberately implements geometry only: it downloads a public
Silverstone centreline/width data set, converts it into a uniformly sampled
closed track, validates the geometry, and renders a static track plot.

This second slice adds a configurable, quasi-static point-mass vehicle model
and calculates the fastest *feasible centreline* speed profile. It does not
yet optimise the racing line; that belongs to the next commit, so the dynamics
model remains independently testable.

## Run the first slice

From this directory:

```bash
uv sync --dev
uv run python -m racing_line.prepare_track --download
uv run python -m racing_line.simulate_lap
uv run python -m pytest
```

The first command that includes `--download` saves the upstream data locally
under `data/raw/`; this local cache and all generated artefacts are ignored by
Git. The preprocessing command creates:

- `data/processed/silverstone_geometry.npz` — NumPy arrays for later commits;
- `data/processed/silverstone_geometry.json` — reproducibility metadata;
- `output/silverstone_geometry.png` — static geometry visualisation.
- `data/processed/silverstone_centreline_lap.npz` — centreline speed/force traces;
- `data/processed/silverstone_centreline_lap.json` — lap metrics and car parameters;
- `output/silverstone_centreline_dynamics.png` — speed, braking, curvature, and tyre-use plots.

Run without `--download` after the initial download:

```bash
uv run python -m racing_line.prepare_track
uv run python -m racing_line.simulate_lap
```

## Centreline vehicle model

The baseline uses one editable formula-style `CarModel` in
`src/racing_line/dynamics.py`. At each track sample it calculates aerodynamic
drag and downforce, lateral force `m v² |k|`, and remaining longitudinal grip
from the friction circle:

```text
F_x² + F_y² <= [mu (m g + downforce)]²
```

Corner-speed limits are solved numerically. A forward sweep enforces the
power/drag-limited acceleration constraint; a backwards sweep enforces the
maximum-braking constraint of the next corner. Repeated closed-loop sweeps
produce a feasible speed profile and integrate time as:

```text
dt = 2 ds / (v_i + v_i+1)
```

The command refuses a lap that exceeds a local curvature limit or the stated
friction-circle limit (apart from a small numerical tolerance). The resulting
lap is the control case against which the next commit's optimised racing line
will be measured.

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
