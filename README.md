# ApexLab

This directory is a self-contained, physics-based racing-line project. The
first slice deliberately implements geometry only: it downloads a public
Silverstone centreline/width data set, converts it into a uniformly sampled
closed track, validates the geometry, and renders a static track plot.

This second slice adds a configurable, quasi-static point-mass vehicle model
and calculates the fastest *feasible centreline* speed profile. It does not
yet optimise the racing line; that belongs to the next commit, so the dynamics
model remains independently testable.

The third slice adds the fastest-line calculation: a broad 20-control solution
is expanded into an 80-control periodic cubic spline. The optimiser minimises
simulated lap time while checking all four wheel centres every metre against
the legal asphalt-plus-curb envelope and the existing tyre-force constraints.
The current curb geometry is a declared 1.5 m uniform strip with 90% asphalt
grip; it is a reproducible modelling assumption, not surveyed curb data.

## Run the first slice

From this directory:

```bash
uv sync --dev
uv run python -m racing_line.prepare_track --download
uv run python -m racing_line.simulate_lap
uv run python -m racing_line.optimise_line
uv run python -m racing_line.export_web_data
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
- `data/processed/silverstone_optimised_line.npz` — spline line, speed, and controls;
- `data/processed/silverstone_optimised_line.json` — comparison and validation metrics;
- `output/silverstone_optimised_line.png` — speed-coloured fastest line.

Run without `--download` after the initial download:

```bash
uv run python -m racing_line.prepare_track
uv run python -m racing_line.simulate_lap
uv run python -m racing_line.optimise_line
uv run python -m racing_line.export_web_data
```

## Interactive replay

The local browser viewer is in `web/`. Generate its simulation input, then
serve the project with any static server:

```bash
uv run python -m racing_line.export_web_data
python -m http.server 8080 -d web
```

Open `http://localhost:8080`. The replay switches between centreline and
optimised line, animates the car using the calculated segment times, and shows
the live timer, speed, driving state, boundary limits, and validation metrics.

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

## Minimum-time line optimisation

The optimiser uses lateral control-point offsets `a`, expanded to every 5 m
track sample with a periodic cubic spline. Each candidate line is converted to
curvature and passed through the same forward/backward vehicle-dynamics solver
as the centreline baseline.

```text
minimise:    simulated lap time T(a)
subject to:  -right_width(s) + clearance <= a(s) <= left_width(s) - clearance
```

The result is accepted only when it is no slower than the baseline, every
spline sample remains within the boundaries, and friction-circle validation
succeeds. The generated JSON reports time improvement, minimum clearance,
maximum tyre use, control-point count, and solver status.

## Geometry convention

Each centreline sample is represented in local Cartesian metres as
`(x, y, right_width, left_width)`. The source track is resampled at 5 m
intervals. Its start point is deliberately not repeated at the end; all
derivatives and distances use periodic (closed-loop) indexing.

The right boundary at distance `s` is `C(s) + w_right(s) n_right(s)` and the
left boundary is `C(s) - w_left(s) n_right(s)`, where `n_right` is the
clockwise normal to the direction of travel. The optimiser uses these same
boundary arrays as hard constraints for the vehicle centre.

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
