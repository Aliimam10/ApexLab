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
uv run python -m racing_line.plan_energy
uv run python -m racing_line.pareto_energy
uv run python -m racing_line.export_web_data
uv run python -m racing_line.autonomous_lap
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
uv run python -m racing_line.plan_energy
uv run python -m racing_line.pareto_energy
uv run python -m racing_line.export_web_data
uv run python -m racing_line.autonomous_lap
```

## Interactive replay

The local browser viewer is in `web/`. Generate its simulation input, then
serve the project with any static server:

```bash
uv run python -m racing_line.export_web_data
python -m http.server 8080 -d web
```

Open `http://localhost:8080`. The replay switches between the centreline,
fastest profile, and independently converged +1%, +3%, and +5% energy-aware
profiles. It animates the car using calculated segment times and shows the
live timer, speed, battery energy, regenerative versus friction braking,
boundary limits, validation metrics, and a time--energy Pareto chart.

The energy-aware profiles retain the validated fastest racing line and alter
only the speed/braking plan. Their Pareto points are therefore a comparison of
the time cost of energy saving on one fixed path, not a new joint
line-and-energy optimisation.

## Autonomous path-following extension

`autonomous_lap.py` uses the existing optimised path and force-feasible speed
profile as a reference for a separate feedback simulation. A kinematic bicycle
model follows the reference using Pure Pursuit steering and a bounded PID speed
controller. The controller's acceleration request is clipped by the existing
power, braking, and tyre-force functions; its steering-curvature speed limit
uses a documented 3% safety margin for discrete actuator response. It does
not replace the existing minimum-time vehicle model.

```bash
uv run python -m racing_line.autonomous_lap
```

The command writes `data/processed/silverstone_autonomous_lap.npz` and CSV
telemetry with the timestamped actual path, heading, speed, steering, command,
tracking-error, and boundary-clearance traces; it also writes a JSON summary
and `output/silverstone_autonomous_lap.png`. The plot compares planned and
driven paths, target and actual speed, commanded and actual steering, and
tracking error against boundary clearance. A run is marked complete only after
one reference-lap length has been tracked; it terminates on a track boundary or
lateral-friction violation. Its centre-point boundary check uses the same
declared 1.5 m legal-curb envelope as the existing optimiser; it is not yet a
four-wheel tracking-limit calculation. This first autonomy slice uses ideal
state feedback. Sensor noise, actuator delay, Monte Carlo robustness tests,
and browser replay of the actual path are subsequent extensions.

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

## Energy accounting

The first energy slice measures, but does not yet optimise, the energy of the
baseline and fastest feasible laps. For every 5 m segment it integrates wheel
work as `E_wheel = F_x ds`. Positive wheel force consumes battery energy using
a configurable drivetrain efficiency. Negative wheel force is braking work;
a configurable regenerative-power cap and regeneration efficiency determine
the recoverable battery energy. The remaining braking work is reported as
friction-brake dissipation.

The initial generic assumptions are 94% drive efficiency, 65% regeneration
efficiency, and 150 kW maximum regenerative mechanical power. They are
explicit model inputs rather than claims about a particular F1 power unit.
The next energy slice will optimise the speed profile subject to a lap-time
constraint; this commit only provides the audited baseline measurements.

## Fixed-line energy speed planning

`plan_energy.py` keeps the converged fastest racing line unchanged and solves
for a lower-energy speed profile. Twenty-four bounded speed-ceiling controls
are expanded into a periodic spline and multiply the fastest feasible speed
profile. The existing forward/backward dynamics solver then enforces physical
cornering, acceleration, braking, and tyre constraints below that ceiling.

The numerical problem is:

```text
minimise:    net battery energy E(speed ceiling)
subject to:  lap time <= fastest-lap time × (1 + allowance)
```

The default allowance is 3%. It is configurable, for example:

```bash
uv run python -m racing_line.plan_energy --time-allowance-percent 5
```

The command writes `data/processed/silverstone_energy_plan.json`, its NumPy
trace, and `output/silverstone_energy_plan.png`. The output is accepted only
when SLSQP converges, the lap-time constraint is met, and its net battery
energy is no greater than the fastest profile's.

## Time-energy Pareto frontier

`pareto_energy.py` repeats the fixed-line planner at several allowed lap-time
costs. The default frontier contains the fastest point plus 1%, 3%, and 5%
allowances. A point is retained only when it is nondominated: no other reported
point is both faster and equal-or-lower in battery energy. Each allowance is
solved independently from the fastest profile and must converge and meet its
own time constraint.

```bash
uv run python -m racing_line.pareto_energy --allowances 0,1,3,5
```

The command writes a JSON table, NumPy speed traces, and
`output/silverstone_energy_pareto.png`. The same independently converged
profiles are also exported to the local browser viewer by
`export_web_data.py` for an interactive comparison.

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
