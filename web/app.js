const svg = document.querySelector("#track");
const profileSvg = document.querySelector("#speed-profile");
const ns = "http://www.w3.org/2000/svg";
const $ = (id) => document.querySelector(`#${id}`);

let data;
let active = "optimised";
let running = false;
let elapsed = 0;
let last = 0;
let rate = 1;
let car;
let profileMarker;

// Presentation only: the visual road is widened so approach, apex, and exit
// placement are visible. It never changes the optimised geometry or physics.
const DISPLAY_WIDTH_SCALE = 10;

const node = (name, attributes = {}) => {
  const element = document.createElementNS(ns, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
};
const format = (seconds) => `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(3).padStart(6, "0")}`;
const speedColour = (speed, minimum, maximum) =>
  `hsl(${260 - 210 * ((speed - minimum) / (maximum - minimum || 1))} 82% 58%)`;

fetch("data/silverstone.json")
  .then((response) => {
    if (!response.ok) throw new Error(`Simulation data request failed (${response.status}).`);
    return response.json();
  })
  .then(init)
  .catch((error) => {
    console.error(error);
    document.body.innerHTML = `<p style="padding:2rem">Could not initialise the replay: <code>${error.message}</code></p>`;
  });

function init(payload) {
  data = payload;
  drawStatic();
  updateValidation();
  reset();
  $("play").onclick = () => {
    running = !running;
    $("play").textContent = running ? "Pause" : "Play";
    if (running) {
      last = 0;
      requestAnimationFrame(tick);
    }
  };
  $("reset").onclick = reset;
  $("rate").oninput = (event) => {
    rate = +event.target.value;
    $("rate-output").textContent = `${rate}×`;
  };
  $("line-select").onchange = (event) => {
    active = event.target.value;
    $("line-label").textContent = active === "optimised" ? "OPTIMISED LINE" : "CENTRELINE BASELINE";
    drawStatic();
    reset();
  };
  $("overlays").onchange = () => {
    $("legend").hidden = !$("overlays").checked;
    drawStatic();
    update(elapsed);
  };
}

function widenForDisplay(points) {
  return points.map(([x, y], index) => {
    const [centreX, centreY] = data.track.centreline[index];
    return [centreX + (x - centreX) * DISPLAY_WIDTH_SCALE, centreY + (y - centreY) * DISPLAY_WIDTH_SCALE];
  });
}

function map(points) {
  const all = [...widenForDisplay(data.track.left_boundary), ...widenForDisplay(data.track.right_boundary)];
  const xValues = all.map((point) => point[0]);
  const yValues = all.map((point) => point[1]);
  const pad = 45;
  const width = 1000;
  const height = 760;
  const scale = Math.min(
    (width - 2 * pad) / (Math.max(...xValues) - Math.min(...xValues)),
    (height - 2 * pad) / (Math.max(...yValues) - Math.min(...yValues)),
  );
  const offsetX = (width - scale * (Math.max(...xValues) - Math.min(...xValues))) / 2 - scale * Math.min(...xValues);
  const offsetY = (height - scale * (Math.max(...yValues) - Math.min(...yValues))) / 2 - scale * Math.min(...yValues);
  return points.map(([x, y]) => [x * scale + offsetX, height - (y * scale + offsetY)]);
}

function drawPath(points, colour, width, dash = "") {
  const mapped = map(points);
  svg.appendChild(node("path", {
    d: `M${mapped.map((point) => point.join(",")).join(" L")}`,
    fill: "none", stroke: colour, "stroke-width": width, "stroke-dasharray": dash,
  }));
}

function linePoints() {
  return active === "optimised" ? data.optimised.line : data.track.centreline;
}

function drawStatic() {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const right = widenForDisplay(data.track.right_boundary);
  const left = widenForDisplay(data.track.left_boundary);
  const showOverlays = $("overlays").checked;
  const road = map([...right, ...left.slice().reverse()]);
  svg.appendChild(node("path", {
    d: `M${road.map((point) => point.join(",")).join(" L")} Z`,
    fill: "#6d7480", stroke: "#3c4555", "stroke-width": 1,
  }));
  if (showOverlays) {
    drawPath(right, "#377dff", 2);
    drawPath(left, "#f8c64e", 2);
    if (active === "optimised") drawPath(data.track.centreline, "#c7d1e0", 1, "5 5");
  }
  const path = map(widenForDisplay(linePoints()));
  const speeds = data[active].speed_mps;
  const minimum = Math.min(...speeds);
  const maximum = Math.max(...speeds);
  path.forEach((point, index) => svg.appendChild(node("line", {
    x1: point[0], y1: point[1],
    x2: path[(index + 1) % path.length][0], y2: path[(index + 1) % path.length][1],
    stroke: showOverlays ? speedColour(speeds[index], minimum, maximum) : "#f8fafc",
    "stroke-width": showOverlays ? 2.2 : 0.85,
    "stroke-linecap": "round",
  })));
  car = node("g");
  car.appendChild(node("circle", { r: 8, fill: "#fff", stroke: "#07101d", "stroke-width": 3 }));
  car.appendChild(node("path", { d: "M-3,-4 L7,0 L-3,4 Z", fill: "#061320" }));
  svg.appendChild(car);
  drawSpeedProfile();
}

function profileData() {
  const lap = data[active];
  const times = [0];
  const speeds = [lap.speed_mps[0]];
  let time = 0;
  lap.segment_time_s.forEach((segmentTime, index) => {
    time += segmentTime;
    times.push(time);
    speeds.push(lap.speed_mps[(index + 1) % lap.speed_mps.length]);
  });
  return { times, speeds, lapTime: lap.lap_time_s };
}

function drawSpeedProfile() {
  while (profileSvg.firstChild) profileSvg.removeChild(profileSvg.firstChild);
  const { times, speeds, lapTime } = profileData();
  const width = 1000;
  const height = 250;
  const left = 58;
  const right = 24;
  const top = 22;
  const bottom = 38;
  const minimumSpeed = Math.floor(Math.min(...speeds) * 3.6 / 10) * 10;
  const maximumSpeed = Math.ceil(Math.max(...speeds) * 3.6 / 10) * 10;
  const x = (time) => left + (width - left - right) * time / lapTime;
  const y = (speed) => top + (height - top - bottom) * (1 - (speed * 3.6 - minimumSpeed) / (maximumSpeed - minimumSpeed || 1));

  [0, 0.25, 0.5, 0.75, 1].forEach((fraction) => {
    const value = minimumSpeed + (maximumSpeed - minimumSpeed) * fraction;
    const position = y(value / 3.6);
    profileSvg.appendChild(node("line", {
      x1: left, y1: position, x2: width - right, y2: position,
      stroke: "#26354e", "stroke-width": 1, "stroke-dasharray": "3 5",
    }));
    const label = node("text", { x: left - 10, y: position + 4, fill: "#8d9bb3", "font-size": 11, "text-anchor": "end" });
    label.textContent = `${Math.round(value)}`;
    profileSvg.appendChild(label);
  });
  [0, 0.25, 0.5, 0.75, 1].forEach((fraction) => {
    const time = lapTime * fraction;
    const position = x(time);
    profileSvg.appendChild(node("line", {
      x1: position, y1: top, x2: position, y2: height - bottom,
      stroke: "#1b2940", "stroke-width": 1,
    }));
    const label = node("text", { x: position, y: height - 14, fill: "#8d9bb3", "font-size": 11, "text-anchor": "middle" });
    label.textContent = `${time.toFixed(0)} s`;
    profileSvg.appendChild(label);
  });
  profileSvg.appendChild(node("path", {
    d: `M${times.map((time, index) => `${x(time)},${y(speeds[index])}`).join(" L")}`,
    fill: "none", stroke: "#37d9af", "stroke-width": 2.5,
    "stroke-linejoin": "round", "stroke-linecap": "round",
  }));
  const units = node("text", { x: 15, y: 18, fill: "#8d9bb3", "font-size": 11 });
  units.textContent = "km/h";
  profileSvg.appendChild(units);
  profileMarker = node("line", {
    x1: x(0), y1: top, x2: x(0), y2: height - bottom,
    stroke: "#f8c64e", "stroke-width": 2,
  });
  profileSvg.appendChild(profileMarker);
  $("profile-label").textContent = `${active === "optimised" ? "OPTIMISED LINE" : "CENTRELINE BASELINE"} · VELOCITY VS TIME`;
}

function updateProfileMarker(time) {
  if (!profileMarker || !data) return;
  const lapTime = data[active].lap_time_s;
  const clamped = Math.min(Math.max(time, 0), lapTime);
  const position = 58 + (1000 - 58 - 24) * clamped / lapTime;
  profileMarker.setAttribute("x1", position);
  profileMarker.setAttribute("x2", position);
}

function updateValidation() {
  const metrics = data.metrics;
  $("gain").textContent = `${metrics.improvement_s.toFixed(3)} s`;
  $("validation").innerHTML = `<dt>Baseline lap</dt><dd>${format(data.baseline.lap_time_s)}</dd><dt>Fastest line</dt><dd>${format(data.optimised.lap_time_s)}</dd><dt>Improvement</dt><dd>${metrics.improvement_percent.toFixed(2)}%</dd><dt>Min. wheel clearance</dt><dd>${metrics.minimum_clearance_m.toFixed(2)} m</dd><dt>Curb tyre use</dt><dd>${(metrics.curb_wheel_fraction * 100).toFixed(1)}%</dd><dt>Max tyre use</dt><dd>${metrics.maximum_tyre_utilisation.toFixed(3)}</dd><dt>Controls</dt><dd>${metrics.coarse_control_points} → ${metrics.control_points}</dd><dt>Solver</dt><dd>${metrics.solver_success ? "Converged" : "Rejected"}</dd>`;
}

function reset() {
  running = false;
  elapsed = 0;
  last = 0;
  $("play").textContent = "Play";
  update(0);
}

function sampleAtTime(time) {
  const lapTime = data[active].lap_time_s;
  if (time >= lapTime - 1e-6) return 0;
  let accumulated = 0;
  for (let index = 0; index < data[active].segment_time_s.length; index += 1) {
    accumulated += data[active].segment_time_s[index];
    if (time < accumulated) return index;
  }
  return 0;
}

function update(time) {
  const index = sampleAtTime(time);
  const position = map(widenForDisplay(linePoints()))[index];
  car.setAttribute("transform", `translate(${position[0]} ${position[1]})`);
  const acceleration = data[active].acceleration_mps2[index];
  const speed = data[active].speed_mps[index];
  $("timer").textContent = format(time);
  $("speed").innerHTML = `${Math.round(speed * 3.6)} <small>km/h</small>`;
  $("state").textContent = acceleration < -0.15 ? "BRAKING" : acceleration > 0.15 ? "ACCELERATING" : "CORNERING";
  $("state").style.color = acceleration < -0.15 ? "#ff5c6b" : acceleration > 0.15 ? "#37d9af" : "#f8c64e";
  updateProfileMarker(time);
}

function tick(now) {
  if (!running) return;
  if (!last) last = now;
  elapsed += ((now - last) / 1000) * rate * 6;
  last = now;
  if (elapsed >= data[active].lap_time_s) {
    elapsed = data[active].lap_time_s;
    running = false;
    $("play").textContent = "Replay";
  }
  update(elapsed);
  if (running) requestAnimationFrame(tick);
}
