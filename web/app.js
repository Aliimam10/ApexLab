const svg = document.querySelector("#track"), ns = "http://www.w3.org/2000/svg";
const $ = (id) => document.querySelector(`#${id}`);
let data, active = "optimised", running = false, elapsed = 0, last = 0, rate = 1, car;
// Presentation only: preserve the computed trajectory, but make approach,
// apex, and exit placement legible in the replay.
const DISPLAY_WIDTH_SCALE = 3;
const node = (name, attrs = {}) => { const n = document.createElementNS(ns, name); Object.entries(attrs).forEach(([k,v]) => n.setAttribute(k,v)); return n; };
const format = (s) => `${Math.floor(s/60)}:${(s%60).toFixed(3).padStart(6,"0")}`;
const speedColour = (v, min, max) => `hsl(${260 - 210 * ((v-min)/(max-min || 1))} 82% 58%)`;

fetch("data/silverstone.json")
  .then(r => { if (!r.ok) throw new Error(`Simulation data request failed (${r.status}).`); return r.json(); })
  .then(init)
  .catch(error => { console.error(error); document.body.innerHTML = `<p style='padding:2rem'>Could not initialise the replay: <code>${error.message}</code></p>`; });
function init(payload) {
  data = payload; drawStatic(); updateValidation(); reset();
  $("play").onclick = () => { running = !running; $("play").textContent = running ? "Pause" : "Play"; if(running) { last = 0; requestAnimationFrame(tick); } };
  $("reset").onclick = reset;
  $("rate").oninput = e => { rate = +e.target.value; $("rate-output").textContent = `${rate}×`; };
  $("line-select").onchange = e => { active = e.target.value; $("line-label").textContent = active === "optimised" ? "OPTIMISED LINE" : "CENTRELINE BASELINE"; drawStatic(); reset(); };
  $("overlays").onchange = () => { $("legend").hidden = !$("overlays").checked; drawStatic(); update(elapsed); };
}
function widenForDisplay(points) { return points.map(([x,y],i) => { const [cx,cy] = data.track.centreline[i]; return [cx + (x-cx)*DISPLAY_WIDTH_SCALE, cy + (y-cy)*DISPLAY_WIDTH_SCALE]; }); }
function map(points) { const all = [...widenForDisplay(data.track.left_boundary),...widenForDisplay(data.track.right_boundary)]; const xs=all.map(p=>p[0]),ys=all.map(p=>p[1]), pad=45, w=1000,h=760, sx=(w-2*pad)/(Math.max(...xs)-Math.min(...xs)),sy=(h-2*pad)/(Math.max(...ys)-Math.min(...ys)),s=Math.min(sx,sy), ox=(w-s*(Math.max(...xs)-Math.min(...xs)))/2-s*Math.min(...xs),oy=(h-s*(Math.max(...ys)-Math.min(...ys)))/2-s*Math.min(...ys); return points.map(([x,y])=>[x*s+ox,h-(y*s+oy)]); }
function drawPath(points, colour, width, dash="") { const p=map(points); svg.appendChild(node("path",{d:`M${p.map(x=>x.join(",")).join(" L")}`,fill:"none",stroke:colour,"stroke-width":width,"stroke-dasharray":dash})); return p; }
function linePoints(){return active==="optimised"?data.optimised.line:data.track.centreline;}
function drawStatic() { while(svg.firstChild) svg.removeChild(svg.firstChild); const right=widenForDisplay(data.track.right_boundary), left=widenForDisplay(data.track.left_boundary), showOverlays=$("overlays").checked; const road=map([...right,...left.slice().reverse()]); svg.appendChild(node("path",{d:`M${road.map(x=>x.join(",")).join(" L")} Z`,fill:"#6d7480",stroke:"#3c4555","stroke-width":1})); if(showOverlays){ drawPath(right,"#377dff",2); drawPath(left,"#f8c64e",2); if(active==="optimised") drawPath(data.track.centreline,"#c7d1e0",1,"5 5"); } const p = map(widenForDisplay(linePoints())), speeds=data[active].speed_mps, lo=Math.min(...speeds),hi=Math.max(...speeds); p.forEach((a,i)=>svg.appendChild(node("line",{x1:a[0],y1:a[1],x2:p[(i+1)%p.length][0],y2:p[(i+1)%p.length][1],stroke:showOverlays?speedColour(speeds[i],lo,hi):"#f8fafc","stroke-width":showOverlays?2.2:0.85,"stroke-linecap":"round"}))); car=node("g"); car.appendChild(node("circle",{r:8,fill:"#fff",stroke:"#07101d","stroke-width":3})); car.appendChild(node("path",{d:"M-3,-4 L7,0 L-3,4 Z",fill:"#061320"})); svg.appendChild(car); }
function updateValidation(){ const m=data.metrics, converged=Boolean(m.solver_success); $("gain").textContent=`${m.improvement_s.toFixed(3)} s`; $("validation").innerHTML=`<dt>Baseline lap</dt><dd>${format(data.baseline.lap_time_s)}</dd><dt>Fastest line</dt><dd>${format(data.optimised.lap_time_s)}</dd><dt>Improvement</dt><dd>${m.improvement_percent.toFixed(2)}%</dd><dt>Min. wheel clearance</dt><dd>${m.minimum_clearance_m.toFixed(2)} m</dd><dt>Curb tyre use</dt><dd>${(m.curb_wheel_fraction*100).toFixed(1)}%</dd><dt>Max tyre use</dt><dd>${m.maximum_tyre_utilisation.toFixed(3)}</dd><dt>Controls</dt><dd>${m.coarse_control_points} → ${m.control_points}</dd><dt>Solver</dt><dd>${converged ? "Converged" : "Rejected"}</dd>`; }
function reset(){ running=false; elapsed=0; last=0; $("play").textContent="Play"; update(0); }
function sampleAtTime(t){ const lap=data[active].lap_time_s; if(t >= lap - 1e-6) return 0; let elapsedTime=0; for(let i=0;i<data[active].segment_time_s.length;i++){ elapsedTime += data[active].segment_time_s[i]; if(t < elapsedTime) return i; } return 0; }
function update(t){ const i=sampleAtTime(t), q=map(widenForDisplay(linePoints()))[i]; car.setAttribute("transform",`translate(${q[0]} ${q[1]})`); const a=data[active].acceleration_mps2[i],v=data[active].speed_mps[i]; $("timer").textContent=format(t); $("speed").innerHTML=`${Math.round(v*3.6)} <small>km/h</small>`; $("state").textContent=a<-0.15?"BRAKING":a>0.15?"ACCELERATING":"CORNERING"; $("state").style.color=a<-0.15?"#ff5c6b":a>0.15?"#37d9af":"#f8c64e"; }
function tick(now){ if(!running)return; if(!last)last=now; elapsed+=((now-last)/1000)*rate*6; last=now; if(elapsed>=data[active].lap_time_s){elapsed=data[active].lap_time_s;running=false;$("play").textContent="Replay";} update(elapsed); if(running)requestAnimationFrame(tick); }
