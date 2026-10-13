const svg = document.querySelector("#track"), ns = "http://www.w3.org/2000/svg";
const $ = (id) => document.querySelector(id);
let data, active = "optimised", running = false, elapsed = 0, last = 0, rate = 1;
const node = (name, attrs = {}) => { const n = document.createElementNS(ns, name); Object.entries(attrs).forEach(([k,v]) => n.setAttribute(k,v)); return n; };
const format = (s) => `${Math.floor(s/60)}:${(s%60).toFixed(3).padStart(6,"0")}`;
const speedColour = (v, min, max) => `hsl(${260 - 210 * ((v-min)/(max-min || 1))} 82% 58%)`;

fetch("data/silverstone.json").then(r => r.json()).then(init).catch(() => { document.body.innerHTML = "<p style='padding:2rem'>Missing simulation data. Run <code>uv run python -m racing_line.export_web_data</code> from ApexLab first.</p>"; });
function init(payload) {
  data = payload; drawStatic(); updateValidation(); reset();
  $("play").onclick = () => { running = !running; $("play").textContent = running ? "Pause" : "Play"; if(running) requestAnimationFrame(tick); };
  $("reset").onclick = reset;
  $("rate").oninput = e => { rate = +e.target.value; $("rate-output").textContent = `${rate}×`; };
  $("line-select").onchange = e => { active = e.target.value; $("line-label").textContent = active === "optimised" ? "OPTIMISED LINE" : "CENTRELINE BASELINE"; drawStatic(); reset(); };
}
function map(points) { const all = [...data.track.left_boundary,...data.track.right_boundary]; const xs=all.map(p=>p[0]),ys=all.map(p=>p[1]), pad=45, w=1000,h=760, sx=(w-2*pad)/(Math.max(...xs)-Math.min(...xs)),sy=(h-2*pad)/(Math.max(...ys)-Math.min(...ys)),s=Math.min(sx,sy), ox=(w-s*(Math.max(...xs)-Math.min(...xs)))/2-s*Math.min(...xs),oy=(h-s*(Math.max(...ys)-Math.min(...ys)))/2-s*Math.min(...ys); return points.map(([x,y])=>[x*s+ox,h-(y*s+oy)]); }
function drawPath(points, colour, width, dash="") { const p=map(points); svg.append(node("path",{d:`M${p.map(x=>x.join(",")).join(" L")}`,fill:"none",stroke:colour,"stroke-width":width,"stroke-dasharray":dash})); return p; }
function linePoints(){return active==="optimised"?data.optimised.line:data.track.centreline;}
function drawStatic() { svg.replaceChildren(); drawPath(data.track.right_boundary,"#377dff",2); drawPath(data.track.left_boundary,"#f8c64e",2); const p = map(linePoints()), speeds=data[active].speed_mps, lo=Math.min(...speeds),hi=Math.max(...speeds); p.forEach((a,i)=>svg.append(node("line",{x1:a[0],y1:a[1],x2:p[(i+1)%p.length][0],y2:p[(i+1)%p.length][1],stroke:speedColour(speeds[i],lo,hi),"stroke-width":active==="optimised"?5:3,"stroke-linecap":"round"}))); if(active==="optimised") drawPath(data.track.centreline,"#c7d1e0",1,"5 5"); window.car=node("g"); window.car.append(node("circle",{r:10,fill:"#fff",stroke:"#07101d","stroke-width":4})); window.car.append(node("path",{d:"M-4,-5 L8,0 L-4,5 Z",fill:"#061320"})); svg.append(window.car); }
function updateValidation(){ const m=data.metrics; $("gain").textContent=`${m.improvement_s.toFixed(3)} s`; $("validation").innerHTML=`<dt>Baseline lap</dt><dd>${format(data.baseline.lap_time_s)}</dd><dt>Fastest line</dt><dd>${format(data.optimised.lap_time_s)}</dd><dt>Improvement</dt><dd>${m.improvement_percent.toFixed(2)}%</dd><dt>Min. clearance</dt><dd>${m.minimum_clearance_m.toFixed(2)} m</dd><dt>Max tyre use</dt><dd>${m.maximum_tyre_utilisation.toFixed(3)}</dd><dt>Control points</dt><dd>${m.control_points}</dd>`; }
function reset(){ running=false; elapsed=0; $("play").textContent="Play"; update(0); }
function update(t){ const lap=data[active].lap_time_s, f=(t%lap)/lap, i=Math.floor(f*data[active].speed_mps.length); const q=map(linePoints())[i]; window.car.setAttribute("transform",`translate(${q[0]} ${q[1]})`); const a=data[active].acceleration_mps2[i],v=data[active].speed_mps[i]; $("timer").textContent=format(t); $("speed").innerHTML=`${Math.round(v*3.6)} <small>km/h</small>`; $("state").textContent=a<-0.15?"BRAKING":a>0.15?"ACCELERATING":"CORNERING"; $("state").style.color=a<-0.15?"#ff5c6b":a>0.15?"#37d9af":"#f8c64e"; }
function tick(now){ if(!running)return; if(!last)last=now; elapsed+=((now-last)/1000)*rate*6; last=now; if(elapsed>=data[active].lap_time_s){elapsed=data[active].lap_time_s;running=false;$("play").textContent="Replay";} update(elapsed); if(running)requestAnimationFrame(tick); }
