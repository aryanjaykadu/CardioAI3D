// CardioAI - app.js. Libraries live in ./lib, data in ./data (index.html maps 'three' to ./lib/three.module.js).
import * as THREE from './lib/three.module.js';
import { OrbitControls } from './lib/OrbitControls.js';

const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const fine = matchMedia('(hover: hover) and (pointer: fine)').matches;
const lvl = p => (p >= 65 ? 'high' : p >= 40 ? 'mid' : 'ok');
const LEVEL_COLOR = { ok: 0x30d158, mid: 0xff9f0a, high: 0xe11d48 };
const f1 = n => (Math.round(n * 10) / 10).toFixed(1);
function tween(el, to) { // eased count between values; instant under reduced motion
  const from = +el.dataset.v || 0; el.dataset.v = to; cancelAnimationFrame(el._r);
  if (reduced) { el.textContent = f1(to) + '%'; return; }
  const t0 = performance.now();
  const step = now => { const k = Math.min(1, (now - t0) / 450), e = 1 - (1 - k) ** 3; el.textContent = f1(from + (to - from) * e) + '%'; if (k < 1) el._r = requestAnimationFrame(step); };
  el._r = requestAnimationFrame(step);
}

/* ---------- Simulator (illustrative logistic deltas, NOT trained coefficients) ---------- */
const SIM = { // key: [slider id, logit change per unit, SHAP feature name, measurement key]
  age: ['w-age', 0.03, 'Age', 'Age'], ef: ['w-ef', -0.06, 'EF-TTE', 'EF_TTE'],
  rwma: ['w-rwma', 0.6, 'Region RWMA', 'Region_RWMA'], bp: ['w-bp', 0.01, 'BP', 'BP'],
  tg: ['w-tg', 0.004, 'TG', 'TG'], esr: ['w-esr', 0.01, 'ESR', 'ESR'],
  cp: ['w-cp', 0.9, 'Typical Chest Pain', 'Typical_Chest_Pain'], dm: ['w-dm', 0.7, 'DM', 'DM'],
};
const VESSEL_GAIN = { CAD: 1, LAD: 1, LCX: 0.8, RCA: 0.8 };
const logit = p => Math.log(p / (100 - p));
const sigm = z => 100 / (1 + Math.exp(-z));
const clampP = p => Math.min(99.5, Math.max(0.5, p));

let patients = [], cur = null, baseline = {}, state = {}, selected = null;

const readSlider = id => { const el = document.getElementById(id); return el.type === 'checkbox' ? +el.checked : +el.value; };
const writeSlider = (id, v) => { const el = document.getElementById(id); el.type === 'checkbox' ? (el.checked = !!v) : (el.value = v); const o = $(`output[for="${id}"]`); if (o) o.textContent = v; };

function compute() {
  const d = {}; let z = 0;
  for (const k in SIM) { d[k] = readSlider(SIM[k][0]) - baseline[k]; z += SIM[k][1] * d[k]; }
  const out = {};
  for (const n of ['CAD', 'LAD', 'LCX', 'RCA']) {
    const p0 = clampP(cur.predictions[n].probability);
    out[n] = clampP(sigm(logit(p0) + z * VESSEL_GAIN[n]));
  }
  const shap = cur.shap_contributions.map(s => {
    const k = Object.keys(SIM).find(x => SIM[x][2] === s.feature);
    return { feature: s.feature, v: s.importance + (k ? SIM[k][1] * d[k] : 0) };
  }).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const changed = Object.keys(d).some(k => d[k] !== 0);
  return { out, shap, changed };
}

/* ---------- UI render ---------- */
const ROWS = { Age: 'Age', BP: 'BP', HR: 'PR', 'EF-TTE': 'EF_TTE', Region_RWMA: 'Region_RWMA', Typical_Chest_Pain: 'Typical_Chest_Pain', DM: 'DM', TG: 'TG', ESR: 'ESR' };
const YN = v => (v ? 'Yes' : 'No');

function renderMeasurements() {
  const live = k => (SIM_BY_M[k] ? readSlider(SIM_BY_M[k]) : cur.measurements[k]);
  for (const [row, key] of Object.entries(ROWS)) {
    const td = $(`#biomarker-rows tr[data-key="${row}"] td`); if (!td) continue;
    const v = live(key);
    td.textContent = ['Typical_Chest_Pain', 'DM'].includes(row) ? YN(v) : v;
    td.style.color = (row === 'Region_RWMA' && v >= 2) ? 'var(--alert)' : '';
  }
}
const SIM_BY_M = Object.fromEntries(Object.values(SIM).map(([id, , , m]) => [m, id]));

function render() {
  const { out, shap, changed } = compute();
  renderMeasurements();
  const chip = $('#risk-chip'); chip.dataset.level = lvl(out.CAD); tween($('#cad-risk'), out.CAD);
  $('#panel-cad').dataset.level = lvl(out.CAD); tween($('#cad-card-pct'), out.CAD);
  $('#cad-status').textContent = changed ? 'Simulated risk' : cur.predictions.CAD.status;
  $('#cad-sev').textContent = changed ? 'Differs from catheterization baseline' : `${cur.predictions.CAD.severity}. Cath result: ${cur.predictions.CAD.actual}`;
  $$('.vcard').forEach(c => {
    const n = c.dataset.vessel, p = cur.predictions[n]; c.dataset.level = lvl(out[n]);
    tween($('.vcard__pct', c), out[n]);
    $('.vcard__status', c).textContent = changed ? 'Simulated risk' : p.status;
    $('.vcard__sev', c).textContent = changed ? 'Differs from catheterization baseline' : `${p.severity}. Cath result: ${p.actual}`;
    c.classList.toggle('is-selected', selected === n);
  });
  let cum = 0; const rows = shap.map(r => { const a = cum; cum += r.v; return { ...r, a, b: cum }; });
  const lo = Math.min(0, ...rows.map(r => Math.min(r.a, r.b))), hi = Math.max(0, ...rows.map(r => Math.max(r.a, r.b))), span = (hi - lo) || 1;
  const list = $('#shap-list'), keep = new Set(rows.map(r => r.feature));
  $$('.shap__row', list).forEach(li => keep.has(li.dataset.f) || li.remove());
  rows.forEach((r, i) => {
    let li = $(`.shap__row[data-f="${CSS.escape(r.feature)}"]`, list);
    if (!li) { li = document.createElement('li'); li.className = 'shap__row'; li.dataset.f = r.feature; li.innerHTML = '<span></span><span class="shap__track"><i></i></span><b class="num"></b>'; }
    li.children[0].textContent = r.feature; li.children[2].textContent = (r.v > 0 ? '+' : '') + r.v.toFixed(2);
    const bar = li.querySelector('i');
    bar.style.left = (Math.min(r.a, r.b) - lo) / span * 100 + '%'; bar.style.width = Math.abs(r.v) / span * 100 + '%'; bar.style.background = r.v < 0 ? 'var(--green)' : 'var(--ruby)';
    if (list.children[i] !== li) list.insertBefore(li, list.children[i] || null);
  });
  let tot = $('.shap__total', list);
  if (!tot) { tot = document.createElement('li'); tot.className = 'shap__total'; tot.innerHTML = '<span>Net contribution</span><b class="num"></b>'; }
  $('b', tot).textContent = (cum > 0 ? '+' : '') + cum.toFixed(2); list.appendChild(tot);
  const dl = out.CAD - clampP(cur.predictions.CAD.probability);
  $('#whatif-delta').textContent = changed ? `Simulated change vs. baseline: ${dl > 0 ? '+' : ''}${f1(dl)} pts` : 'Change vs. baseline: 0.0 pts';
  heart && heart.update(out);
}

function selectPatient(i) {
  cur = patients[i]; if (!cur) return;
  $$('#patient-tabs .seg__btn').forEach((b, j) => b.setAttribute('aria-selected', j === i));
  $('#patient-id').textContent = `Patient ${i + 1}`; $('#panel-patient .muted').textContent = cur.title.replace(/^Patient Case \d+:\s*/, '');
  for (const k in SIM) { baseline[k] = cur.measurements[SIM[k][3]]; writeSlider(SIM[k][0], baseline[k]); }
  selected = null; setECG(); render();
}

/* ---------- ECG oscilloscope (schematic Lead II) ---------- */
const ecg = { hr: 70, st: 0, color: '#30d158' };
function setECG() {
  const e = cur.shap_contributions.find(x => x.feature === 'St Elevation');
  ecg.hr = cur.measurements.PR || 70; ecg.st = e && e.importance > 0 ? Math.min(5, e.importance) : 0; // mm, taken from the attribution
  const lv = ecg.st >= 2 ? 'high' : ecg.st >= 0.5 ? 'mid' : 'ok';
  ecg.color = { high: '#ff4d6d', mid: '#ff9f0a', ok: '#30d158' }[lv];
  const b = $('#ecg-badge'); b.dataset.level = lv; b.textContent = { high: 'STEMI Pattern', mid: 'ST Elevation', ok: 'Normal Sinus Rhythm' }[lv];
  $('#ecg-hr').textContent = ecg.hr + ' bpm'; $('#ecg-st').textContent = (ecg.st > 0 ? '+' : '') + ecg.st.toFixed(1) + ' mm';
}
function initECG() {
  const cv = $('#ecg-canvas'), ctx = cv.getContext('2d'), PX = 4, SPEED = 25 * PX; // 25 mm/s at 4 css px per mm; 10 mm/mV
  let w = 0, h = 0, x = 0, prev = null, tSim = 0, last = performance.now();
  const size = () => { const r = cv.getBoundingClientRect(), d = Math.min(devicePixelRatio, 2); w = r.width; h = r.height; cv.width = Math.round(w * d); cv.height = Math.round(h * d); ctx.setTransform(d, 0, 0, d, 0, 0); x = 0; prev = null; };
  new ResizeObserver(size).observe(cv);
  const gs = (u, c, k) => Math.exp(-(((u - c) / k) ** 2)), sg = z => 1 / (1 + Math.exp(-z));
  const mv = tt => { // millivolts: P, Q, R, S, ST segment (elevated by ecg.st mm), T
    const rr = 60 / ecg.hr, u = (tt % rr) - Math.min(0.32, rr * 0.4), st = ecg.st / 10;
    return 0.15 * gs(u, -0.2, 0.035) - 0.12 * gs(u, -0.035, 0.012) + 1.1 * gs(u, 0, 0.014) - 0.28 * gs(u, 0.032, 0.014) + st * (sg((u - 0.05) / 0.01) - sg((u - 0.26) / 0.03)) + (0.3 + st * 0.8) * gs(u, 0.28, 0.06);
  };
  const frame = now => {
    requestAnimationFrame(frame);
    const dt = Math.min(0.05, (now - last) / 1000); last = now; if (!w) return;
    ctx.globalCompositeOperation = 'destination-out'; ctx.fillStyle = 'rgba(0,0,0,0.05)'; ctx.fillRect(0, 0, w, h); ctx.globalCompositeOperation = 'source-over'; // phosphor decay
    const n = Math.max(1, Math.ceil(SPEED * dt / 2)), base = h * 0.62; let y = base;
    ctx.strokeStyle = ecg.color; ctx.lineWidth = 1.6; ctx.lineJoin = 'round'; ctx.shadowColor = ecg.color; ctx.shadowBlur = 8;
    for (let i = 0; i < n; i++) {
      const nx = x + SPEED * dt / n; tSim += dt / n; y = base - mv(tSim) * 10 * PX;
      if (nx >= w) { x = 0; prev = null; continue; }
      if (prev !== null) { ctx.beginPath(); ctx.moveTo(x, prev); ctx.lineTo(nx, y); ctx.stroke(); }
      x = nx; prev = y;
    }
    ctx.shadowBlur = 0; ctx.clearRect(x + 3, 0, 14, h); // erase bar ahead of the tracer
    ctx.fillStyle = '#fff'; ctx.shadowColor = ecg.color; ctx.shadowBlur = 12; ctx.beginPath(); ctx.arc(x, y, 2.2, 0, 6.283); ctx.fill(); ctx.shadowBlur = 0; // neon tracer
  };
  requestAnimationFrame(frame);
}

/* ---------- 3D heart (procedural anatomy) ---------- */
let heart = null;
const V = a => (Array.isArray(a) ? new THREE.Vector3(...a) : a.clone());
const cap = (t, e0, e1) => Math.min(1, Math.sqrt(Math.max(t, 0) / e0)) * Math.min(1, Math.sqrt(Math.max(1 - t, 0) / e1)); // closes/rounds a loft's ends
// Lofts an elliptical cross-section along a spline; prof(t) -> [width radius, depth radius]; depth faces anterior (+z)
function loft(pts, prof, seg = 72, rad = 40) {
  const curve = new THREE.CatmullRomCurve3(pts.map(V)), pos = [], idx = [], Z = new THREE.Vector3(0, 0, 1), Y = new THREE.Vector3(0, 1, 0);
  for (let i = 0; i <= seg; i++) {
    const t = i / seg, P = curve.getPointAt(t), T = curve.getTangentAt(t), up = Math.abs(T.dot(Z)) > 0.9 ? Y : Z;
    const S = new THREE.Vector3().crossVectors(T, up).normalize(), D = new THREE.Vector3().crossVectors(S, T).normalize(), [ra, rb] = prof(t);
    for (let j = 0; j <= rad; j++) { const q = (j / rad) * Math.PI * 2, c = Math.cos(q) * ra, s = Math.sin(q) * rb; pos.push(P.x + S.x * c + D.x * s, P.y + S.y * c + D.y * s, P.z + S.z * c + D.z * s); }
  }
  for (let i = 0; i < seg; i++) for (let j = 0; j < rad; j++) { const a = i * (rad + 1) + j, b = a + rad + 1; idx.push(a, b, a + 1, b, b + 1, a + 1); }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals(); return g;
}
const AO = new THREE.CatmullRomCurve3([[0.02, 0.68, 0.08], [-0.02, 1.2, 0.08], [0.05, 1.65, 0], [0.4, 1.92, -0.2], [0.8, 1.7, -0.4], [0.9, 1.2, -0.5]].map(V));
const arch = (t, d, r) => { const p = AO.getPointAt(t); return [[p, new THREE.Vector3(p.x + d[0] / 2, p.y + d[1] / 2, p.z), new THREE.Vector3(p.x + d[0], p.y + d[1], p.z)], u => { const k = cap(u, 0.0001, 0.15); return [r * k, r * k]; }, 32, 16]; };
const vessel = (pts, r) => [pts, u => { const k = cap(u, 0.0001, 0.12); return [r * k, r * k]; }, 32, 16];
const ANATOMY = [
  [[[0.15, 0.62, -0.2], [0.3, 0.05, 0], [0.5, -0.7, 0.2], [0.68, -1.38, 0.4]], t => { const k = (1 - 0.9 * t ** 1.45) * cap(t, 0.05, 0.02); return [0.92 * k, 0.78 * k]; }], // left ventricle: conical, curving anteriorly to the apex
  [[[-0.55, 0.6, 0.35], [-0.5, 0, 0.5], [-0.2, -0.7, 0.55], [0.15, -1.2, 0.5]], t => { const k = (1 - 0.7 * t ** 1.6) * cap(t, 0.1, 0.1); return [0.42 * k, 0.3 * k]; }], // right ventricle: crescent over the septum
  [[[-0.78, 0.1, -0.05], [-0.82, 0.55, -0.05], [-0.78, 1, -0.02]], t => { const k = cap(t, 0.3, 0.3); return [0.36 * k, 0.38 * k]; }], // right atrium
  [[[0, 0.95, -0.55], [0.5, 1, -0.55], [0.95, 0.9, -0.45]], t => { const k = cap(t, 0.3, 0.3); return [0.33 * k, 0.38 * k]; }], // left atrium (posterior)
  [[[0.9, 0.95, -0.1], [1, 0.85, 0.2], [0.85, 0.65, 0.35]], t => { const k = cap(t, 0.3, 0.3); return [0.12 * k, 0.12 * k]; }, 32, 16], // left atrial appendage
  [[[-0.6, 0.95, 0.15], [-0.4, 1.1, 0.4], [-0.25, 0.95, 0.55]], t => { const k = cap(t, 0.3, 0.3); return [0.13 * k, 0.13 * k]; }, 32, 16], // right atrial appendage
  [AO.points, t => { const k = (0.2 - 0.03 * t) * cap(t, 0.0001, 0.05); return [k, k]; }, 96, 32], // ascending aorta and arch
  arch(0.4, [-0.12, 0.5], 0.075), arch(0.52, [0, 0.52], 0.058), arch(0.64, [0.2, 0.42], 0.068), // brachiocephalic, left carotid, left subclavian
  [[[-0.2, 0.6, 0.55], [-0.12, 1.1, 0.55], [0, 1.45, 0.4]], t => { const k = (0.28 - 0.1 * t) * cap(t, 0.0001, 0.0001); return [k, k]; }, 48, 24], // pulmonary trunk from the infundibulum
  vessel([[0, 1.45, 0.4], [0.5, 1.6, 0.05], [0.95, 1.5, -0.3]], 0.1), vessel([[0, 1.45, 0.4], [-0.5, 1.6, 0.05], [-0.95, 1.4, -0.3]], 0.1), // left / right pulmonary arteries
];
// Coronary trees: waypoints [x, y, z, mode]; mode names the axis the ray is cast along to find the shell surface
const MODES = { z: [2, 1], zb: [2, -1], x: [0, 1], xn: [0, -1] };
const TREES = {
  LAD: { main: [[0.05, 0.62, 0, 'z'], [0.12, 0.25, 0, 'z'], [0.25, -0.2, 0, 'z'], [0.4, -0.7, 0, 'z'], [0.55, -1.15, 0, 'z'], [0.6, -1.32, 0, 'z']] },
  LCX: { main: [[0.25, 0.58, 0, 'z'], [0.7, 0.5, 0, 'z'], [0, 0.3, -0.05, 'x'], [0, -0.1, -0.35, 'x'], [0, -0.55, -0.45, 'x']],
    br: [[0.55, [[0, -0.35, -0.2, 'x'], [0, -0.85, -0.3, 'x']], 0.022], [0.78, [[0, -0.85, -0.45, 'x'], [0.2, -1.15, -0.3, 'x']], 0.02]] }, // obtuse marginals
  RCA: { main: [[-0.1, 0.58, 0, 'z'], [-0.6, 0.42, 0, 'z'], [-0.92, 0, 0, 'z'], [-0.9, -0.45, 0, 'z'], [-0.45, -0.8, 0, 'zb']],
    br: [[0.5, [[-0.65, -0.45, 0, 'z'], [-0.35, -0.95, 0, 'z']], 0.022], [1, [[-0.1, -0.9, 0, 'zb'], [0.25, -1.15, 0, 'zb']], 0.024]] }, // acute marginal, posterior descending
};
const VSHADER = {
  vertexShader: 'varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }',
  fragmentShader: 'varying vec2 vUv; uniform vec3 uColor; uniform float uTime,uFlow,uGain; void main(){ float p=sin(vUv.x*16.-uTime*3.)*.5+.5; float g=.45+uFlow*.9*p*p; gl_FragColor=vec4(uColor*g*uGain*2.2,1.); }',
};

const HALO = {
  vertexShader: 'uniform float uHalo; varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position+normal*uHalo,1.); }',
  fragmentShader: 'varying vec2 vUv; uniform vec3 uColor; uniform float uTime,uFlow,uGain; void main(){ float p=sin(vUv.x*16.-uTime*3.)*.5+.5; float g=(.35+uFlow*.65*p*p)*uGain; gl_FragColor=vec4(uColor*g,.7*g); }',
};
const DROP = 0.1; // heart sits 10% lower in its frame, clear of the CAD risk badge

function initHeart() {
  const host = $('#scene'), W = () => innerWidth, H = () => innerHeight;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setSize(W(), H());
  renderer.toneMapping = THREE.ACESFilmicToneMapping; host.appendChild(renderer.domElement);
  const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(35, W() / H(), 0.1, 60); cam.position.set(0, 0.2, 9);

  // Procedural equirectangular studio map painted on a canvas, prefiltered by PMREM for sharp glints
  const ec = document.createElement('canvas'); ec.width = 1024; ec.height = 512; const cx = ec.getContext('2d');
  const sky = cx.createLinearGradient(0, 0, 0, 512); sky.addColorStop(0, '#1c1c30'); sky.addColorStop(0.5, '#08080f'); sky.addColorStop(1, '#030305');
  cx.fillStyle = sky; cx.fillRect(0, 0, 1024, 512);
  [[110, 80, 230, 60, '#ffffff'], [560, 60, 270, 44, '#ffffff'], [850, 140, 70, 170, '#a78bfa'], [330, 250, 60, 130, '#ff5577'], [690, 330, 210, 30, '#6366f1']].forEach(([px, py, w, h, c]) => { cx.shadowColor = c; cx.shadowBlur = 24; cx.fillStyle = c; cx.fillRect(px, py, w, h); });
  const etex = new THREE.CanvasTexture(ec); etex.mapping = THREE.EquirectangularReflectionMapping; etex.colorSpace = THREE.SRGBColorSpace;
  scene.environment = new THREE.PMREMGenerator(renderer).fromEquirectangular(etex).texture; etex.dispose();

  const controls = new OrbitControls(cam, renderer.domElement);
  Object.assign(controls, { enableDamping: true, dampingFactor: 0.05, enablePan: false, minDistance: 3, maxDistance: 13, autoRotate: !reduced, autoRotateSpeed: 0.6 });
  controls.target.set(0, 0.2, 0);
  let goal = null, tgt = null;
  controls.addEventListener('start', () => { goal = tgt = null; controls.autoRotate = false; });

  const group = new THREE.Group(), BASE = 0.92; scene.add(group);
  const shellMat = new THREE.MeshPhysicalMaterial({ color: 0xf43f5e, transmission: 0.82, roughness: 0.08, ior: 1.48, thickness: 3.2, clearcoat: 1, clearcoatRoughness: 0.04, attenuationColor: new THREE.Color(0x9f1239), attenuationDistance: 5, iridescence: 0.2, envMapIntensity: 1.6 });
  const shell = new THREE.Group(), shellMeshes = []; group.add(shell);
  ANATOMY.forEach(([pts, prof, seg, rad]) => { const m = new THREE.Mesh(loft(pts, prof, seg, rad), shellMat); shell.add(m); shellMeshes.push(m); });

  // Coronary arteries hug the glass: waypoints are ray-cast onto the shell, then sunk slightly beneath it
  group.updateMatrixWorld(true); // group scale is still 1 here, so snapped points are in heart space
  const ray = new THREE.Raycaster();
  const snap = ([x, y, z, m]) => {
    const [ax, sg] = MODES[m], o = new THREE.Vector3(x, y, z), d = new THREE.Vector3(); o.setComponent(ax, 6 * sg); d.setComponent(ax, -sg);
    ray.set(o, d); const h = ray.intersectObjects(shellMeshes, false)[0];
    if (h) return h.point.clone().addScaledVector(h.face.normal, -0.07);
    const p = new THREE.Vector3(x, y, z); p.setComponent(ax, 0.4 * sg); return p;
  };
  const mk = (curve, r, seg, rad, u) => {
    const geo = new THREE.TubeGeometry(curve, seg, r, rad, false);
    group.add(new THREE.Mesh(geo, new THREE.ShaderMaterial({ ...VSHADER, uniforms: u })));
    const halo = new THREE.Mesh(geo, new THREE.ShaderMaterial({ ...HALO, uniforms: { ...u, uHalo: { value: r * 2 } }, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); halo.renderOrder = 2; group.add(halo);
    return geo;
  };
  const vessels = {};
  for (const [name, def] of Object.entries(TREES)) {
    const curve = new THREE.CatmullRomCurve3(def.main.map(snap)), seg = 96, rad = 10;
    const u = { uColor: { value: new THREE.Color(0x30d158) }, uTime: { value: 0 }, uFlow: { value: 1 }, uGain: { value: 1 } };
    const geo = mk(curve, 0.045, seg, rad, u), base = geo.attributes.position.array.slice(), centers = Array.from({ length: seg + 1 }, (_, i) => curve.getPointAt(i / seg));
    (def.br || []).forEach(([t, pts, r]) => mk(new THREE.CatmullRomCurve3([curve.getPointAt(t), ...pts.map(snap)]), r, 40, 8, u));
    if (name === 'LAD') [0.2, 0.35, 0.5, 0.65].forEach(t => { const p = curve.getPointAt(t); mk(new THREE.CatmullRomCurve3([p, p.clone().add(V([0.05, -0.04, -0.2])), p.clone().add(V([0.14, -0.1, -0.45]))]), 0.016, 24, 6, u); }); // septal perforators
    const node = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 12), new THREE.MeshBasicMaterial({ color: 0x30d158, transparent: true, opacity: 0.9 }));
    node.position.copy(curve.getPointAt(0.45)); group.add(node); // lesion node
    const label = document.createElement('div'); label.className = 'vlabel glass'; $('#vessel-labels').appendChild(label);
    vessels[name] = { curve, geo, base, centers, seg, rad, u, label, node, tc: new THREE.Color(0x30d158), s: 0, sT: 0 };
  }
  function pinch(v, s) {
    const a = v.geo.attributes.position, per = v.rad + 1;
    for (let i = 0; i < a.count; i++) {
      const k = Math.floor(i / per), t = k / v.seg, c = v.centers[k];
      const f = (1 - 0.45 * t) * (1 - s * Math.exp(-(((t - 0.45) / 0.08) ** 2)));
      a.setXYZ(i, c.x + (v.base[i * 3] - c.x) * f, c.y + (v.base[i * 3 + 1] - c.y) * f, c.z + (v.base[i * 3 + 2] - c.z) * f);
    }
    a.needsUpdate = true;
  }

  function fit() { // centre heart and telemetry rings together inside the viewport frame
    renderer.setSize(W(), H()); cam.aspect = W() / H();
    const r = $('.viewport').getBoundingClientRect(), cy = r.top + r.height * (0.5 + DROP), st = document.documentElement.style;
    cam.setViewOffset(W(), H(), W() / 2 - (r.left + r.width / 2), H() / 2 - cy, W(), H()); cam.updateProjectionMatrix();
    st.setProperty('--tcx', r.left + r.width / 2 + 'px'); st.setProperty('--tcy', cy + 'px'); st.setProperty('--tsz', Math.min(r.height * 1.3, r.width * 1.2) + 'px');
  }
  addEventListener('resize', fit); fit();

  const bump = (x, c, w) => Math.exp(-(((x - c) / w) ** 2));
  let hr = 70, beatT = 0; const clock = new THREE.Clock();
  renderer.setAnimationLoop(() => {
    const dt = Math.min(0.1, clock.getDelta()); beatT += dt * hr / 60;
    if (goal) { cam.position.lerp(goal, 0.06); if (cam.position.distanceTo(goal) < 0.02) goal = null; }
    if (tgt) { controls.target.lerp(tgt, 0.06); if (controls.target.distanceTo(tgt) < 0.01) tgt = null; }
    controls.update();
    const ph = beatT % 1, c = reduced ? 0 : -0.024 * bump(ph, 0.12, 0.09) + 0.012 * bump(ph, 0.55, 0.14); // systolic squeeze (2.4% max), then diastolic compliance
    group.scale.set(BASE * (1 + 0.7 * c), BASE * (1 + c), BASE * (1 + 0.7 * c));
    for (const n in vessels) {
      const v = vessels[n]; v.u.uTime.value = beatT * 2; v.u.uColor.value.lerp(v.tc, 0.1); v.node.material.color.copy(v.u.uColor.value);
      if (Math.abs(v.sT - v.s) > 0.002) { v.s += (v.sT - v.s) * 0.1; pinch(v, v.s); }
    }
    group.updateMatrixWorld();
    for (const n in vessels) {
      const p = group.localToWorld(vessels[n].curve.getPointAt(0.4)).project(cam);
      vessels[n].label.style.transform = `translate(${(p.x * 0.5 + 0.5) * W() - 24}px,${(-p.y * 0.5 + 0.5) * H() - 16}px)`;
    }
    renderer.render(scene, cam);
  });

  const VIEWS = { anterior: [0, 0.2, 9], posterior: [0, 0.2, -9], vessels: [-4.5, 1.6, 7.6] };
  return {
    update(out) {
      hr = cur.measurements.PR || 70;
      for (const n in vessels) {
        const v = vessels[n], l = lvl(out[n]);
        v.tc.setHex(LEVEL_COLOR[l]); v.u.uGain.value = selected ? (selected === n ? 1.5 : 0.6) : 1;
        v.sT = Math.min(0.85, (out[n] / 100) * 0.85); v.label.dataset.level = l; v.label.textContent = `${n} ${f1(out[n])}%`;
      }
    },
    view(k) { goal = V(VIEWS[k]); tgt = new THREE.Vector3(0, 0.2, 0); controls.autoRotate = false; },
    focus(n) { // glide to the lesion node, viewed from outside the heart
      const p = group.localToWorld(vessels[n].curve.getPointAt(0.45)), dir = p.clone().sub(new THREE.Vector3(0, 0.2, 0)).normalize();
      tgt = p.clone(); goal = p.clone().add(dir.multiplyScalar(4)); controls.autoRotate = false;
    },
    layer(k, on) { k === 'shell' ? (shell.visible = on) : Object.values(vessels).forEach(v => (v.u.uFlow.value = on ? 1 : 0)); },
  };
}

/* ---------- Pointer physics: instant press + rAF-throttled tilt ---------- */
function bindMotion() {
  document.addEventListener('pointerdown', e => { const el = e.target.closest('.pressable'); if (el) { el.classList.add('is-pressed'); const end = () => { el.classList.remove('is-pressed'); removeEventListener('pointerup', end); removeEventListener('pointercancel', end); }; addEventListener('pointerup', end); addEventListener('pointercancel', end); } });
  if (!fine || reduced) return;
  $$('.tilt').forEach(el => {
    let raf = 0, ev;
    el.addEventListener('pointermove', e => { ev = e; if (raf) return; raf = requestAnimationFrame(() => { raf = 0; const r = el.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width, py = (ev.clientY - r.top) / r.height; el.classList.add('is-tilting'); el.style.setProperty('--ry', ((px - 0.5) * 12) + 'deg'); el.style.setProperty('--rx', ((0.5 - py) * 12) + 'deg'); el.style.setProperty('--mx', px * 100 + '%'); el.style.setProperty('--my', py * 100 + '%'); }); });
    el.addEventListener('pointerleave', () => { cancelAnimationFrame(raf); raf = 0; el.classList.remove('is-tilting'); ['--rx', '--ry'].forEach(p => el.style.removeProperty(p)); el.style.setProperty('--sheen', 0); setTimeout(() => el.style.removeProperty('--sheen'), 300); });
  });
}

/* ---------- Safety disclaimer dialog ---------- */
function initDisclaimer() {
  const dlg = document.createElement('dialog'); dlg.className = 'modal';
  dlg.setAttribute('aria-labelledby', 'modal-title'); dlg.setAttribute('aria-describedby', 'modal-text');
  dlg.innerHTML = `<div class="modal__body glass"><h2 id="modal-title">Safety disclaimer</h2><div id="modal-text">
    <p>CardioAI is clinical decision support for qualified cardiologists. It does not diagnose, and it must not replace clinical judgment or invasive angiography.</p>
    <p>Predictions come from a research model evaluated on a limited cohort. Treat them as one input alongside the patient's history, imaging and catheterization findings.</p>
    <p>What-If values come from a simplified simulation. They are estimates, not measurements, and not outputs of the trained model.</p></div>
    <form method="dialog"><button class="chip pressable modal__close" autofocus>I understand</button></form></div>`;
  document.body.appendChild(dlg);
  const pill = $('#disclaimer'); pill.setAttribute('aria-haspopup', 'dialog');
  pill.addEventListener('click', () => { dlg.showModal(); pill.setAttribute('aria-expanded', 'true'); });
  dlg.addEventListener('close', () => pill.setAttribute('aria-expanded', 'false'));
  dlg.addEventListener('click', e => { if (e.target === dlg) dlg.close(); }); // click on the dimmed backdrop
}

/* ---------- Boot ---------- */
async function boot() {
  bindMotion(); initECG();
  try { heart = initHeart(); } catch (e) { console.warn('WebGL unavailable', e); }
  try { patients = await (await fetch('./data/patients_data.json')).json(); }
  catch { $('#patient-id').textContent = 'Data not found'; $('#panel-patient .muted').textContent = 'Serve this folder over http and keep patients_data.json inside ./data.'; return; }
  $$('#patient-tabs .seg__btn').forEach((b, i) => { if (!patients[i]) b.disabled = true; b.addEventListener('pointerdown', () => selectPatient(i)); });
  $$('.field input').forEach(el => el.addEventListener('input', () => { const o = $(`output[for="${el.id}"]`); if (o) o.textContent = el.value; render(); }));
  $('#whatif-reset').addEventListener('pointerdown', () => { for (const k in SIM) writeSlider(SIM[k][0], baseline[k]); render(); });
  $$('.vcard').forEach(c => {
    const n = c.dataset.vessel;
    c.addEventListener('pointerdown', () => { selected = selected === n ? null : n; render(); });
    $('.vcard__focus', c).addEventListener('pointerdown', e => { e.stopPropagation(); selected = n; render(); heart && heart.focus(n); });
  });
  $$('[data-view]').forEach(b => b.addEventListener('pointerdown', () => heart && heart.view(b.dataset.view)));
  $$('[data-layer]').forEach(b => b.addEventListener('pointerdown', () => { const on = b.getAttribute('aria-pressed') !== 'true'; b.setAttribute('aria-pressed', on); heart && heart.layer(b.dataset.layer, on); }));
  initDisclaimer();
  document.addEventListener('click', e => { // Enter/Space (click with detail 0) reuses the pointerdown handlers
    if (e.detail !== 0) return; const el = e.target.closest('button');
    if (el && !el.closest('dialog')) el.dispatchEvent(new PointerEvent('pointerdown'));
  });
  selectPatient(0);
}
boot();
