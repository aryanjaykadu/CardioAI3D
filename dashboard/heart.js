// Procedural anatomical heart for CardioAI (schematic, not patient imaging).
// The myocardium is a signed-distance field (ventricles, atria, outflow tract, appendages)
// meshed by casting rays from the centre; grooves are where the parts blend, and are
// coloured as epicardial fat. Great vessels are tubes; coronaries are projected onto
// the surface so they run in the grooves.
import * as THREE from './lib/three.module.js';
import { OrbitControls } from './lib/OrbitControls.js';

const v3 = (x, y, z) => new THREE.Vector3(x, y, z);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

/* ------------------------------------------------------------------ SDF primitives */
function sdRoundCone(p, a, b, r1, r2) { // Inigo Quilez, exact
  const ba = b.clone().sub(a), l2 = ba.lengthSq(), rr = r1 - r2, a2 = l2 - rr * rr, il2 = 1 / l2;
  const pa = p.clone().sub(a), y = pa.dot(ba), z = y - l2;
  const xv = pa.clone().multiplyScalar(l2).sub(ba.clone().multiplyScalar(y)), x2 = xv.lengthSq();
  const y2 = y * y * l2, z2 = z * z * l2, k = Math.sign(rr) * rr * rr * x2;
  if (Math.sign(z) * a2 * z2 > k) return Math.sqrt(x2 + z2) * il2 - r2;
  if (Math.sign(y) * a2 * y2 < k) return Math.sqrt(x2 + y2) * il2 - r1;
  return (Math.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}
function sdEllipsoid(p, c, r) {
  const q = p.clone().sub(c), k0 = v3(q.x / r.x, q.y / r.y, q.z / r.z).length(), k1 = v3(q.x / (r.x * r.x), q.y / (r.y * r.y), q.z / (r.z * r.z)).length();
  return k1 === 0 ? -Math.min(r.x, r.y, r.z) : k0 * (k0 - 1) / k1;
}
const smin = (a, b, k) => { const h = clamp(0.5 + 0.5 * (b - a) / k, 0, 1); return b * (1 - h) + a * h - k * h * (1 - h); };

// Coordinates: x = viewer's right (patient's left), y = up, z = toward viewer (anterior).
const PARTS = [
  { f: p => sdRoundCone(p, v3(0.08, 0.3, -0.22), v3(0.66, -1.28, 0.18), 0.8, 0.15), k: 0.10 },               // left ventricle
  { f: p => sdRoundCone(p, v3(-0.38, 0.22, 0.3), v3(0.2, -0.98, 0.46), 0.56, 0.12), k: 0.22 },                // right ventricle (anterior)
  { f: p => sdRoundCone(p, v3(-0.05, 0.3, 0.5), v3(0.12, 0.92, 0.42), 0.36, 0.25), k: 0.18 },                 // RV outflow tract (conus)
  { f: p => sdEllipsoid(p, v3(-0.88, 0.58, 0.02), v3(0.46, 0.56, 0.5)), k: 0.14 },                            // right atrium
  { f: p => sdEllipsoid(p, v3(0.22, 0.82, -0.6), v3(0.62, 0.38, 0.42)), k: 0.14 },                            // left atrium (posterior)
  { f: p => sdEllipsoid(p, v3(0.72, 0.86, 0.12), v3(0.24, 0.13, 0.2)), k: 0.08 },                             // left atrial appendage
  { f: p => sdEllipsoid(p, v3(-0.58, 0.98, 0.42), v3(0.26, 0.16, 0.22)), k: 0.08 },                           // right atrial appendage
];
function sdf(p) { let d = PARTS[0].f(p); for (let i = 1; i < PARTS.length; i++) d = smin(d, PARTS[i].f(p), PARTS[i].k); return d; }
function grooveDepth(p) { let m = Infinity; for (const P of PARTS) m = Math.min(m, P.f(p)); return m - sdf(p); } // >0 where parts blend
const CENTER = v3(0.0, 0.0, 0.0);

function surfaceAlong(dir) { // outermost zero crossing along a ray from CENTER
  let t = 3.2, prev = sdf(CENTER.clone().addScaledVector(dir, t));
  for (let s = 3.2; s > 0; s -= 0.05) {
    const d = sdf(CENTER.clone().addScaledVector(dir, s));
    if (d < 0) { let lo = s, hi = s + 0.05; for (let i = 0; i < 14; i++) { const m = (lo + hi) / 2; sdf(CENTER.clone().addScaledVector(dir, m)) < 0 ? lo = m : hi = m; } return (lo + hi) / 2; }
    prev = d; t = s;
  }
  return 0.2;
}
const projectToSurface = (p, lift = 0) => { const d = p.clone().sub(CENTER).normalize(); return CENTER.clone().addScaledVector(d, surfaceAlong(d) + lift); };
function sdfNormal(p) { const e = 0.004; return v3(sdf(v3(p.x + e, p.y, p.z)) - sdf(v3(p.x - e, p.y, p.z)), sdf(v3(p.x, p.y + e, p.z)) - sdf(v3(p.x, p.y - e, p.z)), sdf(v3(p.x, p.y, p.z + e)) - sdf(v3(p.x, p.y, p.z - e))).normalize(); }
const hash = (x, y, z) => { const s = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453; return s - Math.floor(s); };

function buildMyocardium() {
  const g = new THREE.SphereGeometry(1, 220, 160), pos = g.attributes.position, n = pos.count;
  const nrm = new Float32Array(n * 3), col = new Float32Array(n * 3);
  const muscle = new THREE.Color('#9c2234'), muscleDark = new THREE.Color('#5e1220'), fat = new THREE.Color('#d9b45e'), tmp = new THREE.Color();
  const dir = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    dir.set(pos.getX(i), pos.getY(i), pos.getZ(i)).normalize();
    const p = CENTER.clone().addScaledVector(dir, surfaceAlong(dir));
    pos.setXYZ(i, p.x, p.y, p.z);
    const N = sdfNormal(p); nrm.set([N.x, N.y, N.z], i * 3);
    const gd = grooveDepth(p), fatAmt = clamp((gd - 0.03) / 0.07, 0, 1) * 0.7;
    const mott = 0.82 + 0.18 * hash(Math.round(p.x * 40), Math.round(p.y * 40), Math.round(p.z * 40));
    tmp.copy(muscleDark).lerp(muscle, clamp(0.55 + 0.45 * N.y + 0.2 * N.z, 0, 1)).multiplyScalar(mott).lerp(fat, fatAmt);
    col.set([tmp.r, tmp.g, tmp.b], i * 3);
  }
  g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return g;
}

/* ------------------------------------------------------------------ tubes */
function taperTube(points, r0, r1, radial = 20, segs = 80) {
  const curve = new THREE.CatmullRomCurve3(points), geo = new THREE.TubeGeometry(curve, segs, 1, radial, false);
  const pos = geo.attributes.position, per = radial + 1;
  for (let i = 0; i <= segs; i++) {
    const t = i / segs, c = curve.getPointAt(t), r = r0 + (r1 - r0) * t;
    for (let j = 0; j < per; j++) { const k = i * per + j; const q = v3(pos.getX(k), pos.getY(k), pos.getZ(k)).sub(c).multiplyScalar(r).add(c); pos.setXYZ(k, q.x, q.y, q.z); }
  }
  geo.computeVertexNormals();
  return { geo, curve };
}
const P = a => a.map(p => v3(...p));
const GREAT = {
  aorta: { pts: P([[-0.12, 0.55, 0.05], [-0.22, 1.2, 0.12], [-0.12, 1.72, 0.0], [0.3, 1.9, -0.3], [0.6, 1.6, -0.62], [0.62, 0.9, -0.82], [0.6, 0.2, -0.85]]), r: [0.23, 0.18], color: '#c8434f' },
  brachio: { pts: P([[-0.15, 1.75, -0.02], [-0.3, 2.15, 0.0], [-0.38, 2.4, 0.02]]), r: [0.085, 0.07], color: '#c8434f' },
  carotid: { pts: P([[0.08, 1.88, -0.15], [0.08, 2.25, -0.15], [0.06, 2.45, -0.14]]), r: [0.065, 0.055], color: '#c8434f' },
  subclav: { pts: P([[0.3, 1.86, -0.32], [0.48, 2.2, -0.35], [0.58, 2.4, -0.36]]), r: [0.075, 0.06], color: '#c8434f' },
  pulm: { pts: P([[0.1, 0.85, 0.42], [0.22, 1.25, 0.32], [0.3, 1.45, 0.1]]), r: [0.25, 0.22], color: '#3f63c4' },
  lpa: { pts: P([[0.3, 1.45, 0.1], [0.68, 1.5, -0.15], [1.0, 1.42, -0.4]]), r: [0.15, 0.12], color: '#3f63c4' },
  rpa: { pts: P([[0.25, 1.45, 0.05], [-0.15, 1.45, -0.3], [-0.6, 1.38, -0.5]]), r: [0.15, 0.12], color: '#3f63c4' },
  svc: { pts: P([[-0.9, 0.85, 0.0], [-0.86, 1.5, 0.02], [-0.82, 2.1, 0.02]]), r: [0.17, 0.16], color: '#3550a8' },
  ivc: { pts: P([[-0.82, 0.25, -0.12], [-0.78, -0.2, -0.2], [-0.74, -0.55, -0.22]]), r: [0.18, 0.17], color: '#3550a8' },
  pv1: { pts: P([[0.65, 0.95, -0.75], [1.0, 1.05, -0.85]]), r: [0.08, 0.075], color: '#b3424e' },
  pv2: { pts: P([[0.62, 0.68, -0.78], [0.98, 0.6, -0.9]]), r: [0.08, 0.075], color: '#b3424e' },
  pv3: { pts: P([[-0.15, 0.95, -0.78], [-0.45, 1.02, -0.92]]), r: [0.08, 0.075], color: '#b3424e' },
};
// Coronary courses (projected onto the surface).
const CORONARY = {
  LAD: { main: P([[-0.05, 0.62, 0.2], [0.12, 0.6, 0.45], [0.1, 0.35, 0.8], [0.2, -0.1, 0.85], [0.35, -0.55, 0.8], [0.52, -0.95, 0.6], [0.72, -1.3, 0.35]]),
         br: [P([[0.15, 0.1, 0.85], [0.5, -0.2, 0.75], [0.8, -0.45, 0.5]]), P([[0.25, -0.35, 0.85], [0.6, -0.65, 0.6]])] }, // diagonals
  LCX: { main: P([[0.12, 0.6, 0.45], [0.5, 0.55, 0.4], [0.85, 0.35, 0.05], [0.9, 0.2, -0.4], [0.55, 0.05, -0.85], [0.2, -0.05, -0.95]]),
         br: [P([[0.92, 0.3, -0.1], [1.05, -0.3, -0.2], [1.0, -0.75, -0.1]]), P([[0.75, 0.15, -0.65], [0.85, -0.45, -0.6]])] }, // obtuse marginals
  RCA: { main: P([[-0.35, 0.6, 0.25], [-0.6, 0.48, 0.55], [-0.95, 0.15, 0.5], [-1.0, -0.3, 0.2], [-0.75, -0.6, -0.35], [-0.2, -0.55, -0.8], [0.15, -0.45, -0.9]]),
         br: [P([[-0.85, -0.2, 0.55], [-0.55, -0.6, 0.6], [-0.2, -0.85, 0.55]]), P([[0.0, -0.5, -0.88], [0.3, -0.95, -0.7], [0.55, -1.2, -0.4]])] }, // acute marginal, PDA
};
function surfaceCurve(pts, lift, n = 70) {
  const raw = new THREE.CatmullRomCurve3(pts.map(p => projectToSurface(p, lift)));
  return new THREE.CatmullRomCurve3(Array.from({ length: n + 1 }, (_, i) => projectToSurface(raw.getPointAt(i / n), lift)));
}

const VSH = {
  vertexShader: 'varying vec2 vUv; varying vec3 vN; void main(){ vUv=uv; vN=normalize(normalMatrix*normal); gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }',
  fragmentShader: 'varying vec2 vUv; varying vec3 vN; uniform vec3 uColor; uniform float uTime,uGain,uFlow; void main(){ float p=sin(vUv.x*22.-uTime*4.)*.5+.5; float rim=pow(1.-abs(vN.z),2.); vec3 c=uColor*(.6+.6*uFlow*p*p+.35*rim)*uGain; gl_FragColor=vec4(pow(c,vec3(1./2.2)),1.); }',
};

/* ------------------------------------------------------------------ scene */
export function initHeart({ host, labelsHost, reduced, getHR }) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
  renderer.outputColorSpace = THREE.SRGBColorSpace; host.appendChild(renderer.domElement);
  const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(34, 1, 0.1, 60);

  // soft studio environment for wet-tissue reflections
  const ec = document.createElement('canvas'); ec.width = 512; ec.height = 256; const cx = ec.getContext('2d');
  const sky = cx.createLinearGradient(0, 0, 0, 256); sky.addColorStop(0, '#3a3a4a'); sky.addColorStop(0.55, '#121218'); sky.addColorStop(1, '#050507');
  cx.fillStyle = sky; cx.fillRect(0, 0, 512, 256);
  [[60, 30, 140, 40, '#ffffff'], [300, 24, 150, 30, '#ffe9e0'], [420, 90, 40, 80, '#9a8cff']].forEach(([x, y, w, h, c]) => { cx.filter = 'blur(10px)'; cx.fillStyle = c; cx.fillRect(x, y, w, h); });
  const etex = new THREE.CanvasTexture(ec); etex.mapping = THREE.EquirectangularReflectionMapping; etex.colorSpace = THREE.SRGBColorSpace;
  scene.environment = new THREE.PMREMGenerator(renderer).fromEquirectangular(etex).texture; etex.dispose();
  scene.add(new THREE.HemisphereLight(0xffe8e8, 0x1a0a14, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(-3, 4, 5); scene.add(key);
  const rim = new THREE.DirectionalLight(0x8f7bff, 1.6); rim.position.set(4, 1, -4); scene.add(rim);

  const group = new THREE.Group(); group.rotation.z = 0.08; scene.add(group);
  const myoMat = new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.42, metalness: 0, clearcoat: 0.55, clearcoatRoughness: 0.3, sheen: 0.5, sheenColor: new THREE.Color('#ff9a9a'), sheenRoughness: 0.6, transparent: false });
  const myo = new THREE.Mesh(buildMyocardium(), myoMat); group.add(myo);
  const vesselMeshes = [];
  for (const g of Object.values(GREAT)) {
    const m = new THREE.Mesh(taperTube(g.pts, g.r[0], g.r[1], 28, 64).geo, new THREE.MeshPhysicalMaterial({ color: g.color, roughness: 0.35, clearcoat: 0.6, clearcoatRoughness: 0.25 }));
    group.add(m); vesselMeshes.push(m);
    const capM = new THREE.Mesh(new THREE.CircleGeometry(g.r[1], 24), new THREE.MeshStandardMaterial({ color: '#2a0a10', side: THREE.DoubleSide }));
    const end = g.pts[g.pts.length - 1], prev = g.pts[g.pts.length - 2]; capM.position.copy(end); capM.lookAt(end.clone().add(end.clone().sub(prev))); group.add(capM); vesselMeshes.push(capM);
  }

  const vessels = {};
  for (const [name, def] of Object.entries(CORONARY)) {
    const u = { uColor: { value: new THREE.Color(0x30d158) }, uTime: { value: 0 }, uGain: { value: 1 }, uFlow: { value: 1 } };
    const mat = new THREE.ShaderMaterial({ ...VSH, uniforms: u });
    const curve = surfaceCurve(def.main, 0.012);
    const seg = 120, rad = 12, geo = new THREE.TubeGeometry(curve, seg, 0.042, rad, false);
    group.add(new THREE.Mesh(geo, mat));
    def.br.forEach(b => group.add(new THREE.Mesh(new THREE.TubeGeometry(surfaceCurve(b, 0.008, 40), 50, 0.024, 8, false), mat)));
    const base = geo.attributes.position.array.slice(), centers = Array.from({ length: seg + 1 }, (_, i) => curve.getPointAt(i / seg));
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.1, 0.018, 10, 32), new THREE.MeshBasicMaterial({ color: 0x30d158, transparent: true, opacity: 0.95 }));
    const at = curve.getPointAt(0.42), tan = curve.getTangentAt(0.42); ring.position.copy(at); ring.lookAt(at.clone().add(tan)); group.add(ring);
    const label = document.createElement('div'); label.className = 'vlabel'; labelsHost.appendChild(label);
    vessels[name] = { curve, geo, base, centers, seg, rad, u, ring, label, tc: new THREE.Color(0x30d158), s: 0, sT: 0, show: true };
  }
  function pinch(v, s) { // focal narrowing at 42% of the main vessel
    const a = v.geo.attributes.position, per = v.rad + 1;
    for (let i = 0; i < a.count; i++) {
      const k = Math.floor(i / per), t = k / v.seg, c = v.centers[k];
      const f = (1 - 0.35 * t) * (1 - s * Math.exp(-(((t - 0.42) / 0.05) ** 2)));
      a.setXYZ(i, c.x + (v.base[i * 3] - c.x) * f, c.y + (v.base[i * 3 + 1] - c.y) * f, c.z + (v.base[i * 3 + 2] - c.z) * f);
    }
    a.needsUpdate = true;
  }

  const controls = new OrbitControls(cam, renderer.domElement);
  Object.assign(controls, { enableDamping: true, dampingFactor: 0.06, enablePan: false, minDistance: 3.2, maxDistance: 12, autoRotate: !reduced, autoRotateSpeed: 0.7 });
  const TARGET = v3(0, 0.35, 0); controls.target.copy(TARGET); cam.position.set(0, 0.6, 8.2);
  let goal = null, tgt = null; controls.addEventListener('start', () => { goal = tgt = null; });

  let W = 1, H = 1;
  const fit = () => { W = host.clientWidth || 1; H = host.clientHeight || 1; renderer.setSize(W, H, false); cam.aspect = W / H; cam.updateProjectionMatrix(); };
  new ResizeObserver(fit).observe(host); fit();

  const bump = (x, c, w) => Math.exp(-(((x - c) / w) ** 2));
  let beat = 0, selected = null; const clock = new THREE.Clock(), BASE = 1;
  renderer.setAnimationLoop(() => {
    const dt = Math.min(0.1, clock.getDelta());
    if (host.offsetParent === null || document.hidden) return;
    beat += dt * (getHR() || 70) / 60;
    if (tgt) { controls.target.lerp(tgt, 0.08); if (controls.target.distanceTo(tgt) < 0.01) tgt = null; }
    if (goal) {
      const c0 = cam.position.clone().sub(controls.target), c1 = goal.clone().sub(controls.target);
      const q = new THREE.Quaternion().setFromUnitVectors(c0.clone().normalize(), c1.clone().normalize());
      const step = new THREE.Quaternion().slerp(q, 0.08), len = c0.length() + (c1.length() - c0.length()) * 0.08;
      cam.position.copy(controls.target).add(c0.applyQuaternion(step).setLength(len));
      if (cam.position.distanceTo(goal) < 0.03) goal = null;
    }
    controls.update();
    const ph = beat % 1, c = reduced ? 0 : -0.028 * bump(ph, 0.12, 0.08) + 0.012 * bump(ph, 0.5, 0.14);
    group.scale.set(BASE * (1 + 0.8 * c), BASE * (1 + c), BASE * (1 + 0.8 * c));
    for (const n in vessels) {
      const v = vessels[n]; v.u.uTime.value = beat * 2; v.u.uColor.value.lerp(v.tc, 1 - Math.exp(-dt * 10)); v.ring.material.color.copy(v.u.uColor.value);
      v.ring.scale.setScalar(1 + (reduced ? 0 : 0.15 * Math.sin(beat * Math.PI * 2)) * v.s);
      if (Math.abs(v.sT - v.s) > 0.002) { v.s += (v.sT - v.s) * 0.12; pinch(v, v.s); }
    }
    group.updateMatrixWorld();
    for (const n in vessels) {
      const v = vessels[n], wp = group.localToWorld(v.curve.getPointAt(0.42).clone()), p = wp.clone().project(cam);
      const facing = wp.clone().sub(group.position).normalize().dot(cam.position.clone().sub(wp).normalize()) > -0.15;
      v.label.style.transform = `translate(${(p.x * 0.5 + 0.5) * W + 10}px,${(-p.y * 0.5 + 0.5) * H - 12}px)`;
      v.label.style.opacity = v.show ? (facing ? 1 : 0.35) : 0;
    }
    renderer.render(scene, cam);
  });

  const LEVEL = { ok: 0x30d158, mid: 0xffb000, high: 0xff1f3d, none: 0x9a9aa6 };
  const VIEWS = { anterior: v3(0, 0.6, 8.2), posterior: v3(0, 0.6, -8.2), vessels: v3(-5.2, 2.4, 6.2) };
  return {
    update(P, lvl, fmt) {
      for (const n in vessels) {
        const v = vessels[n], p = P[n], has = p != null, l = has ? lvl(p) : 'none';
        v.tc.setHex(LEVEL[l]); v.u.uGain.value = selected ? (selected === n ? 1.5 : 0.45) : 1;
        v.sT = has ? Math.min(0.8, p * 0.8) : 0; v.show = has;
        v.label.dataset.level = l; v.label.textContent = has ? `${n} ${fmt(p)}` : '';
      }
    },
    select(n) { selected = n; },
    view(k) { goal = VIEWS[k].clone(); tgt = TARGET.clone(); },
    focus(n) {
      const p = group.localToWorld(vessels[n].curve.getPointAt(0.42).clone()), d = p.clone().sub(TARGET).normalize();
      tgt = p.clone(); goal = p.clone().add(d.multiplyScalar(4.2)); controls.autoRotate = false; return true;
    },
    layer(k, on) {
      if (k === 'shell') { myoMat.transparent = !on; myoMat.opacity = on ? 1 : 0.28; myoMat.depthWrite = on; myoMat.needsUpdate = true; vesselMeshes.forEach(m => { m.material.transparent = !on; m.material.opacity = on ? 1 : 0.35; }); }
      else controls.autoRotate = on;
    },
  };
}
