// Liquid Glass for plain HTML (no framework, no build step).
//
// Technique adapted from liquid-glass-react by rdev (MIT,
// https://github.com/rdev/liquid-glass-react) and shuding/liquid-glass: the
// backdrop of an element is blurred/saturated with backdrop-filter, then bent
// by an SVG feDisplacementMap whose map is generated per element size. Three
// displacement passes at slightly different strengths are recombined per
// colour channel, which gives the faint chromatic fringe at the rim.
//
// The displacement map here is our own: a rounded-rectangle signed distance
// field in pixels, so refraction stays confined to a bezel of fixed width
// whatever the element's aspect ratio.
//
// Support: the SVG-filter displacement only renders in Chromium browsers.
// Elsewhere (and when the user prefers reduced transparency) elements keep a
// plain blur. Everything is purely visual and aria-hidden.

const SVG_NS = 'http://www.w3.org/2000/svg';
let defs = null, uid = 0;

export function liquidGlassSupported() {
  const brands = navigator.userAgentData && navigator.userAgentData.brands;
  const chromium = brands ? brands.some(b => /Chromium|Google Chrome|Microsoft Edge/.test(b.brand)) : /Chrome\//.test(navigator.userAgent) && !/Edg\/|OPR\//.test(navigator.userAgent) || /Edg\//.test(navigator.userAgent);
  const lessTransparency = matchMedia('(prefers-reduced-transparency: reduce)').matches;
  return chromium && !/Firefox\//.test(navigator.userAgent) && !lessTransparency;
}

function ensureDefs() {
  if (defs) return defs;
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('aria-hidden', 'true'); svg.setAttribute('focusable', 'false');
  Object.assign(svg.style, { position: 'absolute', width: '0', height: '0', overflow: 'hidden', pointerEvents: 'none' });
  defs = document.createElementNS(SVG_NS, 'defs'); svg.appendChild(defs); document.body.appendChild(svg);
  return defs;
}

// Rounded-rect SDF (negative inside), p relative to the centre, b = half-size.
function sdRoundRect(px, py, bx, by, r) {
  const qx = Math.abs(px) - bx + r, qy = Math.abs(py) - by + r;
  return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - r;
}

// Displacement map: inside a bezel near the edge, sample the backdrop from
// further inward (lens edge). R = x offset, G and B = y offset, 0.5 = none.
function makeMap(w, h, radius, bezel) {
  const S = 0.5; // half resolution is plenty; feImage scales it back up
  const W = Math.max(2, Math.round(w * S)), H = Math.max(2, Math.round(h * S));
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const ctx = c.getContext('2d'), img = ctx.createImageData(W, H), d = img.data;
  const bx = w / 2, by = h / 2, r = Math.min(radius, bx, by), e = 1;
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const px = (i + 0.5) / S - bx, py = (j + 0.5) / S - by;
      const sd = sdRoundRect(px, py, bx, by, r);
      let dx = 0, dy = 0;
      if (sd < 0 && -sd < bezel) {
        // inward normal from the SDF gradient
        const gx = sdRoundRect(px + e, py, bx, by, r) - sdRoundRect(px - e, py, bx, by, r);
        const gy = sdRoundRect(px, py + e, bx, by, r) - sdRoundRect(px, py - e, bx, by, r);
        const gl = Math.hypot(gx, gy) || 1, t = 1 + sd / bezel; // 1 at the edge, 0 at the bezel's inner side
        const m = t * t * (3 - 2 * t);                         // smoothstep falloff
        dx = -(gx / gl) * m; dy = -(gy / gl) * m; // sample inward: the rim magnifies what is just inside it
      }
      const k = (j * W + i) * 4;
      d[k] = 128 + dx * 127; d[k + 1] = 128 + dy * 127; d[k + 2] = 128 + dy * 127; d[k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c.toDataURL();
}

function buildFilter(id, w, h, radius, { scale, bezel, aberration }) {
  const old = document.getElementById(id); if (old) old.remove();
  const f = document.createElementNS(SVG_NS, 'filter');
  f.setAttribute('id', id); f.setAttribute('x', '0'); f.setAttribute('y', '0');
  f.setAttribute('width', w); f.setAttribute('height', h);
  f.setAttribute('filterUnits', 'userSpaceOnUse'); f.setAttribute('color-interpolation-filters', 'sRGB');
  const map = makeMap(w, h, radius, bezel);
  const chan = (name, keep) => `<feColorMatrix in="${name}_D" type="matrix" values="${keep[0]} 0 0 0 0  0 ${keep[1]} 0 0 0  0 0 ${keep[2]} 0 0  0 0 0 1 0" result="${name}"/>`;
  f.innerHTML = `
    <feImage href="${map}" x="0" y="0" width="${w}" height="${h}" preserveAspectRatio="none" result="MAP"/>
    <feDisplacementMap in="SourceGraphic" in2="MAP" scale="${scale}" xChannelSelector="R" yChannelSelector="G" result="R_D"/>
    <feDisplacementMap in="SourceGraphic" in2="MAP" scale="${scale * (1 + aberration * 0.05)}" xChannelSelector="R" yChannelSelector="G" result="G_D"/>
    <feDisplacementMap in="SourceGraphic" in2="MAP" scale="${scale * (1 + aberration * 0.1)}" xChannelSelector="R" yChannelSelector="G" result="B_D"/>
    ${chan('R', [1, 0, 0])}${chan('G', [0, 1, 0])}${chan('B', [0, 0, 1])}
    <feBlend in="G" in2="B" mode="screen" result="GB"/>
    <feBlend in="R" in2="GB" mode="screen"/>`;
  ensureDefs().appendChild(f);
}

/**
 * Turn elements into Liquid Glass.
 * @param {string|Element[]} targets  selector or elements
 * @param {{scale?:number, bezel?:number, aberration?:number, blur?:number, saturation?:number}} opts
 */
export function liquidGlass(targets, opts = {}) {
  const o = { scale: 56, bezel: 18, aberration: 2, blur: 1.5, saturation: 175, ...opts };
  const els = typeof targets === 'string' ? [...document.querySelectorAll(targets)] : targets;
  const on = liquidGlassSupported();
  document.documentElement.classList.toggle('lg-enabled', on);
  for (const el of els) {
    if (el.querySelector(':scope > .lg-warp')) continue;
    const warp = document.createElement('span'); warp.className = 'lg-warp'; warp.setAttribute('aria-hidden', 'true');
    const shine = document.createElement('span'); shine.className = 'lg-shine'; shine.setAttribute('aria-hidden', 'true');
    el.prepend(warp); el.append(shine); el.classList.add('lg');
    warp.style.backdropFilter = warp.style.webkitBackdropFilter = `blur(${o.blur}px) saturate(${o.saturation}%)`;
    el.addEventListener('pointermove', e => {
      const r = el.getBoundingClientRect();
      el.style.setProperty('--lg-x', `${((e.clientX - r.left) / r.width) * 100}%`);
      el.style.setProperty('--lg-y', `${((e.clientY - r.top) / r.height) * 100}%`);
    });
    el.addEventListener('pointerleave', () => { el.style.removeProperty('--lg-x'); el.style.removeProperty('--lg-y'); });
    if (!on) continue;
    const id = `lg-${++uid}`;
    let last = '', timer = 0;
    const rebuild = () => {
      const w = Math.round(el.offsetWidth), h = Math.round(el.offsetHeight);
      if (!w || !h) return;
      const radius = parseFloat(getComputedStyle(el).borderTopLeftRadius) || 0;
      const key = `${w}x${h}x${radius}`; if (key === last) return; last = key;
      buildFilter(id, w, h, radius, { ...o, bezel: Math.min(o.bezel, w / 2, h / 2) });
      warp.style.filter = `url(#${id})`;
    };
    new ResizeObserver(() => { clearTimeout(timer); timer = setTimeout(rebuild, 60); }).observe(el);
    rebuild();
  }
}
