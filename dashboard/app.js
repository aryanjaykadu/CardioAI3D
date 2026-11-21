// CardioAI dashboard. Runs the exported logistic-regression models in the browser.
//   data/model.json    full model (55 inputs, Tehran), 4 targets, bootstrap coefficients
//   data/bedside.json  bedside model (7 inputs, 5 hospitals), leave-one-hospital-out predictions
//   data/patients.json out-of-fold predictions for the 303 Tehran patients
//   data/evidence.json evaluation results
import { initHeart } from './heart.js';
import { liquidGlass } from './liquid-glass.js';

const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const pct = p => p > 0.995 ? '>99%' : p < 0.005 ? '<1%' : (p * 100).toFixed(p < 0.1 ? 1 : 0) + '%';
const lvl = p => (p >= 0.65 ? 'high' : p >= 0.4 ? 'mid' : 'ok');
const sigm = z => 1 / (1 + Math.exp(-z));
const logit = p => { const q = Math.min(1 - 1e-4, Math.max(1e-4, p)); return Math.log(q / (1 - q)); };
const VESSELS = ['LAD', 'LCX', 'RCA'];
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const quantile = (arr, q) => { const a = [...arr].sort((x, y) => x - y), i = (a.length - 1) * q, lo = Math.floor(i); return a[lo] + (a[Math.ceil(i)] - a[lo]) * (i - lo); };

/* ================================================================ ESC 2019 pre-test probability (Table 5) */
const ESC_T = { // [age band][male, female] in %
  typical: [[3, 5], [22, 10], [32, 13], [44, 16], [52, 27]], atypical: [[4, 3], [10, 6], [17, 6], [26, 11], [34, 19]],
  nonanginal: [[1, 1], [3, 2], [11, 3], [22, 6], [24, 10]], dyspnoea: [[0, 3], [12, 3], [20, 9], [27, 14], [32, 12]],
};
function escPTP(age, male, cp, dyspnoea = 0) {
  const sym = cp === 1 ? 'typical' : cp === 2 ? 'atypical' : cp === 3 ? 'nonanginal' : dyspnoea ? 'dyspnoea' : 'nonanginal';
  const band = age < 40 ? 0 : age < 50 ? 1 : age < 60 ? 2 : age < 70 ? 3 : 4;
  return ESC_T[sym][band][male ? 0 : 1] / 100;
}

/* ================================================================ state */
let FULL, BED, DATA, EV, heart = null;
let mode = 'full';
let cur = null;           // selected patient (object) or null = new patient
let S = {}, S0 = {};      // current and original input values (keyed by input spec key)
let selected = null;      // highlighted vessel
let specs = [];           // input specs for the active mode
const CP_OPTS = [[0, 'None / no chest pain'], [1, 'Typical angina'], [2, 'Atypical chest pain'], [3, 'Non-anginal chest pain']];

/* ================================================================ models as linear predictors */
const zLin = (m, v) => { let s = m.intercept; for (let i = 0; i < v.length; i++) s += m.coef[i] * (v[i] - m.mean[i]) / m.scale[i]; return s; };
const zBoot = (m, row, v) => { let s = row[0]; for (let i = 0; i < v.length; i++) s += row[i + 1] * (v[i] - m.mean[i]) / m.scale[i]; return s; };

const MODES = {
  full: {
    tasks: ['CAD', ...VESSELS],
    lin: t => FULL.tasks[t],
    thr: () => FULL.rule_out_threshold,
    vec(s) {
      const bmi = s.Weight / (s.Length / 100) ** 2;
      return FULL.features.map(f => {
        if (f.key === 'Typical Chest Pain') return +(s.cp === 1);
        if (f.key === 'Atypical') return +(s.cp === 2);
        if (f.key === 'Nonanginal') return +(s.cp === 3);
        if (f.key === 'BMI') return bmi;
        if (f.key === 'Obesity') return +(bmi >= 25);
        return s[f.key];
      });
    },
    labels: () => FULL.features.map(f => f.label),
    esc: s => escPTP(s.Age, s.Sex, s.cp, s.Dyspnea),
  },
  bedside: {
    tasks: ['CAD'],
    lin: () => BED,
    thr: () => (cur && cur.site ? BED.site_thresholds[cur.site] : BED.rule_out_threshold),
    vec(s) {
      return [s.age, s.male, +(s.cp === 1), +(s.cp === 2), +(s.cp === 3),
        s.sbp == null ? BED.medians.sbp : s.sbp, +(s.sbp == null), s.fbs == null ? 0 : s.fbs, +(s.fbs == null), s.stt, s.lvh];
    },
    labels: () => BED.features.map(k => BED.labels[k]),
    esc: s => escPTP(s.age, s.male, s.cp),
  },
};
const M = () => MODES[mode];
const edited = () => specs.some(sp => S[sp.key] !== S0[sp.key]);
function baseProb(t) { // the honest, held-out prediction for a cohort patient
  if (mode === 'full') return cur.oof[t];
  return t === 'CAD' ? cur.loso : null;
}
function prob(t) {
  const m = M().lin(t), v = M().vec(S);
  if (!cur) return sigm(zLin(m, v));
  const b = baseProb(t); if (b == null) return null;
  if (!edited()) return b;
  return sigm(logit(b) + zLin(m, v) - zLin(m, M().vec(S0)));
}
function probRange(t, p) { // 90% bootstrap range of the coefficients, applied around the displayed value
  const m = M().lin(t), v = M().vec(S), z = zLin(m, v);
  const d = m.boot.map(row => zBoot(m, row, v) - z);
  return [sigm(logit(p) + quantile(d, 0.05)), sigm(logit(p) + quantile(d, 0.95))];
}

/* ================================================================ input specs */
function fullSpecs() {
  const out = [], skip = new Set(['Typical Chest Pain', 'Atypical', 'Nonanginal']);
  const coef = FULL.tasks.CAD.coef, F = Object.fromEntries(FULL.features.map((f, i) => [f.key, { ...f, i }]));
  for (const f of FULL.features) {
    if (skip.has(f.key)) continue;
    const i = F[f.key].i, s = { key: f.key, label: f.label, group: f.group, w: Math.abs(coef[i]), min: f.min, max: f.max, p01: f.p01, p99: f.p99 };
    if (f.key === 'BMI' || f.key === 'Obesity') s.kind = 'derived';
    else if (f.key === 'Sex') { s.kind = 'select'; s.opts = [[1, 'Male'], [0, 'Female']]; s.label = 'Sex'; s.p01 = null; }
    else if (f.kind === 'binary') { s.kind = 'binary'; s.p01 = null; }
    else { s.kind = 'number'; s.step = f.kind !== 'continuous' ? 1 : (f.max - f.min > 2000 ? 100 : f.max - f.min > 20 ? 1 : 0.1); }
    out.push(s);
  }
  const cpw = Math.max(...['Typical Chest Pain', 'Atypical', 'Nonanginal'].map(k => Math.abs(coef[F[k].i])));
  out.splice(0, 0, { key: 'cp', label: 'Chest pain type', group: 'symptoms', kind: 'select', opts: CP_OPTS, w: cpw });
  return out;
}
function bedSpecs() {
  const P = BED.patients, rng = k => { const v = P.map(p => p[k]).filter(x => x != null); return { min: Math.min(...v), max: Math.max(...v), p01: quantile(v, 0.01), p99: quantile(v, 0.99) }; };
  return [
    { key: 'age', label: 'Age (years)', kind: 'number', step: 1, ...rng('age') },
    { key: 'male', label: 'Sex', kind: 'select', opts: [[1, 'Male'], [0, 'Female']] },
    { key: 'cp', label: 'Chest pain type', kind: 'select', opts: CP_OPTS },
    { key: 'sbp', label: 'Resting systolic BP (mmHg)', kind: 'number', step: 1, unknown: 'Not measured', ...rng('sbp') },
    { key: 'fbs', label: 'Fasting glucose > 120 mg/dL', kind: 'select', opts: [[0, 'No'], [1, 'Yes'], ['', 'Not measured']] },
    { key: 'stt', label: 'Resting ECG: ST-T abnormality', kind: 'binary' },
    { key: 'lvh', label: 'Resting ECG: LV hypertrophy', kind: 'binary' },
  ];
}
function fieldHTML(sp) {
  const id = 'in-' + sp.key.replace(/\W/g, '_');
  const lab = `<label for="${id}">${esc(sp.label)}</label>`;
  if (sp.kind === 'binary') return `<div class="field" data-k="${esc(sp.key)}">${lab}<input class="switch" id="${id}" type="checkbox" role="switch"></div>`;
  if (sp.kind === 'select') return `<div class="field" data-k="${esc(sp.key)}">${lab}<select id="${id}">${sp.opts.map(([v, t]) => `<option value="${v}">${esc(t)}</option>`).join('')}</select></div>`;
  if (sp.kind === 'derived') return `<div class="field is-derived" data-k="${esc(sp.key)}">${lab}<input id="${id}" type="text" readonly tabindex="-1" style="width:84px;text-align:right" title="Calculated from weight and height"></div>`;
  return `<div class="field" data-k="${esc(sp.key)}">${lab}<input id="${id}" type="number" min="${sp.min}" max="${sp.max}" step="${sp.step}" inputmode="decimal">
    <input type="range" min="${sp.min}" max="${sp.max}" step="${sp.step}" aria-label="${esc(sp.label)} slider">
    ${sp.unknown ? `<label class="unk"><input type="checkbox" class="unk__box"> ${esc(sp.unknown)}</label>` : ''}<span class="err" hidden></span></div>`;
}
function buildInputs() {
  specs = mode === 'full' ? fullSpecs() : bedSpecs();
  if (mode === 'full') {
    const top = [...specs].filter(s => s.kind !== 'derived').sort((a, b) => b.w - a.w).slice(0, 10).map(s => s.key);
    ['Age', 'Sex'].forEach(k => top.includes(k) || top.push(k));
    $('#inputs-top').innerHTML = top.map(k => fieldHTML(specs.find(s => s.key === k))).join('');
    const groups = { history: 'History', symptoms: 'Symptoms', exam: 'Examination', ecg: 'ECG', labs: 'Blood tests', echo: 'Echocardiogram' };
    $('#inputs-all').innerHTML = Object.entries(groups).map(([g, name]) => {
      const fs = specs.filter(s => s.group === g && !top.includes(s.key));
      return fs.length ? `<p class="grp">${name}</p>` + fs.map(fieldHTML).join('') : '';
    }).join('');
    $('#more').hidden = false;
  } else {
    $('#inputs-top').innerHTML = specs.map(fieldHTML).join('');
    $('#inputs-all').innerHTML = ''; $('#more').hidden = true;
  }
  $$('.field').forEach(el => {
    const sp = specs.find(s => s.key === el.dataset.k); if (!sp || sp.kind === 'derived') return;
    const num = $('input[type=number]', el), range = $('input[type=range]', el), sel = $('select', el), sw = $('.switch', el), unk = $('.unk__box', el), err = $('.err', el);
    const setErr = msg => { el.classList.toggle('is-invalid', !!msg); if (err) { err.hidden = !msg; err.textContent = msg || ''; } };
    if (num) {
      const accept = (raw, commit) => {
        const v = raw === '' ? NaN : Number(raw);
        if (Number.isNaN(v)) { setErr(commit ? '' : 'Enter a number'); if (commit) syncField(el); return; }
        if (v < sp.min || v > sp.max) { // never clamp silently while the user is still typing
          if (!commit) { setErr(`Allowed range ${sp.min}–${sp.max}`); return; }
          S[sp.key] = Math.min(sp.max, Math.max(sp.min, v));
        } else S[sp.key] = v;
        setErr(''); if (unk) unk.checked = false; syncField(el); render();
      };
      num.addEventListener('input', () => accept(num.value, false));
      num.addEventListener('change', () => accept(num.value, true));
      range.addEventListener('input', () => accept(range.value, true));
    }
    if (unk) unk.addEventListener('change', () => { S[sp.key] = unk.checked ? null : (S0[sp.key] ?? Math.round(BED.medians.sbp)); setErr(''); syncField(el); render(); });
    if (sel) sel.addEventListener('change', () => { S[sp.key] = sel.value === '' ? null : Number(sel.value); syncField(el); render(); });
    if (sw) sw.addEventListener('change', () => { S[sp.key] = +sw.checked; syncField(el); render(); });
  });
}
function syncField(el) {
  const sp = specs.find(s => s.key === el.dataset.k); if (!sp) return;
  const v = S[sp.key];
  if (sp.kind === 'derived') {
    const bmi = S.Weight / (S.Length / 100) ** 2;
    $('input', el).value = sp.key === 'BMI' ? bmi.toFixed(1) : (bmi >= 25 ? 'Yes' : 'No');
    el.classList.toggle('is-changed', bmi.toFixed(2) !== (S0.Weight / (S0.Length / 100) ** 2).toFixed(2));
    return;
  }
  const num = $('input[type=number]', el), range = $('input[type=range]', el), sel = $('select', el), sw = $('.switch', el), unk = $('.unk__box', el);
  if (num) { if (document.activeElement !== num) num.value = v == null ? '' : v; range.value = v == null ? sp.min : v; num.disabled = range.disabled = v == null; if (unk) unk.checked = v == null; }
  if (sel) sel.value = v == null ? '' : String(v);
  if (sw) sw.checked = !!v;
  el.classList.toggle('is-changed', v !== S0[sp.key]);
  el.classList.toggle('is-unusual', sp.p01 != null && v != null && (v < sp.p01 || v > sp.p99));
}
const syncAll = () => $$('.field').forEach(syncField);

/* ================================================================ patients */
function stateFromFull(xv) {
  const s = {}; FULL.features.forEach((f, i) => { s[f.key] = xv[i]; });
  s.cp = s['Typical Chest Pain'] ? 1 : s.Atypical ? 2 : s.Nonanginal ? 3 : 0;
  return s;
}
const stateFromBed = p => ({ age: p.age, male: p.male, cp: p.cp, sbp: p.sbp, fbs: p.fbs, stt: p.stt, lvh: p.lvh });
const blankState = () => mode === 'full' ? stateFromFull(FULL.features.map(f => f.median))
  : { age: 55, male: 1, cp: 2, sbp: 130, fbs: 0, stt: 0, lvh: 0 };
const truthCAD = p => mode === 'full' ? p.truth.CAD : p.cad;
const ruledOut = p => mode === 'full' ? p.oof.CAD < FULL.rule_out_threshold : p.loso < BED.site_thresholds[p.site];
function verdictOf(p) {
  const low = ruledOut(p), cad = truthCAD(p) === 1;
  if (low && !cad) return { level: 'ok', text: 'Correct: model said low risk and the angiogram was normal.' };
  if (low && cad) return { level: 'high', text: 'Missed: model said low risk, but angiography found CAD. This is the failure that matters.' };
  if (!low && cad) return { level: 'ok', text: 'Correct: model flagged CAD and angiography confirmed it.' };
  return { level: 'mid', text: 'False alarm: model flagged CAD but the angiogram was normal.' };
}
function featuredBedside() {
  const P = BED.patients.map((p, i) => ({ ...p, i }));
  const pick = (title, cond, key) => { const c = P.filter(cond).sort((a, b) => key(a) - key(b))[0]; return c ? { title, i: c.i } : null; };
  return [
    pick('Budapest: normal angiogram, ruled out', p => p.site === 'Budapest' && !p.cad && ruledOut(p), p => p.loso),
    pick('Cleveland: CAD correctly flagged', p => p.site === 'Cleveland' && p.cad && !ruledOut(p), p => -p.loso),
    pick('Zurich: CAD without chest pain', p => p.site === 'Zurich' && p.cad && p.cp === 0 && !ruledOut(p), p => -p.loso),
    pick('MISSED in Tehran: model did not transfer', p => p.site === 'Tehran' && p.cad && ruledOut(p), p => p.loso),
    pick('Long Beach: false alarm', p => p.site === 'Long Beach' && !p.cad, p => -p.loso),
  ].filter(Boolean);
}
function buildPatients() {
  const sel = $('#patient-select');
  if (mode === 'full') {
    $('#cases').innerHTML = DATA.featured.map(c => {
      const p = DATA.patients[c.row], v = verdictOf(p);
      return `<button class="case" role="option" data-id="${c.row}" aria-selected="false"><i data-level="${v.level}"></i><span>${esc(c.title)}</span><small class="num">${pct(p.oof.CAD)}</small></button>`;
    }).join('');
    $('#select-label').textContent = 'or any of the 303 patients (Tehran)';
    sel.innerHTML = `<option value="-1">New patient (enter values)</option>` + DATA.patients.map(p =>
      `<option value="${p.row}">Patient #${p.row + 1}: ${p.truth.CAD ? 'CAD' : 'normal angiogram'} · score ${pct(p.oof.CAD)}</option>`).join('');
  } else {
    $('#cases').innerHTML = featuredBedside().map(c => {
      const p = BED.patients[c.i], v = verdictOf(p);
      return `<button class="case" role="option" data-id="${c.i}" aria-selected="false"><i data-level="${v.level}"></i><span>${esc(c.title)}</span><small class="num">${pct(p.loso)}</small></button>`;
    }).join('');
    $('#select-label').textContent = `or any of the ${BED.patients.length.toLocaleString()} patients from 5 hospitals`;
    const n = {};
    sel.innerHTML = `<option value="-1">New patient (enter values)</option>` + BED.sites.map(site => `<optgroup label="${esc(site)}">` +
      BED.patients.map((p, i) => [p, i]).filter(([p]) => p.site === site).map(([p, i]) => {
        n[site] = (n[site] || 0) + 1;
        return `<option value="${i}">${esc(site)} #${n[site]}: ${p.cad ? 'CAD' : 'normal'} · score ${pct(p.loso)}</option>`;
      }).join('') + '</optgroup>').join('');
  }
  $$('.case').forEach(b => b.addEventListener('click', () => selectPatient(+b.dataset.id)));
}
function selectPatient(id) {
  cur = id < 0 ? null : (mode === 'full' ? DATA.patients[id] : BED.patients[id]);
  S0 = cur ? (mode === 'full' ? stateFromFull(cur.x) : stateFromBed(cur)) : blankState();
  S = { ...S0 }; selected = null;
  $('#patient-select').value = String(id);
  $$('.case').forEach(b => b.setAttribute('aria-selected', String(+b.dataset.id === id)));
  $$('.field').forEach(el => { el.classList.remove('is-invalid'); const e = $('.err', el); if (e) e.hidden = true; });
  syncAll(); render();
}
function setMode(m) {
  mode = m;
  $$('[data-mode]').forEach(b => b.setAttribute('aria-selected', String(b.dataset.mode === m)));
  const ex = EV.external.per_site.filter(r => !r.site.startsWith('All')), a = ex.map(r => r.models.bedside_lr.auroc), e = ex.map(r => r.models.esc.auroc);
  const dep = EV.models.find(x => x.key === EV.config.deployed);
  $('#mode-note').innerHTML = m === 'full'
    ? `Needs ECG, blood tests and echo. <b>Internal validation only</b> (303 patients, one hospital): AUROC ${dep.tasks.CAD.auroc.toFixed(2)} vs ${EV.models.find(x => x.key === 'esc').tasks.CAD.auroc.toFixed(2)} for the guideline.`
    : `Needs only history, BP, a glucose test and a resting ECG. <b>Tested in hospitals it never saw</b>: AUROC ${Math.min(...a).toFixed(2)}–${Math.max(...a).toFixed(2)} (guideline ${Math.min(...e).toFixed(2)}–${Math.max(...e).toFixed(2)}). Check it on local patients before relying on it.`;
  $('#panel-vessels').hidden = m !== 'full';
  buildInputs(); buildPatients();
  selectPatient(m === 'full' ? DATA.featured[0].row : featuredBedside()[0].i);
}

/* ================================================================ render result */
function render() {
  $$('.field.is-derived').forEach(syncField); // BMI / obesity follow weight and height
  const P = Object.fromEntries(M().tasks.map(t => [t, prob(t)]));
  const isEdit = cur && edited(), THR = M().thr(), low = P.CAD < THR;
  const level = low ? 'ok' : lvl(P.CAD);
  $('#edit-state').textContent = !cur ? 'new patient' : isEdit ? 'edited (simulation)' : 'recorded values';
  $('#risk-chip').dataset.level = level; $('#cad-risk').textContent = pct(P.CAD);
  $('#panel-result').dataset.level = level; $('#res-pct').textContent = pct(P.CAD);
  $('#res-decision').textContent = low ? 'Below rule-out line: low risk' : 'Above rule-out line: angiography as planned';
  const held = mode === 'full' ? 'the model never saw this patient' : `model trained without ${cur && cur.site}`;
  $('#res-source').textContent = !cur ? 'Live model on the values you entered.'
    : isEdit ? 'Simulated: held-out score shifted by the model’s response to your edits.'
    : `Held-out score: ${held}.`;
  const [lo, hi] = probRange('CAD', P.CAD);
  $('#res-range').textContent = `Statistical uncertainty (90% bootstrap range): ${pct(lo)} – ${pct(hi)}` + (lo < THR && hi >= THR ? '. The range crosses the rule-out line, so treat this as indeterminate.' : '');
  const unusual = specs.filter(sp => sp.p01 != null && S[sp.key] != null && (S[sp.key] < sp.p01 || S[sp.key] > sp.p99)).map(sp => sp.label.replace(/ \(.*\)$/, ''));
  const warns = [];
  if (unusual.length) warns.push(`${unusual.length} input${unusual.length > 1 ? 's are' : ' is'} outside the range seen in training (${unusual.slice(0, 3).join(', ')}${unusual.length > 3 ? '…' : ''}). The estimate is less reliable.`);
  if (mode === 'bedside' && (S.sbp == null || S.fbs == null)) warns.push('Some inputs are marked "not measured". The model uses a missing-value indicator for them, which is less accurate.');
  if (mode === 'bedside' && cur && cur.site === 'Tehran') warns.push('This hospital is where the bedside model transferred worst (it missed many CAD cases). See Evidence.');
  $('#res-warn').hidden = !warns.length; $('#res-warn').textContent = warns.join(' ');
  $('#meter-fill').style.width = P.CAD * 100 + '%'; $('#meter-thr').style.left = THR * 100 + '%';
  $('#meter-thr-label').textContent = `rule-out line ${pct(THR)}`;
  $('#res-esc').textContent = pct(M().esc(S));
  $('#res-truth').textContent = cur ? (truthCAD(cur) ? 'CAD (≥50% stenosis)' : 'Normal angiogram') : 'unknown';
  const vd = $('#res-verdict');
  if (cur && !isEdit) { const v = verdictOf(cur); vd.hidden = false; vd.dataset.level = v.level; vd.textContent = v.text; } else vd.hidden = true;

  if (mode === 'full') $$('.vcard').forEach(c => {
    const n = c.dataset.vessel; c.dataset.level = lvl(P[n]);
    $('.vcard__pct', c).textContent = pct(P[n]);
    $('.vcard__truth', c).textContent = cur ? `Angiography: ${cur.truth[n] ? 'stenotic' : 'normal'}` : 'Angiography: unknown';
    c.classList.toggle('is-selected', selected === n);
  });

  const m = M().lin('CAD'), v = M().vec(S), labels = M().labels();
  const items = v.map((x, i) => ({ i, c: m.coef[i] * (x - m.mean[i]) / m.scale[i], label: labels[i], x }))
    .filter(d => Math.abs(d.c) > 0.005).sort((a, b) => Math.abs(b.c) - Math.abs(a.c)).slice(0, 8);
  const mx = Math.max(1.2, ...items.map(d => Math.abs(d.c)));
  const showVal = d => (d.x === 0 || d.x === 1) && !/age|bp|\(/i.test(d.label) ? (d.x ? 'Yes' : 'No') : (Math.abs(d.x) >= 100 ? Math.round(d.x) : +d.x.toFixed(1));
  $('#contrib').innerHTML = items.map(d => {
    const w = Math.abs(d.c) / mx * 50, left = d.c >= 0 ? 50 : 50 - w;
    return `<li><span title="${esc(d.label)}">${esc(d.label.replace(/ \(.*\)$/, ''))} <small>${esc(showVal(d))}</small></span>
      <span class="contrib__track"><i style="left:${left}%;width:${w}%;background:var(${d.c >= 0 ? '--s-up' : '--s-down'})"></i></span>
      <b class="num">${d.c >= 0 ? '+' : '−'}${Math.abs(d.c).toFixed(2)}</b></li>`;
  }).join('') || '<li class="muted">No input differs from the average patient.</li>';
  setECG();
  if (heart) { heart.select(selected); heart.update(P, lvl, pct); }
}

/* ================================================================ ECG (schematic from recorded findings) */
const ecg = { hr: 70, st: 0, t: 0.3, q: 0, r: 1.1, w: 1 };
function ecgFlags() {
  if (mode === 'bedside') return { hr: 70, std: S.stt, ti: S.stt, lvh: S.lvh };
  return { hr: S.PR, ste: S['St Elevation'], std: S['St Depression'], ti: S.Tinversion, q: S['Q Wave'], lvh: S.LVH, bbb: S.BBB_LBBB || S.BBB_RBBB, lbbb: S.BBB_LBBB, rbbb: S.BBB_RBBB, prp: S['Poor R Progression'] };
}
function setECG() {
  const g = ecgFlags();
  Object.assign(ecg, { hr: g.hr || 70, st: g.ste ? 0.2 : g.std ? -0.12 : 0, t: g.ti ? -0.25 : 0.3, q: g.q ? 0.35 : 0.08, r: g.lvh ? 1.6 : 1.1, w: g.bbb ? 1.8 : 1 });
  const found = mode === 'bedside'
    ? [[g.std, 'ST-T abnormality'], [g.lvh, 'LVH']].filter(([f]) => f).map(([, n]) => n)
    : [[g.ste, 'ST elevation'], [g.std, 'ST depression'], [g.ti, 'T inversion'], [g.q, 'Q wave'], [g.lvh, 'LVH'], [g.lbbb, 'LBBB'], [g.rbbb, 'RBBB'], [g.prp, 'poor R progression']].filter(([f]) => f).map(([, n]) => n);
  const b = $('#ecg-badge');
  b.dataset.level = found.length ? (g.ste || g.q ? 'high' : 'mid') : 'ok';
  b.textContent = found.length ? `${found.length} finding${found.length > 1 ? 's' : ''}` : 'No abnormal findings';
  $('#ecg-findings').textContent = (found.length ? found.join(', ') : 'none') + (mode === 'full' ? `, pulse ${Math.round(ecg.hr)} bpm` : '');
}
function initECG() {
  const cv = $('#ecg-canvas'), ctx = cv.getContext('2d'), SPEED = 100;
  let w = 0, h = 0, xx = 0, prev = null, tSim = 0, last = performance.now();
  const size = () => { const r = cv.getBoundingClientRect(), d = Math.min(devicePixelRatio, 2); w = r.width; h = r.height; cv.width = Math.round(w * d); cv.height = Math.round(h * d); ctx.setTransform(d, 0, 0, d, 0, 0); xx = 0; prev = null; };
  new ResizeObserver(size).observe(cv);
  const gs = (u, c, k) => Math.exp(-(((u - c) / k) ** 2)), sg = v => 1 / (1 + Math.exp(-v));
  const mv = tt => {
    const rr = 60 / ecg.hr, u = (tt % rr) - Math.min(0.32, rr * 0.4), W = ecg.w;
    return 0.15 * gs(u, -0.2, 0.035) - ecg.q * gs(u, -0.03 * W, 0.012 * W) + ecg.r * gs(u, 0, 0.014 * W) - 0.28 * gs(u, 0.03 * W, 0.014 * W)
      + ecg.st * (sg((u - 0.05 * W) / 0.01) - sg((u - 0.24) / 0.03)) + ecg.t * gs(u, 0.28, 0.06);
  };
  let col = '#30d158', n0 = 0;
  const frame = now => {
    requestAnimationFrame(frame);
    const dt = Math.min(0.05, (now - last) / 1000); last = now; if (!w || document.hidden) return;
    if (++n0 % 30 === 0) col = getComputedStyle($('#ecg-badge')).getPropertyValue('--c').trim() || '#30d158';
    ctx.globalCompositeOperation = 'destination-out'; ctx.fillStyle = 'rgba(0,0,0,0.06)'; ctx.fillRect(0, 0, w, h); ctx.globalCompositeOperation = 'source-over';
    const base = h * 0.6, n = Math.max(1, Math.ceil(SPEED * dt / 2)); let y = base;
    ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.lineJoin = 'round';
    for (let i = 0; i < n; i++) {
      const nx = xx + SPEED * dt / n; tSim += reduced ? 0 : dt / n; y = base - mv(tSim) * h * 0.32;
      if (nx >= w) { xx = 0; prev = null; continue; }
      if (prev !== null) { ctx.beginPath(); ctx.moveTo(xx, prev); ctx.lineTo(nx, y); ctx.stroke(); }
      xx = nx; prev = y;
    }
    ctx.clearRect(xx + 3, 0, 12, h);
  };
  requestAnimationFrame(frame);
}

/* ================================================================ printable report */
function printReport() {
  const P = Object.fromEntries(M().tasks.map(t => [t, prob(t)])), [lo, hi] = probRange('CAD', P.CAD), THR = M().thr();
  const rows = specs.filter(sp => sp.kind !== 'derived').map(sp => {
    const v = S[sp.key], txt = v == null ? 'not measured' : sp.opts ? (sp.opts.find(o => String(o[0]) === String(v)) || [0, v])[1] : sp.kind === 'binary' ? (v ? 'Yes' : 'No') : v;
    return `<tr><td>${esc(sp.label)}</td><td>${esc(txt)}${v !== S0[sp.key] && cur ? ' (edited)' : ''}</td></tr>`;
  }).join('');
  $('#report').innerHTML = `<h1>CardioAI: pre-angiography risk summary</h1>
    <p class="note">${new Date().toLocaleString()} · Model: ${mode === 'full' ? 'Full workup (55 inputs, internally validated, 303 patients)' : 'Bedside (7 inputs, 1,223 patients, 5 hospitals, leave-one-hospital-out validated)'}</p>
    <p class="big">${pct(P.CAD)} probability of CAD</p>
    <p>90% bootstrap range ${pct(lo)}–${pct(hi)}. ${P.CAD < THR ? 'Below' : 'Above'} the rule-out line (${pct(THR)}, set for ≥95% sensitivity). ESC 2019 guideline estimate: ${pct(M().esc(S))}.</p>
    ${mode === 'full' ? `<p>Vessel estimates (weak, AUROC 0.68–0.83): LAD ${pct(P.LAD)}, LCX ${pct(P.LCX)}, RCA ${pct(P.RCA)}.</p>` : ''}
    <h2>Inputs</h2><table>${rows}</table>
    <h2>Important</h2><p class="note">Research prototype, not a medical device. In validation about 1 in 5 patients labelled "low risk" still had CAD. This summary must not be the reason a patient does not receive the test they need. Source: github.com/aryanjaykadu/CardioAI3D</p>`;
  window.print();
}

/* ================================================================ batch scoring (CSV, in-browser only) */
function parseCSV(text) {
  const rows = [], t = text.replace(/^﻿/, ''); let row = [], f = '', q = false;
  for (let i = 0; i < t.length; i++) {
    const c = t[i];
    if (q) { if (c === '"') { if (t[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
    else if (c === '"') q = true;
    else if (c === ',') { row.push(f); f = ''; }
    else if (c === '\n' || c === '\r') { if (c === '\r' && t[i + 1] === '\n') i++; row.push(f); f = ''; if (row.some(x => x.trim() !== '')) rows.push(row); row = []; }
    else f += c;
  }
  row.push(f); if (row.some(x => x.trim() !== '')) rows.push(row);
  return rows;
}
const toCSV = rows => rows.map(r => r.map(v => { const s = v == null ? '' : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; }).join(',')).join('\n');
function download(name, text) { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type: 'text/csv' })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); }
const YES = new Set(['1', 'y', 'yes', 'true', 't']), NO = new Set(['0', 'n', 'no', 'false', 'f']);
const BED_COLS = ['patient_id', 'age', 'sex', 'chest_pain', 'systolic_bp', 'fasting_glucose_over_120', 'ecg_st_t_abnormal', 'ecg_lvh'];
function batchTemplate(m) {
  if (m === 'bedside') return toCSV([BED_COLS, ['A-001', 62, 'male', 'typical', 138, 'no', 'yes', 'no'], ['A-002', 48, 'female', 'atypical', '', '', 'no', 'no']]);
  return toCSV([['patient_id', ...FULL.features.map(f => f.key)], ['example-1', ...FULL.features.map(f => f.median)]]);
}
function scoreRow(m, rec) { // -> {p, lo, hi, esc, errors[]}
  const errs = [];
  const yn = (k, allowBlank) => { const s = String(rec[k] ?? '').trim().toLowerCase(); if (s === '' && allowBlank) return null; if (YES.has(s)) return 1; if (NO.has(s)) return 0; errs.push(`${k}: expected yes/no`); return 0; };
  const num = (k, lo, hi, allowBlank) => { const s = String(rec[k] ?? '').trim(); if (s === '' && allowBlank) return null; const v = Number(s); if (s === '' || Number.isNaN(v)) { errs.push(`${k}: missing or not a number`); return NaN; } if (v < lo || v > hi) errs.push(`${k}: ${v} outside ${lo}–${hi}`); return v; };
  const sexOf = k => { const sx = String(rec[k] ?? '').trim().toLowerCase(); if (['m', 'male', '1'].includes(sx)) return 1; if (['f', 'female', 'fmale', '0'].includes(sx)) return 0; errs.push(`${k}: use male/female`); return 0; };
  let s, mod;
  if (m === 'bedside') {
    const cpRaw = String(rec.chest_pain ?? '').trim().toLowerCase().replace(/[^a-z]/g, '');
    const cp = { typical: 1, typicalangina: 1, atypical: 2, nonanginal: 3, none: 0, asymptomatic: 0, no: 0 }[cpRaw];
    if (cp == null) errs.push('chest_pain: use typical / atypical / nonanginal / none');
    s = { age: num('age', 18, 100), male: sexOf('sex'), cp: cp ?? 0, sbp: num('systolic_bp', 70, 250, true), fbs: yn('fasting_glucose_over_120', true), stt: yn('ecg_st_t_abnormal'), lvh: yn('ecg_lvh') };
    mod = MODES.bedside;
  } else {
    s = {};
    for (const f of FULL.features) {
      if (f.key === 'Sex') s.Sex = sexOf('Sex');
      else if (f.kind === 'binary') s[f.key] = yn(f.key);
      else s[f.key] = num(f.key, f.min, f.max);
    }
    if ((s['Typical Chest Pain'] || 0) + (s.Atypical || 0) + (s.Nonanginal || 0) > 1) errs.push('choose only one chest pain type');
    s.cp = s['Typical Chest Pain'] ? 1 : s.Atypical ? 2 : s.Nonanginal ? 3 : 0;
    mod = MODES.full;
  }
  if (errs.length) return { errors: errs };
  const lin = m === 'bedside' ? BED : FULL.tasks.CAD, v = mod.vec(s), z = zLin(lin, v), p = sigm(z);
  const d = lin.boot.map(row => zBoot(lin, row, v) - z);
  return { p, lo: sigm(z + quantile(d, 0.05)), hi: sigm(z + quantile(d, 0.95)), esc: mod.esc(s), errors: [] };
}
function initBatch() {
  const dlg = $('#batch'); let out = null;
  $('#btn-batch').addEventListener('click', () => { $('#batch-model').value = mode; dlg.showModal(); });
  $('#batch-template').addEventListener('click', () => { const m = $('#batch-model').value; download(`cardioai_${m}_template.csv`, batchTemplate(m)); });
  $('#batch-file').addEventListener('change', async e => {
    const file = e.target.files[0]; if (!file) return;
    const m = $('#batch-model').value, thr = m === 'bedside' ? BED.rule_out_threshold : FULL.rule_out_threshold;
    const fail = msg => { $('#batch-status').textContent = msg; $('#batch-table').innerHTML = ''; $('#batch-download').hidden = true; e.target.value = ''; };
    if (file.size > 5e6) return fail('File is larger than 5 MB. Split it into smaller files.');
    const rows = parseCSV(await file.text());
    if (rows.length < 2) return fail('No data rows found.');
    const head = rows[0].map(h => h.trim()), need = m === 'bedside' ? BED_COLS.slice(1) : FULL.features.map(f => f.key);
    const missing = need.filter(k => !head.includes(k));
    if (missing.length) return fail(`Missing columns: ${missing.slice(0, 8).join(', ')}${missing.length > 8 ? '…' : ''}. Download the template for the exact names.`);
    out = [['patient_id', 'p_cad', 'range_90_low', 'range_90_high', 'decision', 'esc_2019_ptp', 'problems']];
    let ok = 0, bad = 0, low = 0;
    const res = rows.slice(1).map((r, i) => {
      const rec = Object.fromEntries(head.map((h, j) => [h, r[j]])), id = (rec.patient_id || '').trim() || `row ${i + 2}`, s = scoreRow(m, rec);
      if (s.errors.length) { bad++; out.push([id, '', '', '', 'NOT SCORED', '', s.errors.join('; ')]); return { id, s }; }
      ok++; const isLow = s.p < thr; low += isLow;
      out.push([id, s.p.toFixed(4), s.lo.toFixed(4), s.hi.toFixed(4), isLow ? 'below rule-out line' : 'refer as planned', s.esc.toFixed(2), '']);
      return { id, s, isLow };
    });
    $('#batch-status').innerHTML = `Scored <b>${ok}</b> of ${res.length} rows${bad ? `, <b>${bad}</b> could not be scored (see "problems")` : ''}. ${low} below the rule-out line (${pct(thr)}). Highest risk first.`;
    const sorted = res.filter(r => !r.s.errors.length).sort((a, b) => b.s.p - a.s.p).concat(res.filter(r => r.s.errors.length));
    $('#batch-table').innerHTML = `<thead><tr><th>Patient</th><th>P(CAD)</th><th>90% range</th><th>Decision</th><th>ESC 2019</th></tr></thead><tbody>` +
      sorted.slice(0, 200).map(r => r.s.errors.length ? `<tr><td>${esc(r.id)}</td><td class="bad" colspan="4">${esc(r.s.errors.join('; '))}</td></tr>`
        : `<tr><td>${esc(r.id)}</td><td>${pct(r.s.p)}</td><td>${pct(r.s.lo)}–${pct(r.s.hi)}</td><td>${r.isLow ? 'below rule-out line' : 'refer as planned'}</td><td>${pct(r.s.esc)}</td></tr>`).join('') + '</tbody>';
    $('#batch-download').hidden = false; e.target.value = '';
  });
  $('#batch-download').addEventListener('click', () => out && download('cardioai_scored.csv', toCSV(out)));
}

/* ================================================================ evidence charts (inline SVG) */
const svgNS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(svgNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent && parent.appendChild(e); return e; };
function tooltip(host) { let t = $('.tip', host); if (!t) { t = document.createElement('div'); t.className = 'tip'; host.appendChild(t); } return t; }

function tradeoffChart() {
  const host = $('#tradeoff-chart'); host.innerHTML = '';
  const Wd = 720, Ht = 340, m = { l: 52, r: 120, t: 16, b: 44 };
  const svg = el('svg', { viewBox: `0 0 ${Wd} ${Ht}`, role: 'img', 'aria-label': 'Trade-off curve: normal angiograms avoided versus CAD patients missed, per 100 patients' }, host);
  const pts = s => EV.curve.filter(d => d.src === s);
  const xMax = 20, yMax = Math.ceil(Math.max(...EV.curve.map(d => d.avoided)) / 5) * 5;
  const sx = v => m.l + Math.min(v, xMax) / xMax * (Wd - m.l - m.r), sy = v => Ht - m.b - v / yMax * (Ht - m.t - m.b);
  const g = el('g', { class: 'grid' }, svg);
  for (let v = 0; v <= yMax; v += 5) { el('line', { x1: m.l, x2: Wd - m.r, y1: sy(v), y2: sy(v) }, g); el('text', { x: m.l - 8, y: sy(v) + 4, 'text-anchor': 'end' }, svg).textContent = v; }
  for (let v = 0; v <= xMax; v += 5) el('text', { x: sx(v), y: Ht - m.b + 18, 'text-anchor': 'middle' }, svg).textContent = v;
  el('text', { x: (m.l + Wd - m.r) / 2, y: Ht - 6, 'text-anchor': 'middle' }, svg).textContent = 'CAD patients missed per 100 referred (harm)';
  el('text', { x: 14, y: (Ht - m.b + m.t) / 2, transform: `rotate(-90 14 ${(Ht - m.b + m.t) / 2})`, 'text-anchor': 'middle' }, svg).textContent = 'Normal angiograms avoided per 100 (benefit)';
  const series = [['model', 'var(--s-model)', 'CardioAI (full)'], ['esc', 'var(--s-esc)', 'ESC 2019 table']];
  series.forEach(([s, c, name]) => {
    const p = pts(s).filter(d => d.missed <= xMax + 0.01).sort((a, b) => a.missed - b.missed || a.avoided - b.avoided);
    el('path', { d: 'M' + p.map(d => `${sx(d.missed).toFixed(1)},${sy(d.avoided).toFixed(1)}`).join('L'), fill: 'none', stroke: c, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-dasharray': s === 'esc' ? '5 4' : '' }, svg);
    const last = p[p.length - 1];
    el('text', { x: sx(last.missed) + 8, y: sy(last.avoided) + 4, class: 'lbl' }, svg).textContent = name;
  });
  const markers = series.map(([, c]) => el('circle', { r: 6, fill: c, stroke: '#0b0b12', 'stroke-width': 2 }, svg));
  const tip = tooltip(host);
  const hit = el('rect', { x: m.l, y: m.t, width: Wd - m.l - m.r, height: Ht - m.t - m.b, fill: 'transparent' }, svg);
  hit.addEventListener('pointermove', e => {
    const r = svg.getBoundingClientRect(), k = Wd / r.width, mx = (e.clientX - r.left) * k, my = (e.clientY - r.top) * k;
    let best = null, bd = 1e9;
    EV.curve.forEach(d => { if (d.missed > xMax) return; const dd = (sx(d.missed) - mx) ** 2 + (sy(d.avoided) - my) ** 2; if (dd < bd) { bd = dd; best = d; } });
    if (!best || bd > 900) { tip.classList.remove('on'); return; }
    tip.innerHTML = `<b>${best.src === 'model' ? 'CardioAI' : 'ESC 2019'}</b> · threshold ${pct(best.t)}<br>avoided ${best.avoided.toFixed(1)} · missed ${best.missed.toFixed(1)} per 100<br>sensitivity ${(best.sens * 100).toFixed(1)}%`;
    tip.style.left = sx(best.missed) / k + 'px'; tip.style.top = sy(best.avoided) / k + 'px'; tip.classList.add('on');
  });
  hit.addEventListener('pointerleave', () => tip.classList.remove('on'));
  const slider = $('#thr'), out = $('#thr-out');
  const nearest = (s, t) => pts(s).reduce((a, b) => Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a);
  const update = () => {
    const t = +slider.value, d = nearest('model', t);
    out.textContent = pct(t);
    markers[0].setAttribute('cx', sx(d.missed)); markers[0].setAttribute('cy', sy(d.avoided));
    const e = pts('esc').filter(p => p.sens >= d.sens - 1e-9).sort((a, b) => b.avoided - a.avoided)[0] || pts('esc')[0];
    markers[1].setAttribute('cx', sx(e.missed)); markers[1].setAttribute('cy', sy(e.avoided));
    $('#thr-readout').innerHTML = [
      [`${(d.sens * 100).toFixed(1)}%`, 'sensitivity'], [`${(d.spec * 100).toFixed(1)}%`, 'specificity'],
      [d.avoided.toFixed(1), 'normal angiograms avoided / 100'], [d.missed.toFixed(1), 'CAD missed / 100'],
    ].map(([b, s]) => `<div><b>${b}</b><small>${s}</small></div>`).join('') +
      `<div style="grid-column:1/-1"><small>ESC 2019 table at the same sensitivity avoids <b style="display:inline;font-size:inherit">${e.avoided.toFixed(1)}</b> per 100 (grey marker).</small></div>`;
  };
  slider.value = FULL.rule_out_threshold; slider.oninput = update; update();
}

function dotChart(host, rows, { lo = 0.4, hi = 1, ring = null, label = 'AUROC' } = {}) {
  host.innerHTML = '';
  const Wd = 520, rh = 34, m = { l: 200, r: 64, t: 10, b: 34 }, Ht = m.t + m.b + rows.length * rh;
  const svg = el('svg', { viewBox: `0 0 ${Wd} ${Ht}`, role: 'img', 'aria-label': `${label} by group` }, host);
  const sx = v => m.l + (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * (Wd - m.l - m.r);
  const g = el('g', { class: 'grid' }, svg);
  for (let v = lo; v <= hi + 1e-9; v += 0.1) { el('line', { x1: sx(v), x2: sx(v), y1: m.t, y2: Ht - m.b }, g); el('text', { x: sx(v), y: Ht - m.b + 16, 'text-anchor': 'middle' }, svg).textContent = v.toFixed(1); }
  if (lo <= 0.5) el('line', { x1: sx(0.5), x2: sx(0.5), y1: m.t, y2: Ht - m.b, stroke: 'rgba(255,255,255,.3)', 'stroke-dasharray': '3 3' }, svg);
  el('text', { x: (m.l + Wd - m.r) / 2, y: Ht - 4, 'text-anchor': 'middle' }, svg).textContent = label + ' (0.5 = chance)';
  const tip = tooltip(host);
  rows.forEach((r, i) => {
    const y = m.t + i * rh + rh / 2;
    el('text', { x: m.l - 10, y: y + 4, 'text-anchor': 'end', class: 'lbl' }, svg).textContent = r.name;
    if (r.ci) el('line', { x1: sx(r.ci[0]), x2: sx(r.ci[1]), y1: y, y2: y, stroke: 'var(--s-model)', 'stroke-width': 2, 'stroke-linecap': 'round', opacity: 0.55 }, svg);
    if (ring && r.ring != null) el('circle', { cx: sx(r.ring), cy: y, r: 5, fill: 'none', stroke: 'var(--s-esc)', 'stroke-width': 2 }, svg);
    el('circle', { cx: sx(r.v), cy: y, r: 5.5, fill: 'var(--s-model)', stroke: '#0b0b12', 'stroke-width': 2 }, svg);
    el('text', { x: Wd - m.r + 8, y: y + 4 }, svg).textContent = r.v.toFixed(3);
    const hit = el('rect', { x: 0, y: y - rh / 2, width: Wd, height: rh, fill: 'transparent' }, svg);
    hit.addEventListener('pointermove', () => {
      const b = svg.getBoundingClientRect(), k = Wd / b.width;
      tip.innerHTML = r.tip; tip.style.left = sx(r.v) / k + 'px'; tip.style.top = (y - 6) / k + 'px'; tip.classList.add('on');
    });
    hit.addEventListener('pointerleave', () => tip.classList.remove('on'));
  });
}

function renderEvidence() {
  const Mx = Object.fromEntries(EV.models.map(m => [m.key, m])), dep = Mx[EV.config.deployed], escM = Mx.esc;
  const ci = a => `${a[0].toFixed(2)}–${a[1].toFixed(2)}`;
  const ro = dep.rule_out, X = EV.external, sites = X.per_site.filter(r => !r.site.startsWith('All')), pooled = X.per_site.find(r => r.site.startsWith('All'));
  $('#tiles').innerHTML = [
    [dep.tasks.CAD.auroc.toFixed(3), 'AUROC, full model (internal)', `95% CI ${ci(dep.tasks.CAD.auroc_ci)}; ESC table ${escM.tasks.CAD.auroc.toFixed(3)}`],
    [pooled.models.bedside_lr.auroc.toFixed(3), 'AUROC, bedside model in unseen hospitals', `pooled over 5 hospitals; ESC table ${pooled.models.esc.auroc.toFixed(3)}`],
    [ro.normal_caths_avoided_per_100.toFixed(1), 'normal angiograms avoided / 100 (full)', `ESC table: ${escM.rule_out.normal_caths_avoided_per_100.toFixed(1)}`],
    [ro.cad_missed_per_100.toFixed(1), 'CAD patients missed / 100 (full)', `sensitivity ${(ro.sensitivity * 100).toFixed(1)}%, NPV ${ro.npv.toFixed(2)}`],
  ].map(([b, s, sm]) => `<div class="tile glass"><b class="num">${b}</b><span>${s}</span><small>${sm}</small></div>`).join('');

  dotChart($('#external-chart'), X.per_site.map(r => ({ name: `${r.site} (n=${r.n}, ${(r.prevalence * 100).toFixed(0)}% CAD)`, v: r.models.bedside_lr.auroc, ci: r.models.bedside_lr.auroc_ci, ring: r.models.esc.auroc,
    tip: `<b>${esc(r.site)}</b><br>bedside ${r.models.bedside_lr.auroc.toFixed(3)} (${ci(r.models.bedside_lr.auroc_ci)})<br>ESC table ${r.models.esc.auroc.toFixed(3)}<br>missed ${r.models.bedside_lr.cad_missed_per_100.toFixed(1)} / avoided ${r.models.bedside_lr.normal_caths_avoided_per_100.toFixed(1)} per 100` })), { lo: 0.4, ring: true });
  $('#tbl-external').innerHTML = `<thead><tr><th>Held-out hospital</th><th>n</th><th>CAD</th><th>Bedside AUROC</th><th>ESC AUROC</th><th>Δ (95% CI)</th><th>Cal. slope</th><th>Pred − obs</th><th>Sens.</th><th>Avoided /100</th><th>Missed /100</th></tr></thead><tbody>` +
    X.per_site.map(r => { const m = r.models.bedside_lr, e = r.models.esc, d = r.delta_lr_vs_esc;
      return `<tr class="${r.site.startsWith('All') ? 'is-deployed' : ''}"><td>${esc(r.site)}</td><td>${r.n}</td><td>${(r.prevalence * 100).toFixed(0)}%</td><td>${m.auroc.toFixed(3)} <small>(${ci(m.auroc_ci)})</small></td><td>${e.auroc.toFixed(3)}</td>
      <td>${d.delta >= 0 ? '+' : ''}${d.delta.toFixed(3)} <small>(${d.ci[0].toFixed(2)}, ${d.ci[1].toFixed(2)})</small></td><td>${m.cal_slope.toFixed(2)}</td><td>${m.cal_in_large >= 0 ? '+' : ''}${m.cal_in_large.toFixed(2)}</td>
      <td>${(m.sensitivity * 100).toFixed(0)}%</td><td>${m.normal_caths_avoided_per_100.toFixed(1)}</td><td>${m.cad_missed_per_100.toFixed(1)}</td></tr>`; }).join('') + '</tbody>';
  const teh = sites.find(r => r.site === 'Tehran'), zur = sites.find(r => r.site === 'Zurich');
  $('#external-note').innerHTML = `Pooled cross-validation, with hospitals mixed between training and test, gives AUROC ${X.pooled_cv_auroc.toFixed(3)}. Testing on an unseen hospital gives ${pooled.models.bedside_lr.auroc.toFixed(3)}. <b>That gap is what a new hospital should expect.</b>
    The model beats the guideline table in 4 of 5 hospitals. It fails in Tehran (AUROC ${teh.models.bedside_lr.auroc.toFixed(2)}, sensitivity ${(teh.models.bedside_lr.sensitivity * 100).toFixed(0)}%). There, "no chest pain" meant low risk, but at the four US/European sites asymptomatic patients had the most CAD, because they were referred for other reasons. Zurich has only ${Math.round(zur.n * (1 - zur.prevalence))} patients without CAD, so its estimates are very uncertain.`;
  const ks = [0, 25, 50, 100];
  $('#tbl-adapt').innerHTML = `<thead><tr><th>Hospital</th>${ks.map(k => `<th>+${k} local</th>`).join('')}</tr></thead><tbody>` +
    sites.map(r => `<tr><td>${esc(r.site)}</td>${ks.map(k => { const a = X.adaptation.find(a => a.site === r.site && a.k_local === k); return `<td>${a ? `${a.auroc_mean.toFixed(3)} <small>(${a.auroc_p10.toFixed(2)}–${a.auroc_p90.toFixed(2)})</small>` : '<small>n/a</small>'}</td>`; }).join('')}</tr>`).join('') + '</tbody>';

  const row = m => {
    const t = m.tasks.CAD, r = m.rule_out, d = m.delta_vs_esc;
    return `<tr class="${m.key === EV.config.deployed ? 'is-deployed' : ''}"><td>${esc(m.name)}<br><small>${esc(m.note)}</small></td>
      <td>${t.auroc.toFixed(3)} <small>(${ci(t.auroc_ci)})</small></td><td>${d ? (d.delta >= 0 ? '+' : '') + d.delta.toFixed(3) + ` <small>(${d.ci[0].toFixed(2)}, ${d.ci[1].toFixed(2)})</small>` : '—'}</td>
      <td>${(r.sensitivity * 100).toFixed(1)}%</td><td>${(r.specificity * 100).toFixed(1)}%</td><td>${r.npv.toFixed(2)}</td>
      <td>${r.normal_caths_avoided_per_100.toFixed(1)}</td><td>${r.cad_missed_per_100.toFixed(1)}</td><td>${t.brier.toFixed(3)}</td></tr>`;
  };
  $('#tbl-models').innerHTML = `<thead><tr><th>Model</th><th>AUROC (95% CI)</th><th>Δ vs ESC</th><th>Sens.</th><th>Spec.</th><th>NPV</th><th>Avoided /100</th><th>Missed /100</th><th>Brier</th></tr></thead><tbody>${EV.models.map(row).join('')}</tbody>`;
  $('#tbl-vessels').innerHTML = `<thead><tr><th>Model</th><th>LAD</th><th>LCX</th><th>RCA</th></tr></thead><tbody>` +
    EV.models.filter(m => m.tasks.LAD).map(m => `<tr class="${m.key === EV.config.deployed ? 'is-deployed' : ''}"><td>${esc(m.name)}</td>` +
      VESSELS.map(v => `<td>${m.tasks[v].auroc.toFixed(3)} <small>(${ci(m.tasks[v].auroc_ci)})</small></td>`).join('') + '</tr>').join('') + '</tbody>';
  dotChart($('#ablation-chart'), EV.ablation.map(a => ({ name: a.set.replace(' (history + symptoms + exam)', ''), v: a.auroc, ci: a.auroc_ci,
    tip: `<b>${esc(a.set)}</b><br>${a.n_features} inputs · AUROC ${a.auroc.toFixed(3)} (${ci(a.auroc_ci)})<br>avoided ${a.normal_caths_avoided_per_100.toFixed(1)} · missed ${a.cad_missed_per_100.toFixed(1)} per 100` })), { lo: 0.5 });
  dotChart($('#subgroup-chart'), EV.subgroups.map(s => ({ name: `${s.group.replace(' (RWMA = 0)', '')} (n=${s.n})`, v: s.model_auroc, ci: s.model_auroc_ci, ring: s.esc_auroc,
    tip: `<b>${esc(s.group)}</b> · n=${s.n}, CAD ${(s.prevalence * 100).toFixed(0)}%<br>CardioAI ${s.model_auroc.toFixed(3)} (${ci(s.model_auroc_ci)})<br>ESC table ${s.esc_auroc.toFixed(3)}` })), { lo: 0.5, ring: true });
  const reps = dep.tasks.CAD.auroc_per_repeat;
  $('#checks').innerHTML = [
    ['Negative control', `With the labels shuffled, the same pipeline scores AUROC ${EV.negative_control.auroc.toFixed(3)} (chance is 0.5). There is no leakage path.`],
    ['Stable across repeats', `Full-model CAD AUROC per CV repeat: ${reps.map(r => r.toFixed(3)).join(', ')}.`],
    ['No target leakage', 'Angiography results are never inputs. Scaling, imputation, feature selection, C and the rule-out threshold are all fit on training data only (training folds, or training hospitals).'],
    ['Reproducible', `Seed ${EV.config.seed}, ${EV.config.repeats}×${EV.config.folds}-fold, bootstrap B=${EV.config.bootstrap}. Python ${EV.config.python}, scikit-learn ${EV.config.sklearn}. Full run took ${Math.round(EV.config.runtime_s)} s.`],
  ].map(([b, s]) => `<li><b>${b}</b><span class="muted">${s}</span></li>`).join('');
  tradeoffChart();
  $('#nav-status-text').textContent = `Full AUROC ${dep.tasks.CAD.auroc.toFixed(2)} (internal) · Bedside ${pooled.models.bedside_lr.auroc.toFixed(2)} (unseen hospitals)`;
  $('#vessel-auc').textContent = VESSELS.map(v => `${v} ${dep.tasks[v].auroc.toFixed(2)}`).join(', ');
  const la = sites.map(r => r.models.bedside_lr.auroc);
  const fill = { missed: ro.cad_missed_per_100.toFixed(1), npv: ro.npv.toFixed(2), 'loso-range': `${Math.min(...la).toFixed(2)}–${Math.max(...la).toFixed(2)}` };
  $$('[data-fill]').forEach(s => { if (fill[s.dataset.fill]) s.textContent = fill[s.dataset.fill]; });
}

/* ================================================================ tabs, disclaimer, boot */
function showTab(name) {
  if (!['patient', 'evidence', 'about'].includes(name)) name = 'patient';
  $$('[data-tab]').forEach(b => b.setAttribute('aria-selected', String(b.dataset.tab === name)));
  $$('.view').forEach(v => { v.hidden = v.id !== 'view-' + name; });
  if (location.hash !== '#' + name) history.replaceState(null, '', '#' + name);
  if (name === 'evidence' && EV) tradeoffChart();
  window.scrollTo(0, 0);
}
function initDisclaimer() {
  const dlg = document.createElement('dialog'); dlg.className = 'modal';
  dlg.innerHTML = `<div class="modal__body glass"><h2>Research prototype: not for clinical use</h2>
    <p>CardioAI's full model was built and evaluated on 303 patients from one hospital. The bedside model was tested in 5 hospitals, and its accuracy varied a lot between them. Neither is a medical device.</p>
    <p>In validation, about 1 in 5 patients labelled "low risk" still had CAD. It must never be the reason a patient does not get the test they need.</p>
    <form method="dialog"><button class="chip" autofocus>I understand</button></form></div>`;
  document.body.appendChild(dlg);
  $('#disclaimer').addEventListener('click', () => dlg.showModal());
  dlg.addEventListener('click', e => { if (e.target === dlg) dlg.close(); });
  $('#batch').addEventListener('click', e => { if (e.target === $('#batch')) $('#batch').close(); });
}

async function boot() {
  liquidGlass('.nav, .risk-chip, .dock, .legend, .disclaimer');
  initDisclaimer();
  $$('[data-tab]').forEach(b => b.addEventListener('click', () => showTab(b.dataset.tab)));
  $('#nav-status').addEventListener('click', () => showTab('evidence'));
  addEventListener('hashchange', () => showTab(location.hash.slice(1)));
  showTab(location.hash.slice(1) || 'patient');
  try {
    [FULL, BED, DATA, EV] = await Promise.all(['model', 'bedside', 'patients', 'evidence'].map(n => fetch(`./data/${n}.json`).then(r => { if (!r.ok) throw new Error(n); return r.json(); })));
  } catch (e) {
    $('#res-decision').textContent = 'Data not found';
    $('#res-source').textContent = 'Run python evaluate.py and python export_model.py, then serve the dashboard folder over http (python start_dashboard.py).';
    return;
  }
  initECG();
  try { heart = initHeart({ host: $('#scene'), labelsHost: $('#vessel-labels'), reduced, getHR: () => ecg.hr }); }
  catch (e) { console.warn('WebGL unavailable', e); $('#webgl-fallback').hidden = false; }
  $('#patient-select').addEventListener('change', e => selectPatient(+e.target.value));
  $('#reset').addEventListener('click', () => { S = { ...S0 }; $$('.field').forEach(el => { el.classList.remove('is-invalid'); const e = $('.err', el); if (e) e.hidden = true; }); syncAll(); render(); });
  $$('[data-mode]').forEach(b => b.addEventListener('click', () => b.dataset.mode !== mode && setMode(b.dataset.mode)));
  $$('.vcard').forEach(c => c.addEventListener('click', () => { const n = c.dataset.vessel; selected = selected === n ? null : n; render(); if (selected && heart) heart.focus(n); }));
  $$('[data-view]').forEach(b => b.addEventListener('click', () => heart && heart.view(b.dataset.view)));
  $$('[data-layer]').forEach(b => b.addEventListener('click', () => { const on = b.getAttribute('aria-pressed') !== 'true'; b.setAttribute('aria-pressed', String(on)); heart && heart.layer(b.dataset.layer, on); }));
  $('#btn-print').addEventListener('click', printReport);
  initBatch();
  renderEvidence();
  liquidGlass('.tile', { scale: 30, bezel: 14 });
  setMode('full');
}
boot();
