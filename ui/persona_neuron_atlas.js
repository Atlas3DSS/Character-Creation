/* Neuron Atlas: one dot per MLP neuron, a searchable library and a per-neuron drawer. */
const $ = id => document.getElementById(id);
const fmt = n => (n ?? 0).toLocaleString('en-US');
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const tok = s => esc(JSON.stringify(String(s ?? '')).slice(1, -1));
const COLORS = {
  up: { persona: '#a6d860', word: '#efc77c', piece: '#7d8f99', code: '#6cb6ff', number: '#c79bf2', format: '#4d575e', script: '#de90b4', mixed: '#343c41' },
  activation: { identity: '#a6d860', feelings: '#de90b4', relationships: '#efc77c', conflict: '#ff8a65', values: '#f2d45c', facts: '#6cb6ff', everyday: '#4fd1c5', stories: '#c79bf2', other: '#7d8f99', broad: '#30373c' },
  evidence: { labelled: '#a6d860', recipe: '#efc77c', association: '#6cb6ff', trait: '#de90b4', none: '#262c30' } };
COLORS.down = COLORS.up;
const MODES = ['up', 'down', 'activation', 'evidence'];
const S = { sum: null, codes: null, cells: null, inverse: [], mode: 'up', off: new Set(), labels: new Map(), barCat: 'persona',
  lib: { offset: 0, rows: [], total: 0 }, list: [], at: -1, hero: 0, concepts: null, concept: null };
function toast(t) { const el = $('toast'); el.textContent = t; el.hidden = false; clearTimeout(toast.t); toast.t = setTimeout(() => el.hidden = true, 1600); }
async function getJSON(url) { const r = await fetch(url); const v = await r.json(); if (!r.ok) throw new Error(v.error || r.statusText); return v; }
const W = () => S.sum.neurons, N = () => S.sum.total;
const codeName = (mode, code) => S.sum.modes[mode].codes[code];
const catLabel = (mode, c) => S.sum.modes[mode].label[c] || c;
const color = (mode, c) => COLORS[mode][c] || '#555';
const ink = (mode, c) => ['broad', 'none', 'mixed', 'format'].includes(c) ? 'var(--dim)' : color(mode, c);  // readable heading text

async function init() {
  S.sum = await getJSON('/api/neuron-atlas');
  const buf = new Uint8Array(await (await fetch('/neuron-atlas/atlas.bin')).arrayBuffer());
  const n = S.sum.total; S.codes = MODES.map((_, i) => buf.subarray(i * n, (i + 1) * n));
  S.cells = new Uint16Array(buf.buffer.slice(4 * n, 4 * n + 2 * n));
  for (let L = 0; L < S.sum.layers; L++) { const inv = new Int32Array(S.sum.side * S.sum.side).fill(-1);
    for (let k = 0; k < W(); k++) inv[S.cells[L * W() + k]] = k; S.inverse.push(inv); }
  $('h-total').textContent = fmt(n);
  $('h-captures').textContent = `Built from the checkpoint lens and ${fmt(S.sum.captures)} unsteered all-neuron captures over ${fmt(S.sum.prompts)} distinct prompts.`;
  $('m-sel').textContent = S.sum.selectivity_min; $('m-min').textContent = S.sum.domain_min_prompts;
  $('m-setup').innerHTML = [['Model', `${esc(S.sum.model)}, ${S.sum.layers} layers × ${fmt(W())} MLP neurons`], ['Neurons', fmt(n)],
    ['Lens', 'Top 24 tokens per direction; colour from the top 12'], ['Captures', `${fmt(S.sum.captures)} unsteered final-prefill planes, ${fmt(S.sum.prompts)} prompts`],
    ['Prompt types', Object.entries(S.sum.domain_prompts).map(([d, c]) => `${esc(catLabel('activation', d))} ${fmt(c)}`).join(' · ')],
    ['Evidence', Object.entries(S.sum.evidence_counts).map(([d, c]) => `${esc(catLabel('evidence', d))} ${fmt(c)}`).join(' · ')],
    ['Built', new Date(S.sum.built_at * 1000).toLocaleString()]].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  buildGrid(); setMode('up'); hero(); fillCats(); library(true); openFromHash(); sweepStatus(); loadConcepts();
}

/* hero: saved "who are you" answers from multi-neuron recipes, next to their matched baseline */
function hero() {
  const rows = S.sum.hero || [];
  const show = () => { if (!rows.length) { $('h-slide').innerHTML = '<p class="dim">No saved identity comparisons yet.</p>'; return; }
    const r = rows[S.hero]; $('h-pos').textContent = `${S.hero + 1} / ${rows.length}`;
    $('h-slide').innerHTML = `<p class="sender">Qwen3.8-27B · ${esc(r.target.replace(/^experimental-/, ''))} recipe ${r.arm === 'reversal' ? 'reversed' : 'applied'} · ${r.pushes} pushes · seed ${esc(r.seed)}</p>
      <p class="bubble">${esc(r.answer)}</p><p class="sender">Same seed, no pushes</p><p class="bubble base">${esc(r.baseline)}</p>`; };
  $('h-prev').onclick = () => { S.hero = (S.hero - 1 + rows.length) % rows.length; show(); };
  $('h-next').onclick = () => { S.hero = (S.hero + 1) % rows.length; show(); };
  show();
}

function buildGrid() {
  const grid = $('grid'); grid.innerHTML = '';
  for (let L = 0; L < S.sum.layers; L++) {
    const tile = document.createElement('div'); tile.className = 'tile'; tile.dataset.layer = L;
    const c = document.createElement('canvas'); c.width = c.height = S.sum.side; c.setAttribute('aria-label', `Layer ${L}`);
    tile.append(c); tile.insertAdjacentHTML('beforeend', `<span>L${L}</span>`); grid.append(tile);
    tile.addEventListener('mousemove', e => hover(e, L, c)); tile.addEventListener('mouseleave', () => $('tip').hidden = true);
    tile.addEventListener('click', e => { const k = pick(e, L, c); if (k >= 0) openNeuron(L, k, null); });
  }
}
function pick(e, L, c) { const r = c.getBoundingClientRect(); const x = Math.floor((e.clientX - r.left) / r.width * S.sum.side), y = Math.floor((e.clientY - r.top) / r.height * S.sum.side);
  if (x < 0 || y < 0 || x >= S.sum.side || y >= S.sum.side) return -1; return S.inverse[L][x + y * S.sum.side]; }
async function layerLabels(L) { if (!S.labels.has(L)) S.labels.set(L, getJSON(`/api/neuron-atlas/layer?layer=${L}`)); return S.labels.get(L); }
async function hover(e, L, c) {
  const k = pick(e, L, c), tip = $('tip'); if (k < 0) { tip.hidden = true; return; }
  const code = codeName(S.mode, S.codes[MODES.indexOf(S.mode)][L * W() + k]);
  tip.style.left = Math.min(e.clientX + 14, innerWidth - 290) + 'px'; tip.style.top = (e.clientY + 14) + 'px'; tip.hidden = false;
  tip.innerHTML = `<b>L${L} N${k}</b> · ${esc(catLabel(S.mode, code))}`;
  const labels = await layerLabels(L).catch(() => null); if (!labels || tip.hidden) return;
  tip.innerHTML = `<b>L${L} N${k}</b> · ${esc(catLabel(S.mode, code))}<br>↑ ${tok(labels.up[k])} &nbsp; ↓ ${tok(labels.down[k])}`;
}
function hexRGB(h) { const v = parseInt(h.slice(1), 16); return [v >> 16, (v >> 8) & 255, v & 255]; }
function paint() {
  const i = MODES.indexOf(S.mode), codes = S.codes[i], names = S.sum.modes[S.mode].codes;
  const rgb = names.map(c => hexRGB(color(S.mode, c))), on = names.map(c => !S.off.has(c)), side = S.sum.side;
  document.querySelectorAll('.tile canvas').forEach((c, L) => {
    const ctx = c.getContext('2d'), img = ctx.createImageData(side, side), px = img.data;
    for (let k = 0; k < W(); k++) { const code = codes[L * W() + k], p = S.cells[L * W() + k] * 4, [r, g, b] = rgb[code];
      if (on[code]) { px[p] = r; px[p + 1] = g; px[p + 2] = b; px[p + 3] = 255; } else { px[p] = px[p + 1] = px[p + 2] = 34; px[p + 3] = 255; } }
    if (S.concept) concept_overlay(L, px, side);
    ctx.putImageData(img, 0, 0); });
}
/* concept trace: dim the map, light the concept's top neurons per layer (2x2 so a single dot is visible) */
function concept_overlay(L, px, side) {
  for (let p = 0; p < px.length; p += 4) { px[p] *= .22; px[p + 1] *= .22; px[p + 2] *= .22; }
  const lime = hexRGB('#a6d860'), rose = hexRGB('#de90b4'), gold = hexRGB('#efc77c'), q = S.concept.qualified;
  const dot = (k, rgb, size) => { const cell = S.cells[L * W() + k], x = cell % side, y = Math.floor(cell / side);
    for (let dy = 0; dy < size; dy++) for (let dx = 0; dx < size; dx++) { const xx = x + dx, yy = y + dy; if (xx >= side || yy >= side) continue;
      const p = (xx + yy * side) * 4; px[p] = rgb[0]; px[p + 1] = rgb[1]; px[p + 2] = rgb[2]; } };
  for (const [k, d, h] of S.concept.layers[L].neurons) {
    const ok = Math.sign(d) === Math.sign(h) && Math.abs(h) >= S.concepts.replicates, base = d > 0 ? lime : rose;
    dot(k, ok ? base : base.map(v => v * .45), 2); }
  if (q && q[0] === L) dot(q[1], gold, 3);
}
function setMode(mode) {
  S.mode = mode; S.off.clear();
  document.querySelectorAll('.modes button').forEach(b => b.setAttribute('aria-selected', b.dataset.mode === mode));
  const m = S.sum.modes[mode], totals = m.codes.map((_, c) => S.sum.per_layer[mode].reduce((a, row) => a + row[c], 0));
  $('legend').innerHTML = m.codes.map((c, j) => `<button type="button" data-cat="${c}" aria-pressed="true" title="${esc((m.note || {})[c] || '')}"><i style="background:${color(mode, c)}"></i>${esc(m.label[c])} <small>${fmt(totals[j])}</small></button>`).join('');
  $('legend').querySelectorAll('button').forEach(b => b.onclick = () => {
    const c = b.dataset.cat; const solo = S.off.size === 0 || S.off.has(c);
    if (solo) { S.off = new Set(m.codes.filter(x => x !== c)); } else S.off.clear();
    $('legend').querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', !S.off.has(x.dataset.cat)));
    S.barCat = S.off.size ? c : S.barCat; paint(); bylayer(); });
  $('mode-note').textContent = { up: 'What each neuron writes into the vocabulary when pushed up: its top tokens under the static lens.',
    down: 'The same lens for the negated direction, i.e. pushing the neuron down.',
    activation: `Which kind of prompt raises this neuron most, among ${fmt(S.sum.prompts)} unsteered prompts (standardized difference ≥ ${S.sum.selectivity_min}).`,
    evidence: 'What we hold beyond the lens: concept labels, recipe use, association-search hits and Big Five candidates.' }[mode];
  S.barCat = { up: 'persona', down: 'persona', activation: 'feelings', evidence: 'labelled' }[mode];
  paint(); bylayer();
}
document.querySelectorAll('.modes button').forEach(b => b.onclick = () => setMode(b.dataset.mode));
function bylayer() {
  if (S.concept) return conceptBars();
  $('bylayer-head').textContent = 'By layer';
  const m = S.sum.modes[S.mode], c = m.codes.indexOf(S.barCat), rows = S.sum.per_layer[S.mode];
  const share = rows.map(r => r[c] / r.reduce((a, b) => a + b, 0)), top = Math.max(...share, 1e-9);
  $('bylayer-title').textContent = `Share of each layer’s neurons in “${m.label[S.barCat]}” (click a legend entry to change).`;
  $('bylayer').innerHTML = share.map((v, L) => `<div style="height:${(v / top * 100).toFixed(1)}%;background:${color(S.mode, S.barCat)}" title="Layer ${L}: ${(v * 100).toFixed(2)}% (${fmt(rows[L][c])})"></div>`).join('');
}

/* library */
function fillCats() { const mode = $('l-mode').value, m = S.sum.modes[mode];
  $('l-cat').innerHTML = '<option value="">Any</option>' + m.codes.map(c => `<option value="${c}">${esc(m.label[c])}</option>`).join(''); }
function query() { const p = new URLSearchParams({ q: $('q').value, mode: $('l-mode').value, category: $('l-cat').value,
  lmin: $('l-min').value || 0, lmax: $('l-max').value || 63, evidence: $('l-evidence').checked ? '1' : '', sort: $('l-sort').value }); return p; }
async function library(reset) {
  if (reset) { S.lib = { offset: 0, rows: [], total: 0 }; $('cards').innerHTML = ''; }
  const p = query(); p.set('offset', S.lib.offset); p.set('limit', 48);
  $('l-csv').href = '/api/neuron-atlas/library.csv?' + query();
  let r; try { r = await getJSON('/api/neuron-atlas/library?' + p); } catch (e) { $('l-count').textContent = e.message; return; }
  S.lib.total = r.total; S.lib.offset += r.rows.length; S.lib.rows.push(...r.rows);
  $('l-count').textContent = `${fmt(r.total)} neurons match.`;
  $('cards').insertAdjacentHTML('beforeend', r.rows.map((row, j) => card(row, S.lib.rows.length - r.rows.length + j)).join(''));
  $('more').hidden = S.lib.offset >= r.total;
}
function card(r, idx) {
  return `<button type="button" class="card" data-idx="${idx}"><span class="id">L${r.layer} N${r.neuron}</span>
    <div class="dirs"><em>↑</em><span>${tok(r.up)}</span><em>↓</em><span>${tok(r.down)}</span></div>
    <span class="badge"><i style="background:${color('up', r.up_kind)}"></i>${esc(catLabel('up', r.up_kind))}</span>
    <span class="badge"><i style="background:${color('activation', r.domain)}"></i>${esc(catLabel('activation', r.domain))}</span>
    ${r.evidence !== 'none' ? `<span class="badge"><i style="background:${color('evidence', r.evidence)}"></i>${esc(catLabel('evidence', r.evidence))}</span>` : ''}</button>`;
}
$('cards').addEventListener('click', e => { const c = e.target.closest('.card'); if (!c) return; const r = S.lib.rows[+c.dataset.idx];
  openNeuron(r.layer, r.neuron, { rows: S.lib.rows, at: +c.dataset.idx }); });
let debounce; for (const id of ['q', 'l-cat', 'l-min', 'l-max', 'l-sort', 'l-evidence'])
  $(id).addEventListener(id === 'q' ? 'input' : 'change', () => { clearTimeout(debounce); debounce = setTimeout(() => library(true), 250); });
$('l-mode').onchange = () => { fillCats(); library(true); };
$('more').onclick = () => library(false);
async function sweepStatus() {
  try { const r = await getJSON('/api/neuron-atlas/sweep'), c = (r.status || {}).counts || {}, recent = (r.status || {}).last_10_min || {};
    $('sweep-status').textContent = r.status ? `Identity sweep: ${fmt(c.done || 0)} answers done, ${fmt(c.queued || 0)} queued, ${fmt(c.running || 0)} running${c.uncertain ? `, ${fmt(c.uncertain)} uncertain` : ''}. ${fmt(r.neurons_answered)} neurons have answers. Last 10 min: ${fmt(recent.answers)} answers, ${fmt(recent.tokens)} tokens.` : 'Identity sweep: no results synced yet.';
  } catch (e) { $('sweep-status').textContent = 'Identity sweep status unavailable: ' + e.message; }
}
$('sweep').onclick = async () => {
  const name = prompt('Name this sweep batch', ($('q').value || $('l-cat').value || 'batch').slice(0, 40)); if (!name) return;
  $('sweep').disabled = true;
  try { const p = Object.fromEntries(query()); const r = await fetch('/api/neuron-atlas/sweep', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...p, name }) });
    const v = await r.json(); if (!r.ok) throw new Error(v.error); toast(`Queued ${fmt(v.queued_neurons)} neurons (${fmt(v.added)} new answers)`); sweepStatus();
  } catch (e) { toast(e.message); } finally { $('sweep').disabled = false; }
};
setInterval(sweepStatus, 60000);
$('open').onsubmit = e => { e.preventDefault(); openNeuron(+$('o-layer').value, +$('o-neuron').value, null); };

/* neuron drawer */
async function openNeuron(L, n, list) {
  if (list) { S.list = list.rows; S.at = list.at; } else { S.list = []; S.at = -1; }
  $('d-prev').hidden = $('d-next').hidden = !S.list.length;
  history.replaceState(null, '', `#n=${L}:${n}`);
  $('d-title').textContent = `Layer ${L} · neuron ${n}`; $('d-body').innerHTML = '<p class="dim">Loading…</p>'; $('drawer').hidden = false;
  const [a, lens, web] = await Promise.all([getJSON(`/api/neuron-atlas/neuron?layer=${L}&neuron=${n}`),
    getJSON(`/api/lens?layer=${L}&neuron=${n}`).catch(() => null), getJSON(`/api/weight-web?layer=${L}&neuron=${n}&hops=1&fanout=8`).catch(() => null)]);
  const chips = rows => (rows || []).slice(0, 24).map(t => `<span class="chip" title="logit ${t.logit.toFixed(2)}">${tok(t.text)}</span>`).join('');
  const dir = (side, d, rows) => `<div class="box"><h3>Pushed ${side} · <span style="color:${ink('up', d.category)}">${esc(catLabel('up', d.category))}</span></h3>
    <p>Label <b>${tok(d.label)}</b> <span class="dim">· top logit ${d.peak.toFixed(2)}</span></p><div>${chips(rows)}</div></div>`;
  const eff = Object.entries(a.activation.effects).filter(([, v]) => v !== null), top = Math.max(1, ...eff.map(([, v]) => Math.abs(v)));
  const bars = eff.map(([d, v]) => `<span>${esc(catLabel('activation', d))}</span><span class="bar"><b style="background:${color('activation', d)};${v >= 0 ? 'left:50%' : `right:50%`};width:${(Math.abs(v) / top * 50).toFixed(1)}%"></b></span><span>${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}</span>`).join('');
  const ev = a.evidence || { kinds: [] };
  const edges = web && web.graph ? web.graph.nodes.filter(x => !x.selected) : [];
  const near = side => edges.filter(x => side < 0 ? x.layer < L : x.layer > L).map(x => `<button type="button" data-l="${x.layer}" data-n="${x.neuron}">L${x.layer} N${x.neuron}</button>`).join('') || '<span class="dim">None retained.</span>';
  const who = a.identity || { answers: [] };
  const whoBox = who.answers.length ? `<div class="box" style="margin-bottom:12px"><h3>“who are you” with this neuron pushed</h3><div class="who">
      ${who.answers.map(r => `<div class="ans" style="border-left-color:${r.dose > 0 ? 'var(--lime)' : 'var(--rose)'}"><b>Pushed ${r.dose > 0 ? 'up' : 'down'} · ${r.kept ? `strength ${Math.abs(r.dose)}×` : 'incoherent at every strength'}${r.collapsed.length && r.kept ? ` (collapsed at ${r.collapsed.map(x => x + '×').join(', ')})` : ''} · ${r.tokens} tokens</b>${esc(r.answer) || '<i>(empty)</i>'}</div>`).join('')}
      ${who.baseline ? `<div class="ans base"><b>No push</b>${esc(who.baseline.answer)}</div>` : ''}</div></div>`
    : `<div class="box" style="margin-bottom:12px"><h3>“who are you”</h3><p class="dim">Not swept yet. Use “Ask who are you” in the Library to queue it.</p></div>`;
  $('d-body').innerHTML = whoBox + `
    <div class="dir-grid">${dir('up', a.up, lens && lens.positive)}${dir('down', a.down, lens && lens.negative)}</div>
    <div class="box" style="margin-top:12px"><h3>What activates it · <span style="color:${ink('activation', a.activation.domain)}">${esc(catLabel('activation', a.activation.domain))}</span></h3>
      <p class="dim">Final-prefill activation: mean ${a.activation.mean.toFixed(4)}, SD ${a.activation.sd.toFixed(4)}. Standardized difference of each prompt type against all other prompts:</p>
      <div class="bars">${bars || '<span class="dim">No prompt type has enough captures.</span>'}</div>
      <h3>Prompts that raise it most</h3><ol class="prompts">${a.activation.top_prompts.map(p => `<li>${esc(p.text)} <span class="dim">· ${esc(p.category || 'uncategorized')}</span></li>`).join('')}</ol></div>
    <div class="box" style="margin-top:12px"><h3>Our evidence</h3>
      ${ev.kinds.length ? ev.kinds.map(k => `<span class="badge"><i style="background:${color('evidence', k)}"></i>${esc(catLabel('evidence', k))}</span>`).join('') : '<p class="dim">Lens and activation only. No label, recipe or search result names this neuron yet.</p>'}
      ${(ev.labels || []).length ? `<p>Concept labels: ${ev.labels.map(l => `<b>${esc(l.label)}</b> <span class="dim">(${esc(l.direction)}, contrast ${(+l.score).toFixed(2)})</span>`).join(', ')}</p>` : ''}
      ${(ev.recipes || []).length ? `<p>Recipe pushes: ${ev.recipes.map(esc).join(' · ')}</p>` : ''}
      ${ev.traits ? `<p>Big Five alignment: ${Object.entries(ev.traits).map(([t, v]) => `${esc(t)} ${(v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2)}`).join(' · ')}</p>` : ''}</div>
    <div class="box" style="margin-top:12px"><h3>Strongest weight neighbours</h3><p class="dim">Signed cosine of output and read directions in adjacent layers. Affinity, not proven transmission.</p>
      <p>Reads from: </p><div class="edges">${near(-1)}</div><p>Writes to: </p><div class="edges">${near(1)}</div></div>
    <div class="box" style="margin-top:12px"><h3>Try it</h3><p class="dim">Lyra-style notation for a recipe in Explore, at a moderate dose:</p>
      <div class="recipe-line"><span>L${L}N${n}+0.5</span><button type="button" id="d-recipe">Copy</button></div></div>`;
  $('d-body').querySelectorAll('.edges button').forEach(b => b.onclick = () => openNeuron(+b.dataset.l, +b.dataset.n, null));
  $('d-recipe').onclick = () => navigator.clipboard.writeText(`L${L}N${n}+0.5`).then(() => toast('Copied'));
}
function closeDrawer() { $('drawer').hidden = true; history.replaceState(null, '', location.pathname + '#library'); }
$('d-close').onclick = closeDrawer;
$('drawer').addEventListener('click', e => { if (e.target === $('drawer')) closeDrawer(); });
const step = d => { if (!S.list.length) return; S.at = (S.at + d + S.list.length) % S.list.length; const r = S.list[S.at]; openNeuron(r.layer, r.neuron, { rows: S.list, at: S.at }); };
$('d-prev').onclick = () => step(-1); $('d-next').onclick = () => step(1);
$('d-copy').onclick = () => navigator.clipboard.writeText(location.href).then(() => toast('Link copied'));
addEventListener('keydown', e => { if ($('drawer').hidden) return; if (e.key === 'Escape') closeDrawer(); if (e.key === 'ArrowLeft') step(-1); if (e.key === 'ArrowRight') step(1); });
function openFromHash() { const m = location.hash.match(/^#n=(\d+):(\d+)$/); if (m) openNeuron(+m[1], +m[2], null); }

/* concepts */
async function loadConcepts() {
  try { S.concepts = await getJSON('/api/neuron-atlas/concepts'); }
  catch (e) { $('concepts').innerHTML = `<span class="dim">${esc(e.message)}</span>`; return; }
  const rows = [...S.concepts.concepts].sort((a, b) => (b.passed - a.passed) || a.name.localeCompare(b.name));
  $('concepts').innerHTML = rows.map(c => `<button type="button" data-concept="${esc(c.name)}" aria-pressed="false" class="${c.passed ? '' : 'unqualified'}"
    title="${c.passed ? 'Qualified label' : 'Traced, but no single neuron passed the held-out gate'} · peaks at layer ${c.peak_layer} (|d| ${c.peak.toFixed(2)})"><i style="background:#a6d860"></i>${esc(c.name)}</button>`).join('')
    + '<button type="button" id="concept-clear" hidden>Clear</button>';
  $('concepts').querySelectorAll('button[data-concept]').forEach(b => b.onclick = () => pickConcept(b.dataset.concept === (S.concept || {}).name ? null : b.dataset.concept));
  $('concept-clear').onclick = () => pickConcept(null);
  const m = decodeURIComponent(location.hash).match(/^#concept=(.+)$/); if (m) pickConcept(m[1]);
}
async function pickConcept(name) {
  S.concept = name ? await getJSON('/api/neuron-atlas/concept?name=' + encodeURIComponent(name)).catch(e => { toast(e.message); return null; }) : null;
  $('concepts').querySelectorAll('button[data-concept]').forEach(b => b.setAttribute('aria-pressed', b.dataset.concept === (S.concept || {}).name));
  $('concept-clear').hidden = !S.concept; $('concept-top').hidden = !S.concept;
  history.replaceState(null, '', S.concept ? '#concept=' + encodeURIComponent(S.concept.name) : location.pathname + '#atlas');
  paint(); bylayer(); if (S.concept) conceptCards();
}
function conceptBars() {
  const c = S.concept, top = Math.max(...c.layers.map(l => l.max), 1e-9), peak = c.layers.reduce((b, l, L) => l.max > c.layers[b].max ? L : b, 0);
  $('bylayer-head').textContent = `Trace of “${c.name}” through the layers`;
  $('bylayer-title').textContent = `Bar height = strongest separation in that layer (|d|, concept vs hard negatives). Peak at layer ${peak}. Faded bars: under half of that layer's top ${S.concepts.top_per_layer} neurons held up on held-out scenes.`;
  $('bylayer').innerHTML = c.layers.map((l, L) => `<div style="height:${(l.max / top * 100).toFixed(1)}%;background:#a6d860;opacity:${l.replicated >= .5 ? 1 : .35}" title="Layer ${L}: strongest |d| ${l.max.toFixed(2)} · ${fmt(l.strong)} neurons with |d| ≥ ${S.concepts.strong} · ${(l.replicated * 100).toFixed(0)}% of top ${S.concepts.top_per_layer} replicated"></div>`).join('');
  $('bylayer').querySelectorAll('div').forEach((b, L) => b.onclick = () => document.querySelector(`.tile[data-layer="${L}"]`).scrollIntoView({ block: 'center', behavior: 'smooth' }));
}
function conceptCards() {
  const c = S.concept, rows = c.layers.flatMap((l, L) => l.neurons.map(([n, d, h]) => ({ layer: L, neuron: n, d, h }))).sort((a, b) => Math.abs(b.d) - Math.abs(a.d)).slice(0, 24);
  $('concept-name').textContent = c.name;
  $('concept-cards').innerHTML = rows.map((r, i) => { const ok = Math.sign(r.d) === Math.sign(r.h) && Math.abs(r.h) >= S.concepts.replicates;
    return `<button type="button" class="card" data-i="${i}"><span class="id">L${r.layer} N${r.neuron}</span>
      <div class="dirs"><em>d</em><span style="color:${r.d > 0 ? 'var(--lime-strong)' : 'var(--rose)'}">${r.d > 0 ? '+' : '−'}${Math.abs(r.d).toFixed(2)}</span><em>held-out</em><span class="held ${ok ? 'ok' : 'no'}">${r.h >= 0 ? '+' : '−'}${Math.abs(r.h).toFixed(2)} ${ok ? '✓' : '✗'}</span></div></button>`; }).join('');
  $('concept-cards').querySelectorAll('.card').forEach(b => b.onclick = () => { const i = +b.dataset.i; openNeuron(rows[i].layer, rows[i].neuron, { rows, at: i }); });
}

init().catch(e => { $('grid').innerHTML = `<p class="dim">The atlas has not been built yet: ${esc(e.message)}</p>`; });
