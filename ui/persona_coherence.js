(() => {
  'use strict';
  const refs = {}; let timer, active, serial = 0, submitting = false, linked = false;
  const terminal = new Set(['complete', 'failed', 'cancelled', 'interrupted', 'partial', 'incomplete']);
  const node = (tag, text, cls) => {const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n;};
  const field = (label, input) => {const n = node('label', label); n.append(input); return n;};
  async function request(url, body) {const r = await fetch(url, body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {}); const v = await r.json(); if (!r.ok) throw Error(v.error || r.statusText); return v;}
  function details(label, value) {const d = node('details'); d.append(node('summary', label), node('pre', JSON.stringify(value, null, 2))); return d;}
  function mount() {
    if (refs.root) return;
    const parent = document.getElementById('sculptor-view'); if (!parent) return;
    refs.root = node('section', undefined, 'pc-panel'); refs.root.id = 'coherence-evaluation';
    refs.root.append(node('p', 'MEASURE BEFORE KEEPING A RECIPE', 'pc-eyebrow'), node('h2', 'Coherence & evidence checks'), node('p', 'Compare dose scales on a fixed suite. Inspect invented or missing facts, contradictions and repetition. These checks do not grade style or personality.', 'hint'));
    refs.suite = node('div'); refs.recipe = node('select'); refs.recipe.required = true; refs.recipe.setAttribute('aria-label', 'Saved recipe');
    refs.seed = node('input'); refs.seed.type = 'number'; refs.seed.min = 0; refs.seed.step = 1; refs.seed.value = 7035;
    refs.budget = node('input'); refs.budget.type = 'number'; refs.budget.min = 2048; refs.budget.max = 4096; refs.budget.step = 256; refs.budget.value = 2048;
    refs.run = node('button', 'Evaluate this recipe'); refs.run.type = 'submit';
    const form = node('form', undefined, 'pc-controls'); form.append(field('Saved recipe', refs.recipe), field('Shared seed', refs.seed), field('Token budget', refs.budget), refs.run);
    form.onsubmit = async event => {event.preventDefault(); if (submitting || !refs.recipe.value) return; submitting = true; refs.run.disabled = true; refs.status.textContent = 'Submitting explicit evaluation…'; try {const value = await request('/api/coherence/runs', {recipe_id: refs.recipe.value, seed: Number(refs.seed.value), max_tokens: Number(refs.budget.value)}); await load(value.id); await history();} catch (e) {refs.status.textContent = e.message;} finally {submitting = false; refs.run.disabled = !refs.recipe.value;}};
    refs.history = node('select'); refs.history.setAttribute('aria-label', 'Saved coherence runs'); refs.history.onchange = () => {if (refs.history.value) load(refs.history.value);};
    const refresh = node('button', 'Refresh saved evaluations'); refresh.type = 'button'; refresh.onclick = history;
    refs.status = node('p', '', 'hint'); refs.status.setAttribute('aria-live', 'polite'); refs.output = node('div');
    refs.root.append(refs.suite, form, field('Saved evaluations', refs.history), refresh, refs.status, refs.output); parent.append(refs.root);
  }
  async function history() {
    try {
      const [meta, recipes] = await Promise.all([request('/api/coherence'), request('/api/saved-recipes')]);
      refs.suite.replaceChildren(node('p', `${Array.isArray(meta.suite?.cases) ? meta.suite.cases.length : meta.suite?.cases ?? '—'} suite cases · ${meta.suite?.sha256 ? `suite ${meta.suite.sha256.slice(0, 12)}` : 'Suite identity pending'}`, 'hint'), details('Suite and defaults', {suite: meta.suite, defaults: meta.defaults}));
      const current = refs.recipe.value; refs.recipe.replaceChildren(new Option('Choose a saved recipe…', '')); for (const r of recipes.recipes || []) refs.recipe.add(new Option(r.name || r.id, r.id)); refs.recipe.value = current; refs.run.disabled = submitting || !refs.recipe.value; refs.recipe.onchange = () => {refs.run.disabled = submitting || !refs.recipe.value;};
      refs.history.replaceChildren(new Option('Choose a saved evaluation…', '')); for (const r of meta.runs || []) refs.history.add(new Option(`${r.id.slice(0, 10)} · ${r.state} · ${r.input?.recipe_id || r.recipe_id || ''}`, r.id)); if (active) refs.history.value = active;
    } catch (e) {refs.status.textContent = e.message;}
  }
  function render(run) {
    refs.output.replaceChildren(); refs.status.textContent = `${run.id} · ${run.state}`;
    if (!terminal.has(run.state)) {const stop=node('button','Stop after in-flight responses finish');stop.onclick=async()=>{stop.disabled=true;try{await request('/api/coherence/cancel',{id:run.id});await load(run.id);}catch(e){refs.status.textContent=e.message;stop.disabled=false;}};refs.output.append(stop);}
    refs.output.append(node('p', `Recorded evaluation: recipe ${run.input?.recipe_id || '—'} · seed ${run.input?.seed ?? '—'} · budget ${run.input?.max_tokens ?? '—'}`, 'hint'));
    if (run.error) refs.output.append(node('p', typeof run.error === 'string' ? run.error : JSON.stringify(run.error), 'pc-error'));
    if (run.recommendation) refs.output.append(node('h3', 'Recorded recommendation'), node('p', typeof run.recommendation === 'object' ? `${run.recommendation.status || 'Pending'} · scale ${run.recommendation.scale ?? '—'} · ${run.recommendation.reason || ''}` : String(run.recommendation)), details('Recommendation and supporting evidence', run.recommendation));
    const grid = node('div', undefined, 'pc-trials');
    for (const trial of run.trials || []) {
      const card = node('article'); card.append(node('h3', `Scale ${trial.scale ?? '—'} · ${trial.split || '—'}`), node('p', trial.state || 'Pending', 'hint'));
      if (trial.error) card.append(node('p', typeof trial.error === 'string' ? trial.error : JSON.stringify(trial.error), 'pc-error'));
      if (trial.summary) {card.append(node('p', `Contract: ${trial.summary.pass_count ?? '—'} pass · ${trial.summary.fail_count ?? '—'} fail · ${trial.summary.unknown_count ?? '—'} unknown. Backoff evidence: ${trial.summary.guard_state || trial.summary.state || 'pending'}`, 'hint')); for (const issue of trial.summary.issues || []) card.append(node('p', `${issue.case_id || 'Case'} · ${issue.kind || 'Issue'}: ${issue.detail || ''}`, 'pc-error')); const summary = node('pre', JSON.stringify(trial.summary, null, 2)); summary.className = 'pc-summary'; card.append(summary);} else card.append(node('p', 'Evidence checks pending.', 'hint'));
      card.append(details('Exact trial, evidence and batch IDs', trial)); grid.append(card);
    }
    refs.output.append(grid, details('Run input and provenance', {input: run.input, model_identity:run.model_identity,recipe_sha256:run.recipe_sha256,suite_sha256:run.suite_sha256,evaluator_sha256:run.evaluator_sha256,scope:run.scope}));
  }
  async function load(id) {
    clearTimeout(timer); const generation = ++serial; active = id;
    try {const run = await request('/api/coherence/run?id=' + encodeURIComponent(id)); if (generation !== serial) return; render(run); if (!terminal.has(run.state)) timer = setTimeout(() => load(id), 3000);} catch (e) {if (generation === serial) refs.status.textContent = e.message;}
  }
  async function open() {mount(); if (refs.root) {await history();const id=new URLSearchParams(location.search).get('coherence');if(!linked && id && /^[a-f0-9]{32}$/.test(id)){linked=true;await load(id);refs.root.scrollIntoView({block:'start'});}}}
  window.PersonaCoherence = {open};
})();
