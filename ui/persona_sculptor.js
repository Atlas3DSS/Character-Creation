/* Persona studio: explicit plans, immutable recipe builds, and shared-battery comparisons. */
(() => {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const TRAITS = [
    ['openness', 'Openness', 'Familiar & practical', 'Curious & inventive', 'O'],
    ['conscientiousness', 'Conscientiousness', 'Spontaneous & flexible', 'Orderly & deliberate', 'C'],
    ['extraversion', 'Extraversion', 'Reserved & reflective', 'Outgoing & energetic', 'E'],
    ['agreeableness', 'Agreeableness', 'Skeptical & direct', 'Cooperative & warm', 'A'],
    ['neuroticism', 'Neuroticism', 'Steady & unruffled', 'Sensitive & reactive', 'N']
  ];
  const terminal = new Set(['complete', 'partial', 'failed', 'cancelled', 'interrupted', 'incomplete']);
  const app = {ready: false, meta: null, capacity: null, plan: null, pendingPlan: null, build: null, campaign: null,
    dials: Object.fromEntries(TRAITS.map(([key]) => [key, 0])), revision: 0, builtRevision: -1,
    battery: [], selected: new Set(), pinned: null, promptIndex: 0, busy: new Set(), saved: new Set(),
    promptOffset: 0, promptTotal: 0, searchSerial: 0, campaignSerial: 0, loadingCampaign: null, jobs: null, session: null, preset: null, pendingPreset: null};
  const refs = {};
  let searchTimer, pollTimer;
  const clone = value => JSON.parse(JSON.stringify(value));
  const signed = value => `${Number(value) < 0 ? '−' : '+'}${Math.abs(Number(value) || 0).toFixed(2)}`;
  const count = value => Number.isFinite(value) ? value.toLocaleString() : '—';
  function node(tag, text, cls) {
    const value = document.createElement(tag); if (text !== undefined) value.textContent = text;
    if (cls) value.className = cls; return value;
  }
  function button(text, action, cls = '') {const value = node('button', text, cls); value.type = 'button'; value.onclick = action; return value;}
  function input(type, value) {const result = node('input'); result.type = type; result.value = value; return result;}
  function select(values, label) {
    const result = node('select'); if (label) result.setAttribute('aria-label', label);
    for (const [value, text] of values) {const option = node('option', text); option.value = value; result.append(option);} return result;
  }
  function field(label, value, help) {const result = node('label', undefined, 'ps-field'); result.append(node('span', label), value); if (help) result.append(node('small', help)); return result;}
  function row(...children) {const value = node('div', undefined, 'ps-row'); value.append(...children); return value;}
  function hint(text) {return node('p', text, 'ps-hint');}
  function svg(tag, attrs = {}, text) {const value = document.createElementNS(NS, tag); for (const [key, item] of Object.entries(attrs)) value.setAttribute(key, String(item)); if (text !== undefined) value.textContent = text; return value;}
  function human(value) {return String(value || '').replaceAll('_', ' ');}
  function notice(message = '', error = false) {refs.notice.textContent = message; refs.notice.hidden = !message; refs.notice.classList.toggle('ps-error', error);}
  function scroll(target) {target.scrollIntoView({behavior: window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start'});}
  async function request(url, payload) {
    const response = await fetch(url, payload === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    const value = await response.json().catch(() => ({error: `Request failed (${response.status})`}));
    if (!response.ok) throw Error(value.error || response.statusText || 'Request failed.'); return value;
  }
  async function action(key, work) {
    if (app.busy.has(key)) return;
    app.busy.add(key); updateControls();
    try {await work();} catch (error) {notice(error.message || 'The action could not be completed.', true);}
    finally {app.busy.delete(key); updateControls(); if (key.startsWith('save:') || key.startsWith('review:')) renderGallery();}
  }
  function dirty() {app.revision++; renderProfile(); updateControls();}
  function displayDate(value) {
    if (!value) return '—'; const date = new Date(typeof value === 'number' && value < 1e12 ? value * 1000 : value);
    return Number.isFinite(date.valueOf()) ? date.toLocaleString([], {month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'}) : String(value);
  }
  function ensure() {
    if (app.ready) return true;
    const root = document.getElementById('sculptor-view'); if (!root) return false;
    app.ready = true; refs.root = root; root.classList.add('persona-sculptor');
    const header = node('div', undefined, 'ps-header'), title = node('div');
    title.append(node('p', 'CHARACTER CREATION / PERSONA STUDIO', 'ps-eyebrow'), node('h2', 'Shape a voice. Test the recipe.'),
      hint('Describe a persona, sculpt its direction settings, and hear the same prompts answered six different ways.'));
    const steps = node('div', undefined, 'ps-steps');
    steps.append(button('01  Sculpt', () => scroll(refs.sculpt)), button('02  Candidates', () => scroll(refs.gallerySection)), button('03  Compare', () => scroll(refs.resultsSection)));
    header.append(title, steps); refs.notice = node('p', undefined, 'ps-notice'); refs.notice.hidden = true; refs.notice.setAttribute('role', 'status');
    refs.fleet = node('section', undefined, 'ps-fleet');
    const fleetLeft = node('div'); refs.capacity = node('strong', 'Reading the configured fleet…'); refs.sessionStatus = hint('');
    refs.cards = node('div', undefined, 'ps-card-lights'); fleetLeft.append(refs.capacity, refs.cards, refs.sessionStatus);
    const fleetRight = node('div', undefined, 'ps-session-actions');
    refs.sessionMinutes = select([['60', '1 hour'], ['120', '2 hours'], ['240', '4 hours'], ['480', '8 hours']], 'Studio session duration'); refs.sessionMinutes.value = '120';
    refs.startSession = button('Start studio session', startSession, 'ps-primary');
    refs.stopSession = button('Stop this studio session', stopSession, 'ps-stop'); refs.stopSession.hidden = true;
    fleetRight.append(refs.sessionMinutes, refs.startSession, refs.stopSession, button('Jobs & schedules ↗', () => {
      const change = typeof tab === 'function' ? tab : window.tab; if (typeof change === 'function') change('jobs');
    }, 'ps-link-button'), button('Refresh', refreshFleet, 'ps-link-button'));
    refs.fleet.append(fleetLeft, fleetRight);
    refs.startupHint = hint('Cold model preparation can take tens of minutes; session time includes startup. Starting a session requests John and six leased Quincies.');
    refs.startupHint.classList.add('ps-startup-hint');

    refs.presets = node('section', undefined, 'ps-presets');
    const presetHead = node('div', undefined, 'ps-preset-head'), presetTitle = node('div');
    presetTitle.append(node('p', 'START FROM A RECIPE', 'ps-eyebrow'), node('h3', 'A familiar spark. Your own character.'));
    refs.presetSearch = input('search', ''); refs.presetSearch.placeholder = 'Find a preset…'; refs.presetSearch.setAttribute('aria-label', 'Search persona presets'); refs.presetSearch.oninput = renderPresets;
    presetHead.append(presetTitle, refs.presetSearch);
    const presetLayout = node('div', undefined, 'ps-preset-layout'); refs.presetGrid = node('div', undefined, 'ps-preset-grid'); refs.presetPreview = node('div', undefined, 'ps-preset-preview');
    presetLayout.append(refs.presetGrid, refs.presetPreview); refs.presets.append(presetHead, presetLayout);

    refs.sculpt = node('section', undefined, 'ps-sculpt'); refs.sculpt.id = 'ps-sculpt';
    const briefPanel = node('div', undefined, 'ps-brief ps-panel');
    briefPanel.append(node('p', '01 / THE IDEA', 'ps-eyebrow'), node('h3', 'Who are we creating?'));
    refs.brief = node('textarea'); refs.brief.rows = 6; refs.brief.maxLength = 4000;
    refs.brief.placeholder = 'A calm, curious starship navigator: warm with the crew, direct under pressure, and fond of an unexpected idea.';
    refs.brief.setAttribute('aria-label', 'Desired persona'); refs.brief.oninput = () => {app.plan = null; renderPlanEvidence(); dirty();};
    refs.interpret = button('Ask John to interpret', interpret, 'ps-primary');
    refs.planStatus = hint('Your brief becomes a proposed dial profile. It is kept out of the response prompts.');
    refs.applyPlan = button('Apply returned interpretation', () => {if (app.pendingPreset) applyPreset(app.pendingPreset); else if (app.pendingPlan) applyPlan(app.pendingPlan);}, 'ps-secondary'); refs.applyPlan.hidden = true;
    refs.planEvidence = node('div', undefined, 'ps-plan-evidence');
    briefPanel.append(refs.brief, refs.interpret, refs.applyPlan, refs.planStatus, refs.planEvidence);
    const recipeDetails = node('details', undefined, 'ps-advanced'); recipeDetails.append(node('summary', 'Recipe scope & reproducibility'));
    refs.layers = input('text', ''); refs.layers.placeholder = 'Calibrated layer IDs'; refs.layers.oninput = dirty;
    refs.phase = select([['all', 'Prompt + response'], ['prefill', 'Prompt only'], ['decode', 'Response only']], 'Intervention phase'); refs.phase.onchange = dirty;
    refs.variantCount = select(Array.from({length: 6}, (_, i) => [String(i + 1), `${i + 1} candidate${i ? 's' : ''}`]), 'Candidate count'); refs.variantCount.value = '6'; refs.variantCount.onchange = dirty;
    recipeDetails.append(field('Calibrated layers', refs.layers, 'Comma-separated IDs. Only available calibrated layers are accepted.'), field('Apply during', refs.phase), field('Build size', refs.variantCount),
      hint('All candidates use this checkpoint’s measured directions. A layer or trait label does not establish a causal personality effect.'));
    briefPanel.append(recipeDetails);

    const visual = node('div', undefined, 'ps-profile-panel');
    visual.append(node('p', 'REQUESTED INTERVENTION PROFILE', 'ps-eyebrow'));
    refs.profileGraphic = node('div', undefined, 'ps-profile-graphic'); visual.append(refs.profileGraphic);
    refs.profileLabel = node('strong', 'Draft profile', 'ps-profile-label');
    refs.profileScope = node('p', 'Direction settings · no measured personality score', 'ps-profile-scope');
    const legend = node('div', undefined, 'ps-profile-legend'); legend.append(node('span', '● Your settings', 'ps-lime'), node('span', '┄ Neutral dials'));
    visual.append(refs.profileLabel, refs.profileScope, legend, hint('Each spoke is a signed trait direction. The outline visualizes your settings, not a personality prediction.'));
    const dialPanel = node('div', undefined, 'ps-panel ps-dial-panel'); dialPanel.append(node('p', '02 / THE DIALS', 'ps-eyebrow'));
    refs.dials = {}; refs.dialValues = {};
    for (const [key, label, low, high] of TRAITS) {
      const control = input('range', 0); control.min = '-1'; control.max = '1'; control.step = '.05'; control.id = `ps-${key}`;
      const wrap = node('div', undefined, 'ps-dial'), head = node('div', undefined, 'ps-dial-head'), name = node('label', label); name.htmlFor = control.id;
      const value = node('output', '0.00'); value.htmlFor = control.id;
      control.oninput = () => {app.dials[key] = Number(control.value); dirty();};
      head.append(name, value); const poles = node('div', undefined, 'ps-poles'); poles.append(node('span', low), node('span', high));
      wrap.append(head, control, poles); dialPanel.append(wrap); refs.dials[key] = control; refs.dialValues[key] = value;
    }
    const strengthHead = row(node('label', 'Overall strength'), node('output', '0.35')); strengthHead.classList.add('ps-strength-head');
    refs.strength = input('range', .35); refs.strength.min = '0'; refs.strength.max = '.5'; refs.strength.step = '.01'; refs.strength.id = 'ps-strength';
    strengthHead.children[0].htmlFor = refs.strength.id; refs.strengthValue = strengthHead.children[1]; refs.strengthValue.htmlFor = refs.strength.id;
    refs.strength.oninput = dirty;
    dialPanel.append(strengthHead, refs.strength, hint('A shared normalized intervention budget across the selected layers and traits.'),
      button('Reset dials to neutral', () => {for (const [key] of TRAITS) {app.dials[key] = 0; refs.dials[key].value = '0';} dirty();}, 'ps-link-button'));
    refs.sculpt.append(briefPanel, visual, dialPanel);
    const buildBar = node('div', undefined, 'ps-build-bar'); refs.buildStatus = hint('Choose your settings, then build the candidate recipes.');
    refs.buildButton = button('Build candidate recipes', buildCandidates, 'ps-primary ps-build-button'); buildBar.append(refs.buildStatus, refs.buildButton);

    refs.gallerySection = node('section', undefined, 'ps-section'); refs.gallerySection.id = 'ps-gallery';
    const galleryHead = row(node('div'), node('div', undefined, 'ps-row'));
    galleryHead.children[0].append(node('p', '03 / THE CANDIDATES', 'ps-eyebrow'), node('h3', 'A cast of possibilities'));
    refs.galleryNote = hint('Your generated recipes will appear here. No responses have been run.'); galleryHead.children[0].append(refs.galleryNote);
    galleryHead.children[1].append(button('Select all', () => {app.selected = new Set((app.build?.candidates || []).map(item => item.id)); renderGallery(); updateControls();}, 'ps-link-button'));
    galleryHead.classList.add('ps-spread'); refs.gallery = node('div', undefined, 'ps-gallery'); refs.gallerySection.append(galleryHead, refs.gallery);

    refs.batterySection = node('section', undefined, 'ps-section ps-panel'); refs.batterySection.id = 'ps-battery';
    refs.batterySection.append(node('p', '04 / SHARED PROMPT BATTERY', 'ps-eyebrow'), node('h3', 'Give every voice the same moment'),
      hint('One unsteered baseline per prompt, followed by your selected candidates. All use the same neutral system, user text and seed. Registered persona system instructions are excluded and recorded.'));
    const batteryLayout = node('div', undefined, 'ps-battery-layout'), selectedPanel = node('div'), sourcePanel = node('div');
    refs.batteryList = node('div', undefined, 'ps-battery-list'); refs.batteryCount = node('strong', '0 / 12 prompts');
    selectedPanel.append(row(refs.batteryCount, button('Restore default battery', restoreBattery, 'ps-link-button')), refs.batteryList);
    const custom = node('details', undefined, 'ps-custom'); custom.append(node('summary', 'Add your own prompts'));
    refs.custom = node('textarea'); refs.custom.rows = 4; refs.custom.placeholder = 'One prompt…\n\nA second prompt…'; refs.custom.setAttribute('aria-label', 'Custom prompts separated by blank lines');
    custom.append(hint('Separate prompts with a blank line. Each complete prompt is shared across the baseline and candidates.'), refs.custom, button('Add custom prompts', addCustom, 'ps-secondary')); selectedPanel.append(custom);
    const library = node('details', undefined, 'ps-library'); library.open = true; library.append(node('summary', 'Browse registered prompts'));
    refs.corpus = select([['big_five', 'Big Five'], ['scotus', 'SCOTUS'], ['skippy', 'Skippy books'], ['', 'All corpora']], 'Corpus');
    refs.split = select([['discovery', 'Discovery'], ['heldout', 'Held-out validation'], ['all', 'Both splits']], 'Prompt split');
    refs.search = input('search', ''); refs.search.placeholder = 'Find a scenario, category or source…'; refs.search.setAttribute('aria-label', 'Search prompt bank');
    for (const control of [refs.corpus, refs.split]) control.onchange = () => {app.promptOffset = 0; searchPrompts();};
    refs.search.oninput = () => {clearTimeout(searchTimer); searchTimer = setTimeout(() => {app.promptOffset = 0; searchPrompts();}, 280);};
    refs.promptResults = node('div', undefined, 'ps-prompt-results'); refs.promptPage = node('span', '', 'ps-hint');
    refs.prev = button('←', () => {app.promptOffset = Math.max(0, app.promptOffset - 6); searchPrompts();}); refs.prev.setAttribute('aria-label', 'Previous prompt page');
    refs.next = button('→', () => {app.promptOffset += 6; searchPrompts();}); refs.next.setAttribute('aria-label', 'Next prompt page');
    library.append(row(refs.corpus, refs.split), refs.search, refs.promptResults, row(refs.prev, refs.promptPage, refs.next)); sourcePanel.append(library);
    batteryLayout.append(selectedPanel, sourcePanel); refs.batterySection.append(batteryLayout);
    refs.seed = input('number', 7035); refs.seed.min = '0'; refs.seed.max = '4294967295'; refs.seed.step = '1';
    refs.budget = select([['2048', '2,048 tokens'], ['3072', '3,072 tokens'], ['4096', '4,096 tokens']], 'Per-response token budget');
    const runBar = node('div', undefined, 'ps-run-bar'); refs.runCount = node('strong'); refs.runHint = hint('');
    const runNumbers = node('div'); runNumbers.append(refs.runCount, refs.runHint);
    refs.runButton = button('Run shared battery', runCampaign, 'ps-primary');
    runBar.append(field('Shared seed', refs.seed), field('Response budget', refs.budget), runNumbers, refs.runButton);
    refs.batterySection.append(runBar, hint('Thinking is off. Complete answers are required; truncation and errors remain visible. Prompt plus budget must fit the model’s context.'));

    refs.resultsSection = node('section', undefined, 'ps-section'); refs.resultsSection.id = 'ps-results';
    const resultsHead = row(node('div'), node('div', undefined, 'ps-row')); resultsHead.classList.add('ps-spread');
    resultsHead.children[0].append(node('p', '05 / LISTEN & COMPARE', 'ps-eyebrow'), node('h3', 'Keep the baseline in view'));
    refs.history = select([['', 'Recent campaigns…']], 'Choose a campaign'); refs.history.onchange = () => {if (refs.history.value) loadCampaign(refs.history.value, true);};
    refs.cancel = button('Stop future waves', cancelCampaign, 'ps-stop'); refs.cancel.hidden = true;
    resultsHead.children[1].append(refs.history, refs.cancel); refs.progress = node('div', undefined, 'ps-campaign-progress');
    refs.promptTabs = node('div', undefined, 'ps-prompt-tabs'); refs.comparison = node('div', undefined, 'ps-comparison');
    refs.resultsSection.append(resultsHead, refs.progress, refs.promptTabs, refs.comparison,
      hint('Ratings and preferences are yours. They are not automatic evaluations or evidence that a human personality has been reproduced.'));
    root.append(header, refs.fleet, refs.startupHint, refs.notice, refs.presets, refs.sculpt, buildBar, refs.gallerySection, refs.batterySection, refs.resultsSection);
    renderPresets(); renderProfile(); renderGallery(); renderBattery(); renderResults(); updateControls(); return true;
  }
  function safeLink(source) {
    const url = typeof source === 'string' ? source : source.url;
    try {const parsed = new URL(url); if (!['https:', 'http:'].includes(parsed.protocol)) return null;
      const link = node('a', typeof source === 'string' ? parsed.hostname : source.title || source.label || parsed.hostname);
      link.href = parsed.href; link.target = '_blank'; link.rel = 'noopener noreferrer'; return link;
    } catch (_) {return null;}
  }
  function notes(value) {return Array.isArray(value) ? value : value ? [value] : [];}
  function renderPresets() {
    refs.presetGrid.replaceChildren(); refs.presetPreview.replaceChildren();
    const presets = app.meta?.presets || [], query = refs.presetSearch.value.toLowerCase().trim();
    if (!presets.length) {refs.presetGrid.append(hint('Preset recipes will appear when the catalog is available.')); return;}
    if (!presets.some(preset => preset.id === app.preset)) app.preset = presets[0].id;
    for (const [index, preset] of presets.entries()) {
      if (query && ![preset.name, preset.description, preset.brief].filter(Boolean).join(' ').toLowerCase().includes(query)) continue;
      const card = button('', () => {app.preset = preset.id; renderPresets();}, 'ps-preset-card');
      card.classList.toggle('ps-featured', index < 2); card.classList.toggle('ps-active', preset.id === app.preset); card.setAttribute('aria-pressed', String(preset.id === app.preset));
      card.append(node('strong', preset.name), node('small', index < 2 ? 'Inspired approximation' : 'Editable archetype')); refs.presetGrid.append(card);
    }
    if (!refs.presetGrid.children.length) refs.presetGrid.append(hint('No matching preset.'));
    const selected = presets.find(preset => preset.id === app.preset);
    refs.presetPreview.append(node('h4', selected.name), hint(selected.description || 'An editable starting profile to test against the baseline.'));
    if (selected.brief) refs.presetPreview.append(node('blockquote', selected.brief));
    const values = node('div', undefined, 'ps-candidate-values');
    for (const [key, label, , , letter] of TRAITS) {const value = node('span', `${letter} ${signed(selected.dials?.[key])}`); value.title = label; values.append(value);}
    refs.presetPreview.append(values, node('p', 'Inspired settings, not a replication of a person, fictional character or another model’s weights.', 'ps-caution'));
    refs.presetLoad = button(app.busy.has('preset') ? 'Loading recipe…' : 'Load preset & build candidates', () => loadPreset(selected.id), 'ps-primary');
    refs.presetLoad.disabled = app.busy.has('preset') || app.busy.has('build') || app.busy.has('plan'); refs.presetPreview.append(refs.presetLoad);
    const details = node('details'); details.append(node('summary', 'Sources & approximation notes'));
    for (const text of [...notes(selected.evidence_basis), ...notes(selected.notes), ...notes(selected.unsupported_traits)]) details.append(node('p', typeof text === 'string' ? text : text.description || text.name || JSON.stringify(text)));
    const sources = node('ul'); for (const source of selected.sources || []) {const link = safeLink(source); if (link) {const item = node('li'); item.append(link); sources.append(item);}}
    if (sources.children.length) details.append(sources);
    details.append(hint('Loading a preset builds model-bound recipes without asking John or generating evaluation responses. Every dial remains editable.'));
    refs.presetPreview.append(details);
  }
  function applyPreset(value) {
    if (!value?.build?.candidates?.length || !value.plan?.dials) throw Error('The preset did not return a valid recipe build.');
    applyPlan(value.plan); app.pendingPreset = null; app.build = value.build;
    refs.layers.value = (value.build.layers || []).join(', '); refs.phase.value = value.build.phase || 'all'; refs.strength.value = String(value.build.dose_budget ?? .35);
    refs.variantCount.value = String(value.build.candidates.length); app.selected = new Set(value.build.candidates.map(candidate => candidate.id));
    app.builtRevision = app.revision; renderProfile(); renderGallery(); updateControls();
    notice(`Loaded ${value.preset?.name || value.plan.name || 'preset'} and built ${value.build.candidates.length} candidates. The center recipe is saved; review the battery before generating answers.`);
  }
  function loadPreset(id) {return action('preset', async () => {
    renderPresets(); const revision = app.revision;
    const value = await request('/api/sculptor/preset', {id, count: Number(refs.variantCount.value), seed: Number(refs.seed.value)});
    if (revision !== app.revision) {app.pendingPreset = value; app.pendingPlan = null; refs.applyPlan.textContent = 'Apply returned preset'; refs.applyPlan.hidden = false; notice('The preset is ready. Your draft changed while it was loading; apply it when ready.');}
    else applyPreset(value);
    renderPresets();
  });}
  function radar(dials, compact = false) {
    const graphic = svg('svg', {viewBox: '0 0 360 330', role: compact ? 'presentation' : 'img', 'aria-label': 'Requested Big Five direction settings; dashed outline marks zero on every dial.'});
    if (compact) graphic.setAttribute('aria-hidden', 'true');
    const cx = 180, cy = 157, radius = 104;
    const point = (i, r) => {const angle = -Math.PI / 2 + i * Math.PI * 2 / 5; return [cx + Math.cos(angle) * r, cy + Math.sin(angle) * r];};
    for (const scale of [.15, .575, 1]) graphic.append(svg('polygon', {points: TRAITS.map((_, i) => point(i, radius * scale).join(',')).join(' '), class: scale === .575 ? 'ps-neutral-ring' : 'ps-radar-ring'}));
    TRAITS.forEach(([key, label, , , letter], i) => {
      const end = point(i, radius), name = point(i, radius + 35), actual = point(i, radius * (.575 + .425 * (Number(dials[key]) || 0)));
      graphic.append(svg('line', {x1: cx, y1: cy, x2: end[0], y2: end[1], class: 'ps-spoke'}));
      graphic.append(svg('text', {x: name[0], y: name[1] + 3, 'text-anchor': 'middle', class: 'ps-trait-letter'}, letter));
      if (!compact) graphic.append(svg('text', {x: name[0], y: name[1] + 19, 'text-anchor': 'middle', class: 'ps-trait-number'}, signed(dials[key])));
      graphic.append(svg('circle', {cx: actual[0], cy: actual[1], r: compact ? 4 : 4.5, class: 'ps-profile-dot'}));
    });
    graphic.append(svg('polygon', {points: TRAITS.map(([key], i) => point(i, radius * (.575 + .425 * (Number(dials[key]) || 0))).join(',')).join(' '), class: 'ps-profile-shape'}));
    graphic.append(svg('circle', {cx, cy, r: 3, class: 'ps-origin'})); return graphic;
  }
  function renderProfile() {
    if (!app.ready) return;
    refs.profileGraphic.replaceChildren(radar(app.dials));
    for (const [key] of TRAITS) {refs.dialValues[key].textContent = signed(app.dials[key]); refs.dials[key].setAttribute('aria-valuetext', signed(app.dials[key]) + ' relative direction');}
    refs.strengthValue.textContent = Number(refs.strength.value).toFixed(2);
    refs.profileLabel.textContent = app.plan ? (app.plan.method?.startsWith('john') ? 'John’s proposed profile' : app.plan.method === 'preset_heuristic' ? 'Inspired preset profile' : 'Manual dial profile') : 'Your draft profile';
    refs.profileScope.textContent = `${refs.layers.value ? refs.layers.value.split(',').filter(Boolean).length + ' selected layers' : 'Awaiting calibrated layers'} · ${refs.phase.options?.[refs.phase.selectedIndex]?.text || human(refs.phase.value)}`;
  }
  function layerValues() {
    const values = refs.layers.value.split(',').map(value => value.trim()).filter(Boolean).map(Number);
    const available = (app.meta?.layers || app.meta?.calibrated_layers || []).map(item => typeof item === 'number' ? item : item.layer);
    if (!values.length || values.some(value => !Number.isInteger(value) || value < 0 || !available.includes(value)) || new Set(values).size !== values.length) throw Error('Choose distinct available calibrated layer IDs, separated by commas.');
    return values;
  }
  function settings() {return {brief: refs.brief.value.trim(), dials: {...app.dials}, count: Number(refs.variantCount.value), layers: layerValues(), dose_budget: Number(refs.strength.value), phase: refs.phase.value};}
  function applyPlan(plan) {
    if (!plan?.dials || TRAITS.some(([key]) => !Number.isFinite(plan.dials[key]) || Math.abs(plan.dials[key]) > 1)) throw Error('The plan did not return a valid five-trait profile.');
    app.plan = plan; app.pendingPlan = null; app.pendingPreset = null; refs.applyPlan.hidden = true; refs.brief.value = plan.brief || refs.brief.value;
    for (const [key] of TRAITS) {app.dials[key] = plan.dials[key]; refs.dials[key].value = String(plan.dials[key]);}
    dirty(); renderPlanEvidence(); notice('Interpretation applied. You can edit every dial before building candidates.');
  }
  function renderPlanEvidence() {
    refs.planEvidence.replaceChildren();
    if (!app.plan) {refs.planStatus.textContent = 'Your brief becomes a proposed dial profile. It is kept out of the response prompts.'; return;}
    refs.planStatus.textContent = app.plan.method?.startsWith('john') ? 'Interpreted by the local John model. These are proposed settings, not validated effects.' : app.plan.method === 'preset_heuristic' ? 'Preset approximation. Edit the dials and test its effects against the baseline.' : 'Manual plan from your dials. No model interpretation is claimed.';
    if (app.plan.rationale) refs.planEvidence.append(hint(app.plan.rationale));
    const unsupported = app.plan.unsupported_traits || [];
    if (unsupported.length) refs.planEvidence.append(node('p', 'Not represented by these five directions: ' + unsupported.map(value => typeof value === 'string' ? value : value.description || value.name || human(value.trait)).join('; '), 'ps-caution'));
  }
  function interpret() {return action('plan', async () => {
    if (!refs.brief.value.trim()) throw Error('Describe the persona you want to explore.');
    const revision = app.revision;
    const plan = await request('/api/sculptor/plan', {brief: refs.brief.value.trim(), mode: 'john'});
    if (revision !== app.revision) {app.pendingPlan = plan; app.pendingPreset = null; refs.applyPlan.textContent = 'Apply returned interpretation'; refs.applyPlan.hidden = false; notice('John’s interpretation is ready. Your draft changed while it was being prepared; apply it when you are ready.');}
    else applyPlan(plan);
  });}
  function buildCandidates() {return action('build', async () => {
    const snapshot = settings(), revision = app.revision;
    let plan = app.plan;
    if (!plan && snapshot.brief) plan = await request('/api/sculptor/plan', {brief: snapshot.brief, mode: 'manual', dials: snapshot.dials});
    const build = await request('/api/sculptor/build', {...(plan ? {plan_id: plan.id} : {}), ...snapshot});
    if (!Array.isArray(build.candidates) || !build.candidates.length) throw Error('The builder did not return any candidate recipes.');
    app.build = build; app.builtRevision = revision; app.selected = new Set(build.candidates.map(candidate => candidate.id));
    if (revision === app.revision) {app.plan = plan; renderPlanEvidence(); renderProfile();}
    renderGallery(); updateControls();
    notice(revision === app.revision ? `${build.candidates.length} recipes built. Review the shared battery, then run the comparison.` : 'Recipes were built from the earlier settings. Your draft has changed; rebuild to apply the new dials.');
  });}
  function reviewFor(id) {
    if (app.campaign?.build?.id !== app.build?.id) return {};
    const records = app.campaign?.manual_reviews || {};
    return Array.isArray(records) ? records.find(item => item.candidate_id === id) || {} : records[id] || {};
  }
  function campaignCandidates() {return app.campaign?.build?.candidates || app.campaign?.candidates || [];}
  function renderGallery() {
    refs.gallery.replaceChildren(); const candidates = app.build?.candidates || [];
    refs.galleryNote.textContent = candidates.length ? `${candidates.length} immutable recipe candidates · select the ones to test. Saving a recipe does not establish its effects.` : 'Build a candidate set to explore the profile around your chosen settings.';
    if (!candidates.length) {
      const empty = node('div', undefined, 'ps-gallery-empty'); empty.append(node('span', 'O  C  E  A  N', 'ps-empty-letters'), node('strong', 'The cast starts with your dials'), hint('Build recipes first. Responses are generated only when you run a shared prompt battery.')); refs.gallery.append(empty); return;
    }
    for (const [index, candidate] of candidates.entries()) {
      const card = node('article', undefined, 'ps-candidate'); card.classList.toggle('ps-selected', app.selected.has(candidate.id));
      const header = row(node('span', String(index + 1).padStart(2, '0'), 'ps-candidate-number'), node('h4', candidate.name));
      const choose = input('checkbox', candidate.id); choose.checked = app.selected.has(candidate.id); choose.setAttribute('aria-label', `Include ${candidate.name} in the next run`);
      choose.onchange = () => {choose.checked ? app.selected.add(candidate.id) : app.selected.delete(candidate.id); card.classList.toggle('ps-selected', choose.checked); updateControls();}; header.append(choose);
      const miniature = node('div', undefined, 'ps-mini-profile'); miniature.append(radar(candidate.dials || {}, true));
      const values = node('div', undefined, 'ps-candidate-values'); for (const [key, label, , , letter] of TRAITS) {const value = node('span', `${letter} ${signed(candidate.dials?.[key])}`); value.title = label; values.append(value);}
      card.append(header, miniature, values, hint(candidate.explanation || 'A measured-direction recipe to test against the shared baseline.'),
        node('p', `Strength ${Number(candidate.dose_budget ?? app.build.dose_budget).toFixed(2)} · ${human(candidate.phase || app.build.phase)}`, 'ps-recipe-line'));
      const tools = row(button('Use these dials', () => useCandidate(candidate), 'ps-link-button'), button('Select only', () => {app.selected = new Set([candidate.id]); renderGallery(); updateControls();}, 'ps-link-button'));
      const belongs = app.campaign?.build?.id === app.build.id && campaignCandidates().some(item => item.id === candidate.id) && (!app.campaign.input?.candidate_ids || app.campaign.input.candidate_ids.includes(candidate.id));
      const pin = button(app.pinned === candidate.id && belongs ? 'Pinned below ↓' : 'Compare below ↓', () => {app.pinned = candidate.id; renderResults(); renderGallery(); scroll(refs.resultsSection);}, 'ps-secondary'); pin.disabled = !belongs;
      const saved = app.saved.has(app.build.id + ':' + candidate.id);
      const save = button(saved ? 'Recipe saved ✓' : 'Save recipe', () => saveCandidate(candidate), 'ps-secondary'); save.disabled = saved || app.busy.has('save:' + candidate.id);
      const actions = row(pin, save); card.append(tools, actions);
      const review = reviewFor(candidate.id), rating = select([['', 'Not rated'], ['1', '1 · Poor fit'], ['2', '2 · Slight fit'], ['3', '3 · Mixed'], ['4', '4 · Good fit'], ['5', '5 · Strong fit']], `Your rating for ${candidate.name}`);
      rating.value = review.rating == null ? '' : String(review.rating); rating.disabled = !belongs || app.busy.has('review:' + candidate.id);
      rating.onchange = () => saveReview(candidate.id, rating.value ? Number(rating.value) : null, Boolean(review.preferred));
      const preferred = button(review.preferred ? '★ Preferred' : '☆ Prefer', () => saveReview(candidate.id, review.rating ?? null, !review.preferred), 'ps-prefer'); preferred.disabled = !belongs || app.busy.has('review:' + candidate.id); preferred.setAttribute('aria-pressed', String(Boolean(review.preferred)));
      card.append(node('p', 'YOUR REVIEW', 'ps-eyebrow ps-review-label'), row(rating, preferred)); refs.gallery.append(card);
    }
  }
  function useCandidate(candidate) {
    for (const [key] of TRAITS) {app.dials[key] = Number(candidate.dials?.[key]) || 0; refs.dials[key].value = String(app.dials[key]);}
    refs.strength.value = String(candidate.dose_budget ?? app.build.dose_budget); refs.layers.value = (candidate.layers || app.build.layers || []).join(', ');
    refs.phase.value = candidate.phase || app.build.phase || 'all'; app.plan = null; renderPlanEvidence(); dirty();
    notice(`${candidate.name} is now your editable draft. Build a new set to explore around it. To rerun an exact saved candidate, reopen its campaign and choose “Select only”.`); scroll(refs.sculpt);
  }
  function saveCandidate(candidate) {return action('save:' + candidate.id, async () => {
    const savedKey = app.build.id + ':' + candidate.id;
    await request('/api/saved-recipes', {name: candidate.name, model_identity: app.build.model_identity || app.meta.model_identity, pushes: candidate.pushes, phase: candidate.phase || app.build.phase});
    app.saved.add(savedKey); notice(`Saved “${candidate.name}” to the reusable recipe library.`);
  });}
  function saveReview(id, rating, preferred) {return action('review:' + id, async () => {
    if (!app.campaign) return;
    await request('/api/sculptor/review', {campaign_id: app.campaign.id, candidate_id: id, rating, preferred});
    await loadCampaign(app.campaign.id); renderGallery();
  });}
  function promptLabel(prompt, index = 0) {return prompt.label || prompt.category || (prompt.id ? prompt.id.slice(0, 12) : `Custom prompt ${index + 1}`);}
  function promptText(prompt) {return prompt.prompt ?? prompt.text ?? '';}
  function restoreBattery() {app.battery = clone(app.meta?.default_prompts || []).slice(0, 12); renderBattery(); updateControls();}
  function addPrompt(prompt) {
    if (app.battery.length >= 12) throw Error('The battery can contain at most 12 prompts.');
    if (prompt.id && app.battery.some(item => item.id === prompt.id)) return;
    app.battery.push(clone(prompt)); renderBattery(); updateControls();
  }
  function addCustom() {
    try {
      const values = refs.custom.value.split(/\n\s*\n/).map(value => value.trim()).filter(Boolean);
      if (!values.length) throw Error('Write at least one custom prompt.');
      if (app.battery.length + values.length > 12) throw Error('These prompts would exceed the 12-prompt battery limit.');
      for (const text of values) addPrompt({prompt: text, label: `Custom prompt ${app.battery.filter(p => !p.id).length + 1}`, corpus: 'custom', split: 'custom'});
      refs.custom.value = ''; notice();
    } catch (error) {notice(error.message, true);}
  }
  function renderBattery() {
    refs.batteryCount.textContent = `${app.battery.length} / 12 prompts`; refs.batteryList.replaceChildren();
    for (const [index, prompt] of app.battery.entries()) {
      const item = node('div', undefined, 'ps-battery-item'), head = row(node('span', String(index + 1).padStart(2, '0'), 'ps-prompt-number'), node('strong', promptLabel(prompt, index)));
      const remove = button('×', () => {app.battery.splice(index, 1); renderBattery(); updateControls();}); remove.setAttribute('aria-label', `Remove ${promptLabel(prompt, index)}`); head.append(remove);
      const details = node('details'); details.append(node('summary', promptText(prompt).slice(0, 140) + (promptText(prompt).length > 140 ? '…' : '')), node('p', promptText(prompt), 'ps-full-prompt'));
      item.append(head, node('span', `${human(prompt.corpus || 'custom')} · ${prompt.split === 'heldout' ? 'HELD-OUT VALIDATION' : human(prompt.split || 'custom')}`, `ps-source ${prompt.split === 'heldout' ? 'ps-heldout' : ''}`), details);
      refs.batteryList.append(item);
    }
    if (!app.battery.length) refs.batteryList.append(hint('Add a registered prompt or write your own.'));
    if (app.battery.some(item => item.split === 'heldout')) refs.batteryList.append(node('p', 'Held-out prompts are for fixed validation. Using their responses to select or tune a persona consumes that holdout; no independent validation claim remains.', 'ps-caution'));
  }
  async function searchPrompts() {
    if (!app.meta) return;
    const serial = ++app.searchSerial;
    try {
      const query = new URLSearchParams({corpus: refs.corpus.value, split: refs.split.value, q: refs.search.value.trim(), offset: app.promptOffset, limit: 6});
      const value = await request('/api/prompts?' + query); if (serial !== app.searchSerial) return;
      refs.promptResults.replaceChildren(); app.promptTotal = value.total;
      for (const prompt of value.rows) {
        const item = node('div', undefined, 'ps-bank-prompt'), add = button('+', () => {try {addPrompt(prompt);} catch (error) {notice(error.message, true);}});
        add.setAttribute('aria-label', `Add ${promptLabel(prompt)} to shared battery`);
        const description = node('div'); description.append(node('strong', promptLabel(prompt)), node('small', `${human(prompt.corpus)} · ${human(prompt.split)}`), node('p', prompt.preview || promptText(prompt).slice(0, 180)));
        item.append(description, add); refs.promptResults.append(item);
      }
      if (!value.rows.length) refs.promptResults.append(hint('No prompts match these filters.'));
      refs.promptPage.textContent = `${value.total ? app.promptOffset + 1 : 0}–${Math.min(app.promptOffset + value.rows.length, value.total)} / ${count(value.total)}`;
      refs.prev.disabled = app.promptOffset === 0; refs.next.disabled = app.promptOffset + value.rows.length >= value.total;
    } catch (error) {if (serial === app.searchSerial) refs.promptResults.replaceChildren(node('p', error.message, 'ps-error-text'));}
  }
  function updateControls() {
    if (!app.ready) return;
    const ready = Boolean(app.meta), dirtyBuild = app.build && app.builtRevision !== app.revision;
    const mutating = ['plan', 'preset', 'build'].some(key => app.busy.has(key));
    refs.interpret.disabled = !ready || mutating; refs.interpret.textContent = app.busy.has('plan') ? 'John is interpreting…' : 'Ask John to interpret';
    refs.buildButton.disabled = !ready || mutating; refs.buildButton.textContent = app.busy.has('build') ? 'Building recipes…' : `Build ${refs.variantCount.value} candidate recipes`;
    if (refs.presetLoad) {refs.presetLoad.disabled = !ready || mutating; refs.presetLoad.textContent = app.busy.has('preset') ? 'Loading recipe…' : 'Load preset & build candidates';}
    refs.buildStatus.textContent = dirtyBuild ? 'Settings changed. Rebuild before running this draft.' : app.build ? 'Recipes are ready. Select candidates and review the shared battery.' : 'Build uses your dials. Asking John to interpret a brief is optional.';
    refs.buildStatus.classList.toggle('ps-dirty', Boolean(dirtyBuild));
    const n = app.selected.size, p = app.battery.length;
    refs.runCount.textContent = `${n} candidates × ${p} prompts + ${p} baselines = ${(n + 1) * p} responses`;
    refs.runHint.textContent = `${app.capacity?.available || 0} reachable workers · up to 6 concurrent · busy / idle unknown`;
    refs.runButton.disabled = !ready || !app.build || dirtyBuild || !n || !p || !app.capacity?.available || app.busy.has('run') || mutating;
    refs.runButton.textContent = app.busy.has('run') ? 'Starting comparison…' : 'Run shared battery';
    refs.startSession.disabled = !app.jobs?.enabled || Boolean(app.session) || app.busy.has('session');
    refs.startSession.textContent = app.busy.has('session') ? 'Requesting session…' : app.session ? 'Session already requested' : 'Start studio session';
    refs.stopSession.disabled = !app.session || app.session.cancel_requested || app.session.state === 'stopping' || app.busy.has('stop-session');
    refs.stopSession.hidden = !app.session;
    refs.cancel.hidden = !app.campaign || terminal.has(app.campaign.state);
    refs.cancel.disabled = app.busy.has('cancel');
  }
  function runCampaign() {return action('run', async () => {
    if (!app.build || app.builtRevision !== app.revision) throw Error('Build the current settings before starting the battery.');
    const seed = Number(refs.seed.value);
    if (!refs.seed.value.trim() || !Number.isInteger(seed) || seed < 0 || seed > 4294967295) throw Error('Seed must be a whole number from 0 to 4294967295.');
    const candidates = app.build.candidates.filter(candidate => app.selected.has(candidate.id));
    if (!candidates.length || !app.battery.length) throw Error('Select at least one candidate and one prompt.');
    const result = await request('/api/sculptor/campaigns', {build_id: app.build.id, candidate_ids: candidates.map(candidate => candidate.id),
      prompts: app.battery.map(prompt => prompt.id ? {prompt_id: prompt.id} : {prompt: promptText(prompt)}), seed, max_tokens: Number(refs.budget.value), smoke: false});
    if (!result.id) throw Error('No saved campaign ID was returned.');
    app.pinned = candidates[0].id; app.promptIndex = 0; app.campaign = result;
    await loadCampaign(result.id); await loadHistory(); renderGallery(); renderResults(); scroll(refs.resultsSection); schedulePoll();
  });}
  async function loadHistory() {
    try {
      const value = await request('/api/sculptor/campaigns'); const selected = app.campaign?.id || '';
      refs.history.replaceChildren(node('option', 'Recent campaigns…')); refs.history.children[0].value = '';
      for (const campaign of value.campaigns || []) {const option = node('option', `${displayDate(campaign.created)} · ${human(campaign.state)} · ${campaign.id.slice(0, 8)}`); option.value = campaign.id; refs.history.append(option);}
      refs.history.value = selected;
    } catch (error) {refs.history.title = error.message;}
  }
  async function loadCampaign(id, choose = false) {
    if (choose) app.loadingCampaign = id;
    const serial = ++app.campaignSerial;
    try {
      const value = await request('/api/sculptor/campaign?' + new URLSearchParams({id})); if (serial !== app.campaignSerial) return;
      if (value.id !== id) throw Error('Campaign response does not match the selected record.');
      app.campaign = value;
      const historyOption = [...refs.history.options].find(option => option.value === id);
      if (historyOption) historyOption.textContent = `${displayDate(value.created)} · ${human(value.state)} · ${id.slice(0, 8)}`;
      if (choose && value.build) {
        app.build = value.build; app.plan = null; refs.brief.value = value.build.brief || '';
        for (const [key] of TRAITS) {app.dials[key] = Number(value.build.dials?.[key]) || 0; refs.dials[key].value = String(app.dials[key]);}
        refs.layers.value = (value.build.layers || []).join(', '); refs.strength.value = String(value.build.dose_budget ?? .35); refs.phase.value = value.build.phase || 'all';
        refs.variantCount.value = String(value.build.candidates.length); dirty(); app.builtRevision = app.revision;
        app.selected = new Set(value.input?.candidate_ids || value.build.candidates.map(candidate => candidate.id));
        app.battery = clone(value.prompts || value.input?.prompts || []); refs.seed.value = String(value.input?.seed ?? 7035); refs.budget.value = String(value.input?.max_tokens ?? 2048);
        app.promptIndex = 0; app.pinned = campaignCandidates().find(candidate => app.selected.has(candidate.id))?.id; renderGallery(); renderBattery(); renderPlanEvidence();
      }
      renderResults(); updateControls();
    } catch (error) {if (serial === app.campaignSerial) notice(error.message, true);}
    finally {if (choose && app.loadingCampaign === id) app.loadingCampaign = null;}
  }
  function outputPane(cell, title, cls) {
    const panel = node('article', undefined, `ps-answer-panel ${cls}`), head = row(node('h4', title), node('span', human(cell?.state || 'waiting'), 'ps-status'));
    panel.append(head);
    if (!cell) {panel.append(hint('Waiting for this arm to be recorded.')); return panel;}
    const output = cell.output;
    if (cell.error) panel.append(node('p', String(cell.error), 'ps-error-text'));
    if (output) {
      panel.append(node('div', output.answer || 'No answer text returned.', 'ps-answer'));
      const complete = output.complete === true;
      panel.append(node('p', `${complete ? 'Complete answer' : 'Incomplete / not a complete answer'} · ${human(output.finish_reason || 'unknown finish')} · ${count(output.usage?.completion_tokens)} output tokens`, complete ? 'ps-output-note' : 'ps-caution'));
      const provenance = node('details', undefined, 'ps-provenance'); provenance.append(node('summary', 'Runtime & provenance'));
      const facts = node('dl');
      for (const [label, value] of [['Endpoint', cell.endpoint_label || output.endpoint_label], ['Runtime identity', output.runtime_identity],
        ['Model identity', output.provenance?.model_identity], ['Request SHA-256', output.activation_artifact?.request_sha256 || output.provenance?.request_sha256],
        ['Activation artifact SHA-256', output.activation_artifact?.sha256], ['Condition', cls === 'ps-baseline' ? 'Unsteered; shared neutral system' : 'Saved candidate recipe; shared neutral system']]) {
        if (value) facts.append(node('dt', label), node('dd', String(value)));
      }
      provenance.append(facts);
      if (cell.batch_id) {const link = node('a', 'Open the complete batch record ↗'); link.href = '/api/batch?' + new URLSearchParams({id: cell.batch_id}); link.target = '_blank'; link.rel = 'noopener'; provenance.append(link);}
      panel.append(provenance);
    } else panel.append(hint(['queued', 'running'].includes(cell.state) ? 'The response will appear here when this request finishes.' : 'No completed response is available for this arm.'));
    return panel;
  }
  function renderResults() {
    refs.progress.replaceChildren(); refs.promptTabs.replaceChildren(); refs.comparison.replaceChildren();
    const campaign = app.campaign;
    if (!campaign) {refs.comparison.append(node('div', 'Run a shared battery to place the unsteered assistant beside a candidate. The baseline stays visible for every prompt.', 'ps-results-empty')); return;}
    const cells = campaign.cells || [], finished = cells.filter(cell => !['queued', 'running', 'waiting'].includes(cell.state)).length;
    refs.progress.append(node('strong', `${human(campaign.state)} · ${finished} / ${cells.length} responses recorded`),
      hint(`Campaign ${campaign.id.slice(0, 12)} · ${displayDate(campaign.created)} · seed ${campaign.input?.seed ?? '—'} · ${campaign.input?.max_tokens ?? '—'} tokens per response`));
    if (campaign.error) refs.progress.append(node('p', String(campaign.error), 'ps-error-text'));
    const bar = node('progress'); bar.max = Math.max(1, cells.length); bar.value = finished; bar.setAttribute('aria-label', 'Campaign progress'); refs.progress.append(bar);
    const prompts = campaign.prompts || campaign.input?.prompts || [];
    app.promptIndex = Math.max(0, Math.min(app.promptIndex, prompts.length - 1));
    for (const [index, prompt] of prompts.entries()) {
      const control = button(`${index + 1} · ${promptLabel(prompt, index)}`, () => {app.promptIndex = index; renderResults();}, 'ps-prompt-tab');
      control.classList.toggle('ps-active', index === app.promptIndex); control.setAttribute('aria-pressed', String(index === app.promptIndex)); refs.promptTabs.append(control);
    }
    const prompt = prompts[app.promptIndex];
    if (!prompt) {refs.comparison.append(hint('Waiting for the saved prompt snapshots.')); return;}
    const promptInfo = node('div', undefined, 'ps-comparison-prompt'); promptInfo.append(node('span', `${human(prompt.corpus || 'custom')} · ${human(prompt.split || 'custom')} · SHARED NEUTRAL SYSTEM`, 'ps-source'), node('p', promptText(prompt)));
    const candidates = campaignCandidates().filter(candidate => !campaign.input?.candidate_ids || campaign.input.candidate_ids.includes(candidate.id));
    let candidate = candidates.find(item => item.id === app.pinned) || candidates[0];
    if (candidate) app.pinned = candidate.id;
    const chooser = select(candidates.map(item => [item.id, item.name]), 'Pinned candidate for comparison'); chooser.value = candidate?.id || '';
    chooser.onchange = () => {app.pinned = chooser.value; renderResults(); renderGallery();};
    promptInfo.append(row(node('span', 'Compare with', 'ps-hint'), chooser)); refs.comparison.append(promptInfo);
    const baseline = cells.find(cell => cell.prompt_index === app.promptIndex && cell.candidate_id === 'baseline');
    const variant = cells.find(cell => cell.prompt_index === app.promptIndex && cell.candidate_id === candidate?.id);
    const columns = node('div', undefined, 'ps-answer-columns'); columns.append(outputPane(baseline, 'Unsteered assistant', 'ps-baseline'), outputPane(variant, candidate?.name || 'Candidate', 'ps-variant')); refs.comparison.append(columns);
    if (baseline && ['failed', 'incomplete', 'interrupted'].includes(baseline.state)) refs.comparison.append(node('p', 'This prompt has no valid complete baseline. Treat its comparison as incomplete.', 'ps-caution'));
  }
  function cancelCampaign() {return action('cancel', async () => {
    if (!app.campaign) return;
    await request('/api/sculptor/cancel', {id: app.campaign.id}); await loadCampaign(app.campaign.id);
    notice('Stop requested for future waves. In-flight responses may finish and remain in the record.');
  });}
  function startTime() {
    const instant = new Date(Date.now() + 2000), zone = 'America/Los_Angeles';
    try {
      const parts = Object.fromEntries(new Intl.DateTimeFormat('en-CA', {timeZone: zone, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23', timeZoneName: 'longOffset'}).formatToParts(instant).map(part => [part.type, part.value]));
      const offset = parts.timeZoneName.replace('GMT', '') || '+00:00';
      if (!/^[+-]\d{2}:\d{2}$/.test(offset)) throw Error('Unknown offset');
      return {start_at: `${parts.year}-${parts.month}-${parts.day}T${parts.hour}:${parts.minute}:${parts.second}${offset}`, timezone: zone};
    } catch (_) {return {start_at: instant.toISOString(), timezone: 'UTC'};}
  }
  function startSession() {return action('session', async () => {
    await refreshFleet();
    if (app.session) {notice('A studio session is already queued or running. Its status is shown above.'); return;}
    if (app.capacity?.available) {notice('Verified workers are already reachable. You can use the existing fleet.'); return;}
    const result = await request('/api/jobs', {kind: 'studio_session', name: 'Persona studio session', ...startTime(),
      duration_minutes: Number(refs.sessionMinutes.value), target_entries: 1, max_entries: 1, seed: 7035});
    app.session = result; notice('Studio session requested. Cold model preparation can take tens of minutes; session time includes startup.'); await refreshFleet();
  });}
  function stopSession() {return action('stop-session', async () => {
    if (!app.session || app.session.kind !== 'studio_session') return;
    await request('/api/job/cancel', {id: app.session.id}); await refreshFleet(); notice('Stop requested for this studio session’s own resources.');
  });}
  async function refreshFleet() {
    const values = await Promise.allSettled([request('/api/capacity'), request('/api/jobs')]);
    app.capacity = values[0].status === 'fulfilled' ? values[0].value : null;
    if (values[1].status === 'fulfilled') {
      app.jobs = values[1].value;
      app.session = (app.jobs.jobs || []).find(job => {
        const start = job.start_epoch ?? (job.start_at ? Date.parse(job.start_at) / 1000 : NaN);
        return job.kind === 'studio_session' && ['queued', 'running', 'stopping'].includes(job.state) &&
          (job.state !== 'queued' || !Number.isFinite(start) || start <= Date.now() / 1000 + 60);
      }) || null;
    } else {app.jobs = null; app.session = null;}
    const capacity = app.capacity;
    refs.capacity.textContent = capacity ? `${count(capacity.available)} reachable / ${count(capacity.configured)} configured graphics cards` : 'Fleet status unavailable';
    refs.cards.replaceChildren(); for (const card of capacity?.cards || []) {const value = node('span', card.label, 'ps-card-light'); value.classList.toggle('ps-reachable', card.available === true); value.title = card.available ? 'Reachable and identity-verified; busy / idle unknown' : card.error || 'Not reachable'; refs.cards.append(value);}
    refs.sessionStatus.textContent = app.session ? `${human(app.session.state)} · ${app.session.name} · ends ${displayDate(app.session.deadline)}. Cold model preparation can take tens of minutes.` : capacity?.available ? 'Reachability is verified. Busy / idle is unknown; experiments share the existing worker pool.' : app.jobs?.enabled ? 'No diagnostic workers are reachable. Start a studio session to request the leased fleet.' : 'The local scheduler is not enabled or unavailable. Existing reachable workers can still be used.';
    refs.startSession.hidden = Boolean(capacity?.available && !app.session); refs.sessionMinutes.hidden = Boolean(app.session || capacity?.available);
    refs.startupHint.hidden = Boolean(capacity?.available && !app.session);
    const limit = capacity?.context_tokens; const option4096 = [...refs.budget.children].find(option => option.value === '4096');
    if (option4096) option4096.disabled = Number.isFinite(limit) && limit <= 4096;
    if (refs.budget.value === '4096' && option4096?.disabled) refs.budget.value = '3072';
    updateControls();
  }
  async function loadMeta() {
    try {
      const value = await request('/api/sculptor'); app.meta = value; app.capacity = value.capacity || app.capacity;
      const defaults = value.defaults || {};
      refs.layers.value = (defaults.layers || []).join(', '); refs.strength.value = String(defaults.dose_budget ?? .35);
      refs.variantCount.value = String(defaults.count || Math.min(6, value.capacity?.configured || 6)); refs.phase.value = defaults.phase || 'all';
      refs.seed.value = String(defaults.seed ?? 7035); refs.budget.value = String(defaults.max_tokens || 2048);
      restoreBattery(); renderPresets(); renderProfile(); searchPrompts(); await loadHistory(); notice();
    } catch (error) {notice(`Persona studio is not ready: ${error.message}`, true);}
    updateControls();
  }
  function schedulePoll() {
    clearTimeout(pollTimer); pollTimer = setTimeout(async () => {
      if (!document.hidden && !refs.root.classList.contains('hidden')) {
        await refreshFleet();
        if (app.campaign && !app.loadingCampaign && !terminal.has(app.campaign.state)) await loadCampaign(app.campaign.id);
      }
      schedulePoll();
    }, app.campaign && !terminal.has(app.campaign.state) ? 2500 : 10000);
  }
  async function open() {
    if (!ensure()) return;
    if (!app.meta) await loadMeta();
    await refreshFleet(); schedulePoll();
  }
  window.PersonaSculptor = {open};
})();
