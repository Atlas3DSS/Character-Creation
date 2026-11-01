/* Response comparison library: the compare view opens prefilled with − / 0 / + runs that already exist
   (studio comparisons, Neuropedia steering cards paired with their sign-flipped twin, the identity sweep).
   Answers are shown verbatim; every column names its source; missing columns say so and can be queued. */
(() => {
  'use strict';
  const ROOT = '/api/response-study/compare-library';
  const SOURCES = {card: 'Card', studio: 'Studio run', identity: 'Identity sweep', atlas: 'Atlas sweep'};
  const COLS = ['−', '0', '+'];
  let built = false, selected = null, item = null, step = 0, offset = 0, timer = null, seq = 0, live = false;
  const $ = id => document.getElementById(id);
  const node = (tag, text, cls) => {const n = document.createElement(tag); if (text != null) n.textContent = text; if (cls) n.className = cls; return n;};
  async function get(url, body) {
    const r = await fetch(url, body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {});
    let v = null; try { v = await r.json(); } catch (_) {}
    if (!r.ok) throw Error((v && v.error) || (r.status === 403 ? 'Not available on this copy of the studio' : r.statusText));
    return v;
  }
  const when = ts => ts ? new Date(ts * 1000).toLocaleString([], {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'}) : '';

  function build() {
    if (built) return; built = true;
    const view = $('compare-view'), columns = $('columns');
    const shell = node('div', null, 'cl-shell'), side = node('aside', null, 'cl-picker'), main = node('div', null, 'cl-main');
    side.setAttribute('aria-label', 'Existing comparisons');
    const search = node('input'); search.type = 'search'; search.id = 'cl-q'; search.placeholder = 'Neuron (L35N3602), card, character or prompt';
    search.setAttribute('aria-label', 'Search comparisons');
    const filters = node('div', null, 'cl-filters');
    const full = node('label'); const fullBox = node('input'); fullBox.type = 'checkbox'; fullBox.id = 'cl-full'; full.append(fullBox, ' All three columns');
    const kind = node('select'); kind.id = 'cl-kind'; kind.setAttribute('aria-label', 'Recipe size');
    for (const [v, t] of [['all', 'Any recipe'], ['single', 'Single neuron'], ['multi', 'Multi-push']]) kind.append(new Option(t, v));
    const source = node('select'); source.id = 'cl-source'; source.setAttribute('aria-label', 'Source');
    for (const [v, t] of [['all', 'All sources'], ['card', 'Cards'], ['studio', 'Studio runs'], ['identity', 'Identity sweep'], ['atlas', 'Atlas sweep']]) source.append(new Option(t, v));
    filters.append(full, kind, source);
    const counts = node('p', 'Loading existing comparisons…', 'hint'); counts.id = 'cl-counts'; counts.setAttribute('role', 'status');
    const list = node('ol', null, 'cl-list'); list.id = 'cl-list';
    const more = node('button', 'Show more', 'link'); more.type = 'button'; more.id = 'cl-more'; more.hidden = true;
    const runNew = node('button', 'Run a new comparison from the Recipe panel', 'link'); runNew.type = 'button';
    runNew.onclick = () => { if (typeof tab === 'function') tab('map'); try { openPanel('recipe'); $('scene-block').open = true; $('compare').focus(); } catch (_) {} };
    side.append(node('h2', 'Existing comparisons'), search, filters, counts, list, more, runNew);
    const head = node('div', null, 'cl-head'); head.id = 'cl-head';
    const steps = node('div', null, 'cl-steps'); steps.id = 'cl-steps'; steps.setAttribute('role', 'tablist'); steps.setAttribute('aria-label', 'Question');
    const prompt = node('p', null, 'cl-prompt'); prompt.id = 'cl-prompt';
    columns.parentNode.insertBefore(shell, columns);
    main.append(head, steps, prompt, columns);
    shell.append(side, main);
    for (const col of columns.children) {
      const meta = node('div', null, 'cl-meta'); col.insertBefore(meta, col.querySelector('.answer'));
    }
    search.oninput = () => { clearTimeout(timer); timer = setTimeout(() => load(true), 220); };
    fullBox.onchange = kind.onchange = source.onchange = () => load(true);
    more.onclick = () => load(false);
    view.dataset.library = 'on';
  }

  async function load(reset) {
    const mine = ++seq;
    if (reset) offset = 0;
    const q = new URLSearchParams({q: $('cl-q').value, full: $('cl-full').checked ? '1' : '0', kind: $('cl-kind').value,
      source: $('cl-source').value, offset: String(offset), limit: '60'});
    let data;
    try { data = await get(ROOT + '?' + q); }
    catch (e) { $('cl-counts').textContent = 'Comparison library unavailable: ' + e.message; return null; }
    if (mine !== seq) return null;
    const list = $('cl-list');
    if (reset) list.replaceChildren();
    for (const row of data.items) list.append(entry(row));
    offset += data.items.length;
    $('cl-more').hidden = offset >= data.total;
    const c = data.counts, s = c.sources || {};
    const NAMES = {card: 'cards', studio: 'studio runs', identity: 'identity sweep', atlas: 'atlas sweep'};
    const part = k => s[k] ? `${NAMES[k]} ${s[k].full.toLocaleString()}/${s[k].items.toLocaleString()}` : null;
    $('cl-counts').textContent = `${data.total.toLocaleString()} shown by this search · ${c.full.toLocaleString()} of ${c.all.toLocaleString()} have all three columns (` +
      ['card', 'studio', 'identity', 'atlas'].map(part).filter(Boolean).join(', ') + ')' + (data.atlas_sweep ? '' : ' · atlas sweep not started');
    if (!list.children.length) list.append(node('li', 'No comparison matches. Clear the search or filters.', 'hint'));
    mark();
    return data;
  }

  function dots(cols) {
    const box = node('span', null, 'cl-dots');
    (cols || [0, 0, 0]).forEach((on, i) => { const d = node('span', COLS[i], 'cl-dot c' + i + (on ? ' on' : '')); d.title = (on ? 'run' : 'not run yet'); box.append(d); });
    box.setAttribute('aria-label', (cols || []).map((on, i) => COLS[i] + (on ? ' run' : ' missing')).join(', '));
    return box;
  }

  function entry(row) {
    const li = node('li'), b = node('button', null, 'cl-item'); b.type = 'button'; b.dataset.key = row.key;
    const top = node('span', null, 'cl-top');
    const title = row.source === 'studio' ? row.prompt : row.source === 'identity' || row.source === 'atlas' ? row.neurons[0] + ' · “' + row.prompt + '”' : (row.title || row.file);
    top.append(node('strong', title), dots(row.cols));
    const sub = node('span', row.recipe.length > 90 ? row.recipe.slice(0, 88) + '…' : row.recipe, 'cl-recipe');
    const bits = [SOURCES[row.source] || row.source];
    if (row.character) bits.push(row.character);
    if (row.source === 'card') bits.push(row.questions + ' questions'); else if (row.source !== 'studio') bits.push(row.questions + ' dose' + (row.questions === 1 ? '' : 's'));
    if (row.ts) bits.push(when(row.ts));
    if (!row.full) bits.push(row.reverse_queued ? '− queued' : row.source === 'studio' ? row.state : '− not run yet');
    b.append(top, sub, node('span', bits.join(' · '), 'cl-sub'));
    b.onclick = () => select(row.key);
    li.append(b); return li;
  }

  function mark() {
    document.querySelectorAll('#cl-list .cl-item').forEach(b => { const on = b.dataset.key === selected; b.classList.toggle('on', on); if (on) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current'); });
  }

  async function select(key) {
    selected = key; live = false; mark();
    try { clearTimeout(poll); } catch (_) {}   // a running studio comparison stops repainting the columns
    if (key.startsWith('studio:') && typeof showComparison === 'function') {
      const data = await get(ROOT + '/item?key=' + encodeURIComponent(key)).catch(() => null);
      if (data && data.state !== 'complete') { showComparison(key.slice(7)); return; }
      if (data) { item = data; step = 0; render(); }
      return;
    }
    $('comparison-status').textContent = 'Loading…';
    try { item = await get(ROOT + '/item?key=' + encodeURIComponent(key)); }
    catch (e) { $('comparison-status').textContent = e.message; return; }
    if (selected !== key) return;
    step = 0; render();
  }

  function render() {
    if (!item) return;
    const title = item.source === 'card' ? `${item.title}${item.character ? ' · ' + item.character : ''}` : item.title;
    $('comparison-title').textContent = title;
    $('comparison-status').textContent = `${SOURCES[item.source]} · + ${item.recipe} · − ${item.reverse_recipe}` + (item.settings ? ' · ' + item.settings : '');
    const head = $('cl-head'); head.replaceChildren();
    if (item.source === 'card') {
      head.append(node('span', item.verdict ? 'verdict ' + item.verdict : '', 'pill'));
      if (item.reverse_title) head.append(node('span', '− card: ' + item.reverse_title, 'pill'));
      if (item.summary) head.append(node('span', item.summary, 'cl-summary'));
    }
    const bar = $('cl-steps'); bar.replaceChildren(); bar.hidden = item.steps.length < 2;
    item.steps.forEach((s, i) => {
      const b = node('button', s.label); b.type = 'button'; b.setAttribute('role', 'tab'); b.setAttribute('aria-selected', String(i === step));
      b.title = s.prompt; if (i === step) b.classList.add('on');
      const real = s.cells.filter(c => !c.missing && !c.error).length; if (real < 3) b.classList.add('partial');
      b.onclick = () => { step = i; render(); }; bar.append(b);
    });
    const s = item.steps[step];
    $('cl-prompt').textContent = s ? (item.steps.length > 1 ? `${s.label} · ` : 'Prompt · ') + s.prompt : '';
    const columns = $('columns').children;
    for (let i = 0; i < 3; i++) fill(columns[i], s ? s.cells[i] : {missing: true}, i, s);
  }

  function fill(col, cell, i, s) {
    const meta = col.querySelector('.cl-meta'), answer = col.querySelector('.answer');
    meta.replaceChildren(); answer.replaceChildren(); col.classList.toggle('cl-missing', !!cell.missing);
    if (cell.missing) {
      meta.append(node('span', 'not run yet', 'cl-badge miss'));
      answer.append(node('p', i === 1 ? 'No unsteered run exists for this prompt.' : 'No run with this sign exists yet.', 'dim'));
      if (i !== 1) queueButton(answer, i, s);
      return;
    }
    if (cell.error) { meta.append(node('span', 'error', 'cl-badge miss')); answer.append(node('p', cell.error)); }
    else {
      meta.append(node('span', 'real run', 'cl-badge'));
      answer.append(node('div', cell.answer === '' ? '(empty answer)' : cell.answer));
      if (cell.reasoning) { const d = node('details'); d.append(node('summary', 'Thinking'), node('pre', cell.reasoning)); answer.append(d); }
      if (cell.complete === false) answer.append(node('p', 'Incomplete response — excluded from behavioral claims', 'hint'));
    }
    meta.append(node('small', cell.source, 'cl-src'));
  }

  function queueButton(host, i, s) {
    if (!item || item.source === 'studio') return;
    if (item.source === 'card' && i === 0 && item.reverse_queued) {
      host.append(node('p', `Queued on Neuropedia (${item.reverse_queued}). It fills this column when the card worker finishes it.`, 'hint')); return;
    }
    if (item.source === 'card' && i !== 0) return;
    let body, label;
    if (item.source === 'card') { body = {key: item.key}; label = `Queue ${item.reverse_recipe.length > 60 ? 'the reversed recipe' : item.reverse_recipe} as a card`; }
    else {
      const level = Number(String(s.label).replace('±', '')); const dose = i === 0 ? -level : level;
      body = {key: item.key, dose}; label = `Queue L${item.layer} N${item.neuron} ${dose > 0 ? '+' : ''}${dose} as a card`;
    }
    const b = node('button', label, 'primary cl-queue'); b.type = 'button';
    const note = node('p', item.source === 'card' ? 'Goes to Neuropedia’s card queue (POST /api/cards/request): the same 12-question battery as the + card; it pairs here when done.'
      : 'Goes to Neuropedia’s card queue (POST /api/cards/request) as a 12-question card; it shows up under Cards, not inside this sweep.', 'hint');
    b.onclick = async () => {
      b.disabled = true; b.textContent = 'Queueing…';
      try { const r = await get(ROOT + '/queue-reverse', body); b.textContent = r.already ? `Already queued (${r.id})` : `Queued (${r.id})`; if (item.source === 'card') item.reverse_queued = r.id; }
      catch (e) { b.disabled = false; b.textContent = label; note.textContent = e.message; }
    };
    host.append(b, note);
  }

  async function open() {
    build();
    if (selected || live) return;
    const data = await load(true);
    if (!selected && !live && data && data.default) select(data.default);
    else if (!selected && !live && data && data.items.length) select(data.items[0].key);
  }

  /* called by showComparison after it paints a studio run (live or finished) */
  function shown(id, data) {
    build(); live = true; selected = 'studio:' + id; item = null; mark();
    $('cl-head').replaceChildren(); $('cl-steps').replaceChildren(); $('cl-steps').hidden = true;
    const recipe = (data.input.pushes || []).map(p => p.layer != null ? `L${p.layer} N${p.neuron} ${p.dose > 0 ? '+' : ''}${p.dose}` : (p.name || p.kind || 'push')).join(' + ');
    $('cl-prompt').textContent = 'Studio comparison · ' + (recipe || 'no pushes') + ' · ' + data.input.max_tokens + ' tokens';
    const columns = $('columns').children, out = Array.isArray(data.output) ? data.output : null;
    for (let i = 0; i < 3; i++) {
      const meta = columns[i].querySelector('.cl-meta'); if (!meta) continue; meta.replaceChildren();
      const row = out && out[i]; columns[i].classList.toggle('cl-missing', !row);
      meta.append(node('span', row ? (row.error ? 'error' : 'real run') : data.state, row && !row.error ? 'cl-badge' : 'cl-badge miss'));
      meta.append(node('small', `studio comparison ${id.slice(0, 8)} · sign ${[-1, 0, 1][i]} × recipe`, 'cl-src'));
    }
  }

  window.PersonaCompareLibrary = {open, shown, select};
})();
