/* Signed co-activity of measured candidates; no inferred wiring or temporal phase. */
(() => {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const definitions = [
    {key: 'positive', title: 'Rise together', short: 'Positive', detail: 'Positive correlation · r > +0.20'},
    {key: 'negative', title: 'Move oppositely', short: 'Negative', detail: 'Negative correlation · r < −0.20'},
    {key: 'weak', title: 'Weak co-activity', short: 'Weak', detail: 'Closest to zero · |r| ≤ 0.20'}
  ];
  const app = {target: null, corpus: 'all', scope: 'all', data: null, loading: false, error: '', serial: 0};
  const refs = {};
  let controller, returnFocus;
  const coordinate = value => `L${value.layer} · N${value.neuron}`;
  const count = value => Number.isInteger(value) ? value.toLocaleString() : '—';
  const signed = value => `${value < 0 ? '−' : '+'}${Math.abs(value).toFixed(3)}`;
  function node(tag, text, cls) {
    const result = document.createElement(tag);
    if (text !== undefined) result.textContent = text;
    if (cls) result.className = cls;
    return result;
  }
  function button(text, action, cls = '') {
    const result = node('button', text, cls);
    result.type = 'button'; result.onclick = action; return result;
  }
  function svgNode(tag, attributes, text) {
    const result = document.createElementNS(NS, tag);
    Object.entries(attributes || {}).forEach(([key, value]) => result.setAttribute(key, String(value)));
    if (text !== undefined) result.textContent = text;
    return result;
  }
  function filter(label, values, selected, action) {
    const wrap = node('label', undefined, 'pn-filter'), control = node('select');
    for (const [value, text] of values) {
      const option = node('option', text); option.value = value; control.append(option);
    }
    control.value = selected; control.onchange = () => action(control.value);
    wrap.append(node('span', label), control); return wrap;
  }
  function mapped(item) {
    const collection = typeof state !== 'undefined' ? state?.map?.neurons : window.state?.map?.neurons;
    return collection?.find(value => value.layer === item.layer && value.neuron === item.neuron);
  }
  function navigate(item) {
    const target = mapped(item);
    const select = typeof selectNeuron === 'function' ? selectNeuron : window.selectNeuron;
    if (!target || typeof select !== 'function') {
      refs.announcement.textContent = 'This candidate is not in the current map.'; return;
    }
    // The map owns selection and calls PersonaNeighbors.select once through its normal hook.
    select(target);
    if (refs.dialog.open) refs.title.focus({preventScroll: true});
  }
  function ensure() {
    if (refs.dialog) return;
    refs.compact = node('section', undefined, 'pn-compact');
    refs.compact.setAttribute('aria-label', 'Neuron neighborhood');
    const compactHead = node('div', undefined, 'pn-compact-head');
    compactHead.append(node('h3', 'Neuron neighborhood'), node('span', 'DISCOVERY', 'pn-eyebrow'));
    refs.compactStatus = node('p', undefined, 'pn-small');
    refs.spark = node('div', undefined, 'pn-spark');
    refs.compactBody = node('div', undefined, 'pn-compact-body');
    refs.open = button('Explore co-activity ↗', open, 'pn-open');
    refs.compact.append(compactHead, refs.compactStatus, refs.spark, refs.compactBody, refs.open,
      node('p', 'Final prompt token · across prompts · not causal wiring', 'pn-footnote'));

    refs.dialog = node('dialog', undefined, 'pn-dialog');
    refs.dialog.setAttribute('aria-labelledby', 'pn-title');
    refs.dialog.setAttribute('aria-describedby', 'pn-description');
    const head = node('div', undefined, 'pn-head'), heading = node('div');
    heading.append(node('p', 'MEASURED NEURON NEIGHBORHOOD', 'pn-eyebrow'));
    refs.title = node('h2'); refs.title.id = 'pn-title'; refs.title.tabIndex = -1;
    const description = node('p', 'Signed co-activity across original, unsteered discovery prompts. Each value is measured at the final prompt token.', 'pn-description');
    description.id = 'pn-description'; heading.append(refs.title, description);
    refs.close = button('×', close, 'pn-close'); refs.close.setAttribute('aria-label', 'Close neuron neighborhood');
    head.append(heading, refs.close);
    const controls = node('div', undefined, 'pn-controls');
    controls.append(filter('Corpus', [['all', 'All corpora'], ['big_five', 'Big Five'], ['scotus', 'SCOTUS'], ['skippy', 'Skippy books']], app.corpus,
      value => {app.corpus = value; load();}),
    filter('Layers', [['all', 'All layers'], ['same_layer', 'Same layer'], ['cross_layer', 'Other layers']], app.scope,
      value => {app.scope = value; load();}),
    button('Plan adaptive search', () => {
      if (window.PersonaJobs?.fromNeuron) window.PersonaJobs.fromNeuron({...app.target});
      else refs.announcement.textContent = 'The Jobs & schedules workspace is not loaded yet.';
    }, 'pn-plan'),
    button('Refresh', load, 'pn-refresh'));
    refs.summary = node('div', undefined, 'pn-summary');
    refs.warning = node('p', undefined, 'pn-warning');
    refs.content = node('div', undefined, 'pn-content');
    refs.announcement = node('p', undefined, 'pn-sr-only');
    refs.announcement.setAttribute('role', 'status'); refs.announcement.setAttribute('aria-live', 'polite');
    const footer = node('div', undefined, 'pn-footer');
    footer.append(node('strong', 'Co-activity is not connectivity.'), node('span',
      ' Continuous activations, not binary firing. A positive r means these values tend to rise together; a negative r means they tend to move oppositely. Neither measures timing, temporal phase, direct wiring, or causal influence.'));
    refs.provenance = node('details', undefined, 'pn-provenance');
    refs.provenance.append(node('summary', 'Source and limits'));
    refs.provenanceBody = node('div'); refs.provenance.append(refs.provenanceBody);
    refs.dialog.append(head, controls, refs.summary, refs.warning, refs.content, footer, refs.provenance, refs.announcement);
    document.body.append(refs.dialog);
    refs.dialog.addEventListener('cancel', event => {event.preventDefault(); close();});
    refs.dialog.addEventListener('click', event => {
      if (event.target !== refs.dialog) return;
      const bounds = refs.dialog.getBoundingClientRect();
      if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) close();
    });
    refs.dialog.addEventListener('close', () => {
      const destination = returnFocus?.isConnected ? returnFocus : refs.open.isConnected ? refs.open : null;
      destination?.focus({preventScroll: true});
    });
  }
  function open() {
    ensure(); if (!app.target || refs.dialog.open) return;
    returnFocus = document.activeElement;
    refs.dialog.showModal(); refs.close.focus({preventScroll: true});
  }
  function close() {if (refs.dialog?.open) refs.dialog.close();}
  function select(value) {
    if (!value || !Number.isInteger(value.layer) || value.layer < 0 || !Number.isInteger(value.neuron) || value.neuron < 0) return;
    ensure();
    app.target = {layer: value.layer, neuron: value.neuron};
    const unit = document.getElementById('unit');
    if (unit) unit.insertBefore(refs.compact, unit.children[2] || null);
    load();
  }
  function validate(value, target) {
    if (!value || value.target?.layer !== target.layer || value.target?.neuron !== target.neuron || typeof value.available !== 'boolean') {
      throw Error('The neighborhood response does not match the selected neuron.');
    }
    if (value.available) {
      if (!Number.isInteger(value.source?.n_samples) || value.source.n_samples < 8 || value.source.split !== 'discovery') {
        throw Error('The neighborhood source lacks the required discovery samples.');
      }
      for (const {key} of definitions) {
        if (!Array.isArray(value.groups?.[key])) throw Error('The neighborhood response has an invalid group.');
        for (const item of value.groups[key]) {
          if (!Number.isInteger(item.layer) || item.layer < 0 || !Number.isInteger(item.neuron) || item.neuron < 0 ||
              !Number.isFinite(item.r) || Math.abs(item.r) > 1 || !Number.isInteger(item.n) || item.n < 8) {
            throw Error('The neighborhood response contains an invalid measurement.');
          }
        }
      }
    }
    return value;
  }
  async function load() {
    if (!app.target) return;
    const serial = ++app.serial, target = {...app.target};
    controller?.abort(); controller = new AbortController();
    app.data = null; app.error = ''; app.loading = true; render();
    const query = new URLSearchParams({layer: target.layer, neuron: target.neuron, corpus: app.corpus, scope: app.scope, limit: 12});
    try {
      const response = await fetch(`/api/neighbors?${query}`, {signal: controller.signal});
      const value = await response.json();
      if (!response.ok) throw Error(value.error || `Neighborhood request failed (${response.status}).`);
      if (serial !== app.serial) return;
      app.data = validate(value, target);
    } catch (error) {
      if (serial !== app.serial || error.name === 'AbortError') return;
      app.error = error.message || 'Unable to load the neighborhood.';
    } finally {
      if (serial === app.serial) {app.loading = false; render();}
    }
  }
  function statusText() {
    if (app.loading) return 'Reading measured co-activity…';
    if (app.error) return app.error;
    if (!app.data?.available) return app.data?.message || 'No measured neighborhood is available yet.';
    return `${count(app.data.source.n_samples)} prompts · ${count(app.data.source.candidates)} measured candidates`;
  }
  function render() {
    if (!app.target) return;
    const data = app.data, source = data?.source;
    refs.title.textContent = coordinate(app.target);
    refs.compactStatus.textContent = statusText();
    refs.compact.setAttribute('aria-busy', String(app.loading));
    refs.compactBody.replaceChildren(); refs.spark.replaceChildren();
    refs.summary.replaceChildren(); refs.content.replaceChildren(); refs.provenanceBody.replaceChildren();
    refs.content.setAttribute('aria-busy', String(app.loading));
    refs.warning.hidden = true;
    if (source) {
      for (const [value, label] of [[count(source.n_samples), 'prompts in this filter'], [count(source.candidates), 'measured candidates'],
        [Number.isInteger(source.eligible_candidates) ? count(source.eligible_candidates) : '—', 'eligible neighbors']]) {
        const stat = node('div', undefined, 'pn-stat'); stat.append(node('strong', value), node('span', label)); refs.summary.append(stat);
      }
      renderProvenance(source);
      const warnings = [];
      if (app.corpus === 'all') warnings.push('Pooled correlations can reflect differences between corpora. Compare within a corpus. Most observations are Skippy book prompts.');
      if (Number.isInteger(source.n_samples) && source.n_samples < 30) warnings.push(`Only ${source.n_samples} prompts in this filter: correlations can be unstable and do not establish a general relationship.`);
      if (warnings.length) {refs.warning.hidden = false; refs.warning.textContent = warnings.join(' ');}
    }
    if (app.loading || app.error || !data?.available) {
      const message = node('div', undefined, 'pn-empty');
      message.append(node('strong', app.loading ? 'Reading the candidate set' : 'Neighborhood unavailable'), node('p', statusText()));
      if (!app.loading) message.append(node('p', 'At least eight varying discovery observations are required. No links are inferred for missing or constant measurements.', 'pn-small'), button('Try again', load));
      refs.content.append(message);
      refs.compactBody.append(node('p', app.loading ? 'A signed view will appear when the measurements are ready.' : 'Open the neighborhood for source details and corpus filters.', 'pn-small'));
    } else {
      refs.spark.append(network(data, true));
      const chips = node('div', undefined, 'pn-chips');
      for (const definition of definitions) {
        const first = data.groups[definition.key][0];
        chips.append(node('span', `${definition.short}: ${first ? 'r ' + signed(first.r) : 'none'}`, `pn-chip pn-${definition.key}`));
      }
      refs.compactBody.append(chips);
      if (app.corpus === 'all') refs.compactBody.append(node('p', 'Pooled correlations can reflect differences between corpora. Compare within a corpus.', 'pn-compact-warning'));
      else if (source.n_samples < 30) refs.compactBody.append(node('p', `Only ${source.n_samples} prompts: treat these correlations as exploratory.`, 'pn-compact-warning'));
      if (app.corpus !== 'all' || app.scope !== 'all') refs.compactBody.append(node('p',
        `${app.corpus === 'all' ? 'All corpora' : app.corpus === 'big_five' ? 'Big Five' : app.corpus === 'scotus' ? 'SCOTUS' : 'Skippy books'} · ${app.scope === 'all' ? 'all layers' : app.scope === 'same_layer' ? 'same layer' : 'other layers'}`, 'pn-small'));
      const figure = node('figure', undefined, 'pn-network'), scroll = node('div', undefined, 'pn-network-scroll');
      scroll.append(network(data, false)); figure.append(scroll, node('figcaption',
        'A few ranked candidates per group. Lines show signed correlations to the selected neuron; their placement is illustrative. This is a measured candidate set, not the full model connectome.'));
      refs.content.append(figure);
      const lists = node('div', undefined, 'pn-lists');
      for (const definition of definitions) lists.append(rankedList(data.groups[definition.key], definition));
      refs.content.append(lists);
      if (data.status === 'no_eligible_neighbors') refs.content.append(node('p', data.message, 'pn-warning'));
    }
    if (!app.loading) refs.announcement.textContent = `${coordinate(app.target)}. ${statusText()}`;
  }
  function rankedList(entries, definition) {
    const section = node('section', undefined, `pn-ranked pn-${definition.key}`), head = node('div', undefined, 'pn-rank-head');
    head.append(node('h3', definition.title), node('span', `${entries.length} shown`, 'pn-small'));
    section.append(head, node('p', definition.detail, 'pn-small'));
    if (!entries.length) {section.append(node('p', 'No eligible candidates in this range.', 'pn-list-empty')); return section;}
    const list = node('ol');
    entries.forEach((item, index) => {
      const row = node('li'), select = button('', () => navigate(item), 'pn-neighbor');
      select.setAttribute('aria-label', `${coordinate(item)}, correlation ${signed(item.r)}, ${item.n} prompts. Select neuron.`);
      const rank = node('span', String(index + 1), 'pn-rank'), name = node('span', undefined, 'pn-neighbor-name');
      name.append(node('strong', coordinate(item)), node('small', `${item.strength} · n = ${count(item.n)}`));
      const statistic = node('span', `r ${signed(item.r)}`, 'pn-r');
      const bar = node('span', undefined, 'pn-bar'); bar.setAttribute('aria-hidden', 'true');
      const fill = node('span'); fill.style.width = `${Math.abs(item.r) * 100}%`; bar.append(fill);
      select.append(rank, name, statistic, bar);
      if (!mapped(item)) {select.disabled = true; select.title = 'This candidate is not in the current map.';}
      row.append(select); list.append(row);
    });
    section.append(list); return section;
  }
  function network(data, compact) {
    const width = compact ? 320 : 900, height = compact ? 106 : 354;
    const svg = svgNode('svg', {viewBox: `0 0 ${width} ${height}`, class: compact ? 'pn-spark-svg' : 'pn-network-svg',
      role: compact ? 'img' : 'group', 'aria-label': compact ? 'Signed co-activity preview. Open the neighborhood for the ranked measurements.' : 'Signed co-activity of selected candidates. The same measurements are available in the ranked lists below.'});
    if (compact) svg.setAttribute('aria-hidden', 'true');
    const cx = width / 2, cy = compact ? 46 : 145, links = svgNode('g', {class: 'pn-links'}), vertices = svgNode('g');
    svg.append(links, vertices);
    const perGroup = compact ? 2 : 4;
    for (const [groupIndex, definition] of definitions.entries()) {
      const items = data.groups[definition.key].slice(0, perGroup);
      if (!compact) {
        const labelX = groupIndex === 0 ? 122 : groupIndex === 1 ? 778 : 450;
        vertices.append(svgNode('text', {x: labelX, y: groupIndex === 2 ? 253 : 24, class: `pn-network-heading pn-${definition.key}`, 'text-anchor': 'middle'}, definition.title));
      }
      items.forEach((item, index) => {
        let x, y;
        if (compact) {
          x = groupIndex === 0 ? 38 + index * 37 : groupIndex === 1 ? width - 38 - index * 37 : 132 + index * 56;
          y = groupIndex === 2 ? 91 : 22 + index * 35;
        } else if (groupIndex === 2) {
          x = 162 + (index + (4 - items.length) / 2) * 192; y = 291;
        } else {
          x = groupIndex === 0 ? 77 + (index % 2) * 132 : 823 - (index % 2) * 132;
          y = 69 + Math.floor(index / 2) * 111;
        }
        links.append(svgNode('line', {x1: cx, y1: cy, x2: x, y2: y, class: `pn-link pn-${definition.key}`,
          'stroke-width': compact ? 1 + Math.abs(item.r) * 1.6 : 1 + Math.abs(item.r) * 3,
          opacity: .28 + Math.abs(item.r) * .55}));
        const group = svgNode('g', {class: `pn-vertex pn-${definition.key}`});
        group.append(svgNode('circle', {cx: x, cy: y, r: compact ? 4.5 + Math.abs(item.r) * 3 : 7 + Math.abs(item.r) * 4}));
        if (!compact) {
          group.append(svgNode('circle', {cx: x, cy: y, r: 27, class: 'pn-hit-area'}),
            svgNode('text', {x, y: y + 27, 'text-anchor': 'middle', class: 'pn-coordinate'}, `L${item.layer} N${item.neuron}`),
            svgNode('text', {x, y: y + 43, 'text-anchor': 'middle', class: 'pn-value'}, `r ${signed(item.r)}`),
            svgNode('title', {}, `${coordinate(item)} · r ${signed(item.r)} · ${item.n} prompts`));
          if (mapped(item)) {
            group.setAttribute('role', 'button'); group.setAttribute('tabindex', '0');
            group.setAttribute('aria-label', `Select ${coordinate(item)}, r ${signed(item.r)}`);
            group.addEventListener('click', () => navigate(item));
            group.addEventListener('keydown', event => {if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); navigate(item);}});
          }
        }
        vertices.append(group);
      });
    }
    const hub = svgNode('g', {class: 'pn-hub'});
    hub.append(svgNode('circle', {cx, cy, r: compact ? 16 : 40, class: 'pn-hub-halo'}), svgNode('circle', {cx, cy, r: compact ? 8 : 25}));
    if (compact) hub.append(svgNode('text', {x: cx, y: cy + 30, 'text-anchor': 'middle'}, `L${app.target.layer} N${app.target.neuron}`));
    else hub.append(svgNode('text', {x: cx, y: cy - 2, 'text-anchor': 'middle'}, `L${app.target.layer}`),
      svgNode('text', {x: cx, y: cy + 14, 'text-anchor': 'middle'}, `N${app.target.neuron}`),
      svgNode('text', {x: cx, y: cy + 62, 'text-anchor': 'middle', class: 'pn-selected-label'}, 'SELECTED NEURON'));
    vertices.append(hub); return svg;
  }
  function renderProvenance(source) {
    const facts = node('dl');
    for (const [label, value] of [['Observation', 'Continuous MLP input activation at the final prompt token'],
      ['Statistic', source.statistic || 'Pearson correlation across prompts'], ['Population', 'Original, unsteered discovery prompts only'],
      ['Scope', `${count(source.candidates)} measured candidates; not all model neurons`],
      ['Display thresholds', 'Weak: |r| ≤ 0.20. Strong: |r| ≥ 0.60. Descriptive conventions, not significance tests.'],
      ['Model identity', source.model_identity], ['Map SHA-256', source.map_sha256], ['Activation source SHA-256', source.artifact_sha256]]) {
      if (value) facts.append(node('dt', label), node('dd', String(value)));
    }
    refs.provenanceBody.append(facts);
    if (Array.isArray(source.limitations) && source.limitations.length) {
      const list = node('ul'); for (const value of new Set(source.limitations)) if (typeof value === 'string') list.append(node('li', value));
      refs.provenanceBody.append(list);
    }
  }
  window.PersonaNeighbors = {select, open, close};
  // Supports a late-loaded script after the map already has a selection.
  if (typeof selectedUnit !== 'undefined' && selectedUnit) select(selectedUnit);
})();
