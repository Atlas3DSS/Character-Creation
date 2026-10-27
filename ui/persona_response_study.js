(() => {
  'use strict';
  let timer, busy = false;
  const node = (tag, text) => {const n = document.createElement(tag); if (text != null) n.textContent = text; return n;};
  const fmt = value => typeof value === 'number' ? Number(value.toPrecision(3)).toString() : '—';
  function table(headers, rows) {
    const wrap = node('div'); wrap.style.overflowX = 'auto';
    const t = node('table'); t.style.width = '100%'; t.style.textAlign = 'left';
    const head = node('tr'); headers.forEach(h => head.append(node('th', h))); t.append(head);
    rows.forEach(row => {const tr = node('tr'); row.forEach(v => tr.append(node('td', v))); t.append(tr);});
    wrap.append(t); return wrap;
  }
  function render(data) {
    const host = document.getElementById('response-study'); if (!host) return;
    const priorContext = host.querySelector('[data-context]')?.value;
    const priorBatch = host.querySelector('[data-batch]')?.value;
    host.replaceChildren(node('h3', 'Response-contrast pilot · 256 answers'));
    const pilotState = data.pilot.state === 'frozen' ? 'running · plan fixed' : data.pilot.state;
    host.append(node('p', `Pilot: ${data.pilot.complete}/${data.pilot.expected} complete · ${pilotState}. Development: ${data.development.complete}/${data.development.expected} complete.`));
    host.append(node('p', 'Eight authored development families fit the directions. Evaluation uses 12 separate discovery and 9 heldout families, two seeds and six arms: 252 answers plus four replays. Replays are not independent evidence.'));
    host.append(node('p', 'A: dry, skeptical wit · B: warm, attentive candor · each also reversed, alongside baseline and matched random direction. No automatic preset promotion.'));
    const select = node('select'); select.dataset.batch = ''; select.setAttribute('aria-label', 'Study scene and seed batch');
    select.append(new Option('Choose a scene / seed batch', ''));
    for (const [name, study] of [['Pilot', data.pilot], ['Development', data.development]]) {
      for (const b of study.batches) if (b.id) select.append(new Option(`${name} · ${b.replay ? 'replays' : b.scene || 'scene'} · ${b.split || ''} · seed ${b.seed} · ${b.complete}/${b.total} · ${b.state}`, b.id));
    }
    select.value = priorBatch || '';
    const link = node('a', 'Open batch responses'); link.style.marginLeft = '12px';
    function updateLink() {if (select.value) link.href = `/?batch=${encodeURIComponent(select.value)}#lab`; else link.removeAttribute('href');}
    select.onchange = updateLink; updateLink(); host.append(select, link);
    host.append(node('h4', 'Blinded response judgments'));
    if (!data.analysis) host.append(node('p', 'No completed judgment analysis available. Activity and propagation do not establish style or coherence.'));
    else {
      host.append(node('p', 'Paired mean differences; seeds averaged within scene. Issue and uncertain rates stay separate. Intervals are descriptive, unadjusted 95% scene-bootstrap intervals.'));
      for (const split of ['discovery', 'heldout']) {
        const block = data.analysis[split]; if (!block) continue;
        host.append(node('h4', `${split} · ${block.n_scenes} scenes`));
        const rows = Object.entries(block.paired_contrasts).map(([name, m]) => [name,
          ...['dry_wit', 'attentive_candor', 'fidelity_issue', 'fidelity_uncertain', 'coherence_issue', 'coherence_uncertain'].map(k => {
            const v = m[k]; return v ? `${fmt(v.mean)} [${(v.ci95 || []).map(fmt).join(', ')}]` : '—';
          })]);
        host.append(table(['Arm minus control', 'Dry wit', 'Attentive candor', 'Fidelity issue', 'Fidelity uncertain', 'Coherence issue', 'Coherence uncertain'], rows));
      }
    }
    if (data.repair) {
      host.append(node('h4', 'Lower-dose repair probe · 42 answers'));
      host.append(node('p', 'Seven selected discovery failures, not fresh heldout evidence. Lower doses reduced some issues; baseline and random controls were cleaner. No promotion.'));
      const names = {baseline:'Baseline',reference:'Direction +0.10',half:'Direction +0.05',quarter:'Direction +0.025',random_half:'Random +0.05',random_quarter:'Random +0.025'};
      host.append(table(['Arm','Fidelity issues','Fidelity uncertain','Coherence issues','Coherence uncertain'], Object.entries(data.repair.arms).map(([arm,r]) => [names[arm],`${r.fidelity.issue}/${r.n}`,`${r.fidelity.uncertain}/${r.n}`,`${r.coherence.issue}/${r.n}`,`${r.coherence.uncertain}/${r.n}`])));
      const links=node('p');
      data.repair.study.batches.forEach((b,i) => {if (!b.id) return; const a=node('a',`${b.scene} · seed ${b.seed}`);a.href=`/?batch=${encodeURIComponent(b.id)}#lab`;links.append(a,node('span',i===data.repair.study.batches.length-1?'':' · '));});host.append(links);
    }
    host.append(node('h4', 'Fixed-input propagation'));
    host.append(node('p', 'Same prompt and fixed input tokens; symmetric downstream response to a source pulse. This measures sensitivity, not direct wiring or prose coherence. Values below are residual L2 change per unit dose, not raw activation norms.'));
    const choices = data.propagation.measurements;
    if (!choices.length) host.append(node('p', `No propagation measurements available · ${data.propagation.state}.`));
    else {
      const picker = node('select'); picker.dataset.context = ''; picker.setAttribute('aria-label', 'Propagation context and direction');
      choices.forEach((c, i) => picker.append(new Option(`${c.context || 'Context'} · ${c.direction || 'Direction'}`, String(i))));
      if (priorContext && +priorContext < choices.length) picker.value = priorContext;
      const grid = node('div');
      function draw() {
        const c = choices[+picker.value];
        grid.replaceChildren(table(['Layer / token lag', '0', '1', '2', '4', '8'], [31,32,48,63].map(layer => [String(layer), ...[0,1,2,4,8].map(lag => fmt(c.metrics.find(m => m.layer === layer && m.lag === lag)?.small_symmetric_derivative_l2))])));
      }
      picker.onchange = draw; host.append(picker, grid); draw();
      host.append(node('p', 'Small symmetric derivative: (response at +0.05 − response at −0.05) / 0.10; four layers × five token lags.'));
    }
    host.append(node('h4', 'Self-rating utility diagnostic'));
    if (!data.self_rating) host.append(node('p', 'No completed self-rating analysis available.'));
    else {
      const r = data.self_rating;
      host.append(node('p', `${r.n_valid}/${r.n_selected} valid ratings · four discovery families only; diagnostic, not a reward or training signal.`));
      host.append(table(['Score', 'Spearman', 'Within-scene centered', 'Ceiling fraction'], Object.entries(r.metrics).map(([name, m]) => [name, fmt(m.spearman), fmt(m.within_scene_centered_spearman), fmt(m.ceiling_fraction)])));
    }
    host.append(node('hr'));
  }
  async function open() {
    clearTimeout(timer); if (busy) return; busy = true;
    try {const response = await fetch('/api/response-study'); if (!response.ok) throw new Error(`Study summary unavailable (${response.status})`); render(await response.json());}
    catch (error) {const host = document.getElementById('response-study'); if (host && !host.children.length) host.append(node('p', error.message));}
    finally {busy = false; timer = setTimeout(() => {if (!document.hidden && !document.getElementById('studies-view')?.classList.contains('hidden')) open();}, 30000);}
  }
  window.PersonaResponseStudy = {open};
})();
