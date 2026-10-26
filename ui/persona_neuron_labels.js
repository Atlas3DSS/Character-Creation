(() => {
  'use strict';
  let data = {labels: [], candidates: []}, selected;
  const node = (tag, text, cls) => {const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n;};
  const matching = (rows, n) => (rows || []).filter(r => r.layer === n.layer && r.neuron === n.neuron);
  function badge(row) {const n = node('span', `${row.direction === 'inverse' ? 'Inverse · ' : ''}${row.label} · ${(Math.floor(row.confidence.lower_bound * 1000)/10).toFixed(1)}%+`, 'nl-badge'); n.title = `${row.confidence.samples} held-out positive predictions · observed precision ${(row.confidence.value * 100).toFixed(1)}% · ${row.confidence.method} · Wilson 95% lower bound ${(row.confidence.lower_bound * 100).toFixed(1)}%`; return n;}
  function select(n) {
    selected = n; const host = document.getElementById('neuron-label-selected'); if (!host) return;
    host.replaceChildren(node('h3', 'Calibrated labels'));
    const rows = matching(data.labels, n).slice(0, 3); rows.forEach(r => host.append(badge(r)));
    if (!rows.length) host.append(node('p', 'No labels pass the held-out precision lower-bound >70% evidence threshold for this neuron.', 'hint'));
    const candidates = matching(data.candidates, n);
    if (candidates.length) {const d = node('details'); d.append(node('summary', `${candidates.length} unvalidated label candidates`)); candidates.slice(0, 12).forEach(r => d.append(node('p', `${r.direction === 'inverse' ? 'Inverse · ' : ''}${r.label}: ${r.reason || 'Validation pending'}`, 'hint'))); d.append(node('pre',JSON.stringify(candidates.slice(0,12),null,2))); host.append(d);}
    if (rows.length) {const d = node('details'); d.append(node('summary', 'Label evidence'), node('pre', JSON.stringify(rows, null, 2))); host.append(d);}
  }
  async function refresh() {
    const host = document.getElementById('neuron-label-overview'); if (!host) return;
    host.replaceChildren(node('h3', 'Neuron labels'));
    try {
      const response = await fetch('/api/neuron-labels'); const value = await response.json(); if (!response.ok) throw Error(value.error || 'Labels unavailable');
      data = value;
      host.append(node('p', 'At most three strong associations per neuron. Held-out probe precision must have a Wilson 95% lower bound above 70%; inverse denotes negative association.', 'hint'));
      if (!data.labels?.length) host.append(node('p', data.reason || 'Validation pending: no labels pass the held-out precision gate yet.', 'hint'));
      for (const row of (data.labels || []).slice(0, 12)) {const b = node('button', `L${row.layer} N${row.neuron} · `); b.type = 'button'; b.append(badge(row)); b.onclick = () => window.selectNeuron?.({...row, atlasOnly: true}); host.append(b);}
      if (data.candidates?.length) {
        const more=node('details');more.append(node('summary',`${data.candidates.length} experimental label candidates · not confidence scores`));
        const search=node('input');search.type='search';search.placeholder='Find a candidate concept';search.setAttribute('aria-label','Candidate concept');const list=node('div');
        const render=()=>{list.replaceChildren();for(const row of data.candidates.filter(r=>r.label.includes(search.value.toLowerCase())).slice(0,12)){const b=node('button',`${row.label} · L${row.layer} N${row.neuron} · candidate`);b.type='button';b.onclick=()=>window.selectNeuron?.({...row,atlasOnly:true});list.append(b);}};
        search.oninput=render;more.append(node('p','Discovery candidates only. Four framings of one authored scene are not four independent replications. Association direction is not a steering-dose sign.','hint'),search,list);render();host.append(more);
      }
      if (selected) select(selected);
    } catch (error) {host.append(node('p', error.message, 'hint'));}
  }
  window.PersonaNeuronLabels = {select, refresh}; refresh();
})();
