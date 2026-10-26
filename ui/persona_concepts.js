/* Search measured evidence; no inference, intervention, or automatic jobs. */
(() => {
  'use strict';
  function node(tag, text, cls) {const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (cls) el.className = cls; return el;}
  function button(text, action) {const el = node('button', text); el.type = 'button'; el.onclick = action; return el;}
  function mount(host) {
    if (!host || host.dataset.conceptsMounted) return;
    host.dataset.conceptsMounted = 'true';
    const form = node('form'), input = node('input'), corpus = node('select'), submit = node('button', 'Find measured associations');
    input.type = 'search'; input.required = true; input.maxLength = 160; input.placeholder = 'Concept words, e.g. curious or justice'; input.setAttribute('aria-label', 'Concept words');
    corpus.setAttribute('aria-label', 'Concept evidence corpus');
    for (const [value, label] of [['all', 'All corpora'], ['big_five', 'Big Five'], ['scotus', 'SCOTUS'], ['skippy', 'Skippy books'], ['adult_sexuality', 'Adult sexuality']]) {const opt = node('option', label); opt.value = value; corpus.append(opt);}
    submit.type = 'submit'; form.append(input, corpus, submit);
    const status = node('p'), body = node('div'); status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
    let serial = 0, controller;
    const lensMode = node('select'); lensMode.setAttribute('aria-label', 'Search method');
    for (const [value, label] of [['lens', 'Checkpoint token lens · all neurons'], ['measured', 'Saved activation associations']]) {const opt = node('option', label); opt.value = value; lensMode.append(opt);}
    form.prepend(lensMode);
    const description = node('p');
    const setMode = () => {const lens = lensMode.value === 'lens'; corpus.hidden = lens; input.maxLength = lens ? 100 : 160; input.placeholder = lens ? 'Word start: lust, evil, sarcas, horny…' : 'Concept words, e.g. curious or justice'; submit.textContent = lens ? 'Find checkpoint candidates' : 'Find measured associations'; description.textContent = lens ? 'Search stored positive and negative vocabulary projections across every MLP layer. Word-start token matches suggest candidates; they do not establish a concept or a causal effect. Try related words separately.' : 'Search saved prompt activations. All query words must match; results require measured examples and controls.';};
    lensMode.onchange = () => {serial++; controller?.abort(); submit.disabled = false; body.replaceChildren(); status.textContent = ''; setMode();};
    host.append(node('h2', 'Explore concepts'), button('Search activations on your own text ↗', () => tab('activation-search')), description, form, status, body); setMode();
    const overnight = node('details'), overnightBody = node('div');
    overnight.append(node('summary', 'Overnight emotion map · measured contrasts'), overnightBody);
    host.append(overnight);
    const loadOvernight = async () => {
      overnightBody.replaceChildren(node('p', 'Loading measured emotion contrasts…'));
      try {
        const response = await fetch('/api/overnight/features'), data = await response.json();
        if (!response.ok) throw Error(data.error || 'Map unavailable');
        overnightBody.replaceChildren(node('p', 'State: ' + (data.state || 'pending')));
        overnightBody.append(node('p', 'Discovery associations across feeling, expression, regulation and recognition, compared with neutral and word-only controls. Shared neuron candidates are not established causal circuits.'));
        if (!data.features?.length) {overnightBody.append(node('p', 'The first emotion captures must finish before candidate rankings are available.')); return;}
        const choice = node('select'), detail = node('div'); choice.setAttribute('aria-label', 'Measured emotion');
        for (const feature of data.features) {const option = node('option', feature.feature + ' · ' + feature.state); option.value = feature.feature; choice.append(option);}
        const render = () => {
          const feature = data.features.find(item => item.feature === choice.value); detail.replaceChildren();
          if (!feature) return;
          detail.append(node('p', feature.claim || 'Exploratory association, awaiting intervention tests.'));
          const table = node('table'), head = node('tr');
          for (const label of ['Neuron', 'Contrast score', 'Frame agreement', 'Inspect']) head.append(node('th',label)); table.append(head);
          for (const candidate of feature.candidates || []) {
            const row = node('tr'), action = node('td');
            action.append(button('Lens & neighborhood', () => {window.PersonaAtlas.select(candidate);window.PersonaAtlas.open();}));
            row.append(node('td',`L${candidate.layer} N${candidate.neuron}`),node('td',Number(candidate.score).toPrecision(4)),node('td',String(candidate.frame_agreement)),action);table.append(row);
          }
          detail.append(table);
          const links = (data.edges || []).filter(edge => edge.source === feature.feature || edge.target === feature.feature);
          if (links.length) detail.append(node('p','Shared candidate coordinates: ' + links.map(edge => `${edge.source === feature.feature ? edge.target : edge.source} (${edge.shared_coordinates})`).join(', ')));
          const evidence = node('details');evidence.append(node('summary','Evidence, counterexamples and binding'),node('pre',JSON.stringify({binding:data.binding,method:data.method,...feature},null,2)));detail.append(evidence);
        };
        choice.onchange=render;overnightBody.append(choice,detail);render();
      } catch(error) {overnightBody.replaceChildren(node('p',error.message));}
      finally {overnightBody.append(button('Refresh emotion map',loadOvernight));}
    };
    overnight.addEventListener('toggle',()=>{if(overnight.open)loadOvernight();});
    function navigate(item, neighborhood = false) {
      const map = typeof state !== 'undefined' ? state.map : window.state?.map;
      const target = map?.neurons?.find(n => n.layer === item.layer && n.neuron === item.neuron);
      const select = typeof selectNeuron === 'function' ? selectNeuron : window.selectNeuron;
      if (target && select) {
        if (!neighborhood && typeof tab === 'function') tab('map');
        select(target);
      }
      if (neighborhood && window.PersonaNeighbors) {window.PersonaNeighbors.select(target || item); window.PersonaNeighbors.open();}
      else if (!target || !select) status.textContent = 'This candidate is not in the current visible map.';
    }
    form.onsubmit = async event => {
      event.preventDefault(); const requestId = ++serial; controller?.abort(); controller = new AbortController();
      submit.disabled = true; status.textContent = 'Searching saved measurements…'; body.replaceChildren();
      try {
        if (lensMode.value === 'lens') {
          status.textContent = 'Searching checkpoint token projections…';
          const response = await fetch('/api/lens/search?q=' + encodeURIComponent(input.value.trim()) + '&word_start=true', {signal: controller.signal});
          const data = await response.json(); if (requestId !== serial) return;
          if (!response.ok) throw Error(data.error || 'Checkpoint search failed');
          status.textContent = `${data.matches.length} token matches · ${data.missing_layers.length} missing layers. ${data.scope}`;
          body.append(node('p', 'These are static output-direction projections, not measured activation or steering scores. Absence from the stored top tokens does not mean the concept is absent.'));
          const table = node('table'), head = node('tr'); for (const label of ['Neuron', 'Pole', 'Token', 'Projection', 'Inspect']) head.append(node('th', label)); table.append(head);
          for (const item of data.matches) {const row = node('tr'), action = node('td'); action.append(button('Lens & weight neighborhood', () => {window.PersonaAtlas.select(item); window.PersonaAtlas.open();})); row.append(node('td', `L${item.layer} · N${item.neuron}`), node('td', item.direction), node('td', JSON.stringify(item.token)), node('td', item.logit.toFixed(3)), action); table.append(row);}
          body.append(table); if (!data.matches.length) body.append(node('p', 'Try a related word or shorter token fragment, or switch to saved activation associations.'));
          return;
        }
        const response = await fetch('/api/concepts?' + new URLSearchParams({q: input.value.trim(), corpus: corpus.value, limit: '20'}), {signal: controller.signal});
        const data = await response.json(); if (requestId !== serial) return;
        if (!response.ok) throw Error(data.error || 'Concept search failed');
        status.textContent = data.message;
        const source = data.source;
        body.append(node('p', `${source.matching_corpus_rows ?? 0} matching discovery rows · ${source.matching_measured_rows ?? 0} measured matches · ${source.unmeasured_matches ?? 0} matches without saved activations`));
        for (const stratum of data.strata || []) body.append(node('p', `${stratum.corpus}: ${stratum.matched} matches / ${stratum.controls} controls — ${stratum.used ? 'included' : 'insufficient support'}`));
        if (data.candidates.length) {
          body.append(node('h3', 'Candidate neurons'), node('p', 'Score: standardized mean activation contrast, weighted within corpora. Ranked by absolute score; a negative score means lower activation. These are exploratory associations.'));
          const table = node('table'), head = node('tr'); for (const label of ['Neuron', 'Contrast', 'Explore']) head.append(node('th', label)); table.append(head);
          for (const item of data.candidates) {
            const row = node('tr'), name = node('td'), actions = node('td'); name.append(button(`L${item.layer} · N${item.neuron}`, () => navigate(item)));
            actions.append(button('Open neighborhood', () => navigate(item, true))); row.append(name, node('td', item.score.toFixed(3)), actions); table.append(row);
          }
          body.append(table);
          if (data.graph && window.PersonaGraph) {
            body.append(node('h3', 'Selected collection: measured co-activity'), node('p', 'Explore relationships among the ranked candidates. This does not establish a semantic feature or causal circuit; pooled corpus/task differences may affect edges.'));
            body.append(window.PersonaGraph.render(data.graph, {compact: false, threshold: .2, layout: 'similarity', bundle: 0, onSelect: item => navigate(item, true)}));
          }
        }
        const examples = node('details'); examples.append(node('summary', 'Matching prompt evidence (up to 12)'));
        for (const item of data.examples) {examples.append(node('h4', `${item.corpus} · ${item.category} · ${item.measured ? 'measured' : 'unmeasured'}`), node('p', item.prompt), node('code', item.id));} body.append(examples);
        const provenance = node('details'); provenance.append(node('summary', 'Method, provenance and limits'), node('p', source.matching), node('p', source.statistic));
        for (const text of source.limitations || []) provenance.append(node('p', text));
        for (const key of ['model_identity', 'artifact_sha256', 'map_sha256', 'corpus_sha256', 'captured_corpus_sha256']) if (source[key]) provenance.append(node('p', `${key}: ${source[key]}`));
        body.append(provenance);
      } catch (error) {if (error.name !== 'AbortError' && requestId === serial) status.textContent = error.message;}
      finally {if (requestId === serial) submit.disabled = false;}
    };
  }
  window.PersonaConcepts = {mount};
})();
