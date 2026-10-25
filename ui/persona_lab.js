/* Reusable recipes and explicit one-to-six-job experiments. Private text stays in the DOM/API. */
(() => {
  'use strict';
  const MAX = 6, BASE = 'builtin:baseline';
  const phases = [['all', 'Prompt + response'], ['prefill', 'Prompt only'], ['decode', 'Response only']];
  const traitNames = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'];
  const app = {ready: false, model: null, recipes: new Map(), prompts: new Map(), selectedRecipes: [BASE],
    selectedPrompts: [], slots: [{prompt: '', recipe: BASE}], editor: {name: 'New recipe', pushes: [], phase: 'all'},
    offset: 0, total: 0, serial: 0, searchSerial: 0, batchSerial: 0, activeBatch: null, submitting: false};
  const refs = {};
  let searchTimer, batchTimer, capacityTimer;
  const copy = value => JSON.parse(JSON.stringify(value));
  function node(tag, text, cls) {
    const value = document.createElement(tag);
    if (text !== undefined) value.textContent = text;
    if (cls) value.className = cls;
    return value;
  }
  function button(text, action, cls = '') {
    const value = node('button', text, cls); value.type = 'button'; value.onclick = action; return value;
  }
  function input(type, value, label) {
    const result = node('input'); result.type = type; result.value = value;
    if (label) result.setAttribute('aria-label', label); return result;
  }
  function select(options, label) {
    const value = node('select'); value.setAttribute('aria-label', label);
    options.forEach(([key, text]) => {const item = node('option', text); item.value = key; value.append(item);});
    return value;
  }
  function field(label, control) {
    const value = node('label', undefined, 'lab-field'); value.append(node('span', label), control); return value;
  }
  function row(...children) {const value = node('div', undefined, 'lab-row'); value.append(...children); return value;}
  function hint(text) {return node('p', text, 'lab-hint');}
  function panel(title, subtitle) {
    const value = node('section', undefined, 'lab-panel'); value.append(node('h3', title));
    if (subtitle) value.append(hint(subtitle)); return value;
  }
  function notify(message = '', error = false) {
    refs.notice.textContent = message; refs.notice.classList.toggle('lab-error', error);
    refs.notice.hidden = !message;
  }
  async function request(url, payload) {
    const response = await fetch(url, payload === undefined ? {} : {method: 'POST',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    const value = await response.json().catch(() => ({error: response.statusText}));
    if (!response.ok) throw Error(value.error || `Request failed (${response.status})`);
    return value;
  }
  function titleFor(prompt) {return prompt.label || prompt.category || (prompt.custom ? 'Custom prompt' : prompt.id);}
  function readable(value) {return String(value || '').replaceAll('_', ' ');}
  function phaseName(value) {return phases.find(item => item[0] === value)?.[1] || value;}
  function pushName(push) {
    if (push.kind === 'trait') return `L${push.layer} · ${readable(push.trait)}`;
    if (push.kind === 'neuron') return `L${push.layer} · N${push.neuron} input`;
    if (push.kind === 'column') return `L${push.layer} · N${push.neuron} output`;
    return `L${push.layer} · ${readable(push.kind)}${push.site ? ' · ' + readable(push.site) : ''}`;
  }
  function recipeDescription(recipe) {
    if (!recipe.pushes.length) return 'Baseline · no activation intervention';
    return `${recipe.pushes.length} push${recipe.pushes.length === 1 ? '' : 'es'} · ${phaseName(recipe.phase)}`;
  }
  function modelMatches(recipe) {return Boolean(recipe) && (!recipe.model_identity || recipe.model_identity === app.model?.identity);}
  function addLocalRecipe(recipe, choose = true) {
    const key = `local:${++app.serial}`;
    app.recipes.set(key, {...copy(recipe), model_identity: app.model?.identity, local: true});
    if (choose && app.selectedRecipes.length < MAX) app.selectedRecipes.push(key);
    return key;
  }
  function toggleSelection(kind, key) {
    const list = kind === 'prompt' ? app.selectedPrompts : app.selectedRecipes;
    const index = list.indexOf(key);
    if (index >= 0) {
      list.splice(index, 1);
      if (kind === 'prompt') app.slots.forEach(slot => {if (slot.prompt === key) slot.prompt = '';});
    } else {
      if (list.length >= MAX) {notify(`Select up to ${MAX} ${kind}s. Remove one to make room.`, true); return;}
      list.push(key);
      if (app.slots.length === 1 && !app.slots[0].prompt && kind === 'prompt') app.slots[0].prompt = key;
    }
    renderSelections(); renderRecipeLibrary(); renderPromptSelection(); renderSlots(); notify();
  }
  function renderSelections() {
    refs.selectedPrompts.replaceChildren();
    refs.promptCount.textContent = `${app.selectedPrompts.length} / 6 selected`;
    for (const id of app.selectedPrompts) {
      const prompt = app.prompts.get(id);
      refs.selectedPrompts.append(button(`${titleFor(prompt)} ×`, () => toggleSelection('prompt', id), 'lab-token'));
    }
    if (!app.selectedPrompts.length) refs.selectedPrompts.append(hint('Select prompts below or add your own.'));
  }
  function renderPromptSelection() {
    refs.promptRows.querySelectorAll('[data-prompt-key]').forEach(control => {
      const chosen = app.selectedPrompts.includes(control.dataset.promptKey);
      control.textContent = chosen ? 'Selected ✓' : 'Select'; control.setAttribute('aria-pressed', String(chosen));
      control.classList.toggle('lab-chosen', chosen);
    });
  }
  async function loadPrompts() {
    const serial = ++app.searchSerial;
    refs.promptStatus.textContent = 'Loading prompts…';
    refs.previous.disabled = refs.next.disabled = true;
    try {
      const params = new URLSearchParams({corpus: refs.corpus.value, split: refs.split.value,
        q: refs.search.value.trim(), offset: String(app.offset), limit: '12'});
      const data = await request('/api/prompts?' + params);
      if (serial !== app.searchSerial) return;
      app.total = data.total; app.offset = data.offset;
      refs.promptRows.replaceChildren();
      for (const prompt of data.rows) {
        app.prompts.set(prompt.id, prompt);
        const card = node('article', undefined, 'lab-prompt');
        const choose = button('Select', () => toggleSelection('prompt', prompt.id));
        choose.dataset.promptKey = prompt.id;
        const heading = row(node('strong', titleFor(prompt)), choose); heading.classList.add('lab-spread');
        const meta = node('div', undefined, 'lab-meta');
        meta.append(node('span', readable(prompt.corpus), 'lab-tag'), node('span', readable(prompt.split), 'lab-tag'),
          node('span', readable(prompt.category), 'lab-dim'));
        const preview = node('p', prompt.preview || prompt.prompt?.slice(0, 210) || '', 'lab-preview');
        const details = node('details', undefined, 'lab-source'); details.append(node('summary', 'Read prompt & source'));
        if (prompt.system) details.append(node('p', 'Profile instruction', 'lab-caption'), node('pre', prompt.system));
        details.append(node('pre', prompt.prompt || prompt.preview || ''), node('small', `Source ID: ${prompt.id}`));
        card.append(heading, meta, preview, details); refs.promptRows.append(card);
      }
      if (!data.rows.length) refs.promptRows.append(hint('No prompts match these filters. Try another corpus or search.'));
      const start = data.total ? data.offset + 1 : 0;
      refs.promptStatus.textContent = `${start}–${data.offset + data.rows.length} of ${Number(data.total).toLocaleString()}`;
      refs.previous.disabled = app.offset === 0; refs.next.disabled = app.offset + data.rows.length >= data.total;
      renderPromptSelection();
    } catch (error) {if (serial === app.searchSerial) {refs.promptStatus.textContent = 'Could not load prompts.'; notify(error.message, true);}}
  }
  function editorValue() {
    const value = {...copy(app.editor), name: refs.recipeName.value.trim(), phase: refs.recipePhase.value};
    if (!value.name) throw Error('Give this recipe a name.');
    for (const push of value.pushes) {
      if (!Number.isInteger(push.layer) || push.layer < 0 || push.layer >= (app.model?.model?.layers || 64))
        throw Error('Each push needs a layer inside this model.');
      if (!Number.isFinite(push.dose) || Math.abs(push.dose) > 4) throw Error('Relative doses must be between −4 and +4.');
      if (['neuron', 'column'].includes(push.kind) && (!Number.isInteger(push.neuron) || push.neuron < 0 || push.neuron >= (app.model?.model?.intermediate || 17408)))
        throw Error('Each neuron number must be inside this model.');
    }
    return value;
  }
  function loadEditor(recipe, savedId = null) {
    app.editor = {name: recipe.name, pushes: copy(recipe.pushes || []), phase: recipe.phase || 'all', savedId};
    renderEditor();
  }
  function renderEditor() {
    refs.recipeName.value = app.editor.name; refs.recipePhase.value = app.editor.phase;
    refs.updateRecipe.disabled = !app.editor.savedId;
    refs.editing.textContent = app.editor.savedId ? 'Editing a saved recipe. Update it or save a separate copy.' : 'Draft recipe · save it to reuse later.';
    refs.pushRows.replaceChildren();
    if (!app.editor.pushes.length) refs.pushRows.append(hint('No pushes: this recipe is an unsteered baseline.'));
    app.editor.pushes.forEach((push, index) => {
      const item = node('div', undefined, 'lab-push-item');
      const line = node('div', undefined, 'lab-push');
      const label = node('span', pushName(push), 'lab-push-name');
      const dose = input('number', push.dose, `Relative dose for ${pushName(push)}`); dose.min = -4; dose.max = 4; dose.step = .05;
      dose.oninput = () => {push.dose = dose.value === '' ? NaN : Number(dose.value);};
      const remove = button('×', () => {app.editor.pushes.splice(index, 1); renderEditor();});
      remove.setAttribute('aria-label', `Remove ${pushName(push)}`);
      line.append(label, field('Dose', dose), remove); item.append(line);
      const target = node('details', undefined, 'lab-push-target'); target.append(node('summary', 'Edit target'));
      const layer = input('number', push.layer, `Layer for push ${index + 1}`); layer.min = 0; layer.max = (app.model?.model?.layers || 64) - 1;
      layer.oninput = () => {push.layer = layer.value === '' ? NaN : Number(layer.value); label.textContent = pushName(push);};
      const controls = row(field('Layer', layer));
      if (['neuron', 'column'].includes(push.kind)) {
        const site = select([['neuron', 'Neuron input'], ['column', 'Output column']], `Site for push ${index + 1}`); site.value = push.kind;
        site.onchange = () => {push.kind = site.value; label.textContent = pushName(push);};
        const coordinate = input('number', push.neuron, `Neuron for push ${index + 1}`); coordinate.min = 0; coordinate.max = (app.model?.model?.intermediate || 17408) - 1;
        coordinate.oninput = () => {push.neuron = coordinate.value === '' ? NaN : Number(coordinate.value); label.textContent = pushName(push);};
        controls.append(field('Site', site), field('Neuron', coordinate));
      } else if (push.kind === 'trait') {
        const trait = select(traitNames.map(name => [name, readable(name)]), `Trait for push ${index + 1}`); trait.value = push.trait;
        trait.onchange = () => {push.trait = trait.value; label.textContent = pushName(push);}; controls.append(field('Trait', trait));
      }
      target.append(controls); item.append(target); refs.pushRows.append(item);
    });
  }
  function renderRecipeLibrary() {
    refs.recipeLibrary.replaceChildren();
    refs.recipeCount.textContent = `${app.selectedRecipes.length} / 6 selected`;
    for (const [key, recipe] of app.recipes) {
      const chosen = app.selectedRecipes.includes(key), matches = modelMatches(recipe);
      const line = node('div', undefined, 'lab-recipe-item');
      const choose = button(chosen ? '✓' : '+', () => toggleSelection('recipe', key), chosen ? 'lab-chosen' : '');
      choose.setAttribute('aria-label', `${chosen ? 'Deselect' : 'Select'} ${recipe.name}`);
      choose.setAttribute('aria-pressed', String(chosen)); choose.disabled = !matches;
      const body = node('div'); body.append(node('strong', recipe.name), node('small', recipeDescription(recipe), 'lab-block'));
      if (recipe.local) body.append(node('small', 'Unsaved variant', 'lab-dim'));
      if (!matches) body.append(node('small', 'Saved for a different model', 'lab-dim'));
      const edit = button('Edit', () => loadEditor(recipe, recipe.id || null)); edit.disabled = !matches;
      line.append(choose, body, edit); refs.recipeLibrary.append(line);
    }
  }
  async function saveRecipe(update) {
    try {
      const recipe = editorValue();
      if (!app.model?.identity) throw Error('Model identity is still loading.');
      refs.saveRecipe.disabled = refs.updateRecipe.disabled = true;
      const payload = {name: recipe.name, model_identity: app.model.identity, pushes: recipe.pushes, phase: recipe.phase};
      if (update) payload.id = recipe.savedId;
      const raw = await request('/api/saved-recipes', payload), saved = raw.recipe || raw;
      app.recipes.set('saved:' + saved.id, saved); loadEditor(saved, saved.id);
      renderRecipeLibrary(); renderSlots(); notify(`Saved “${saved.name}”.`);
    } catch (error) {notify(error.message, true);}
    finally {refs.saveRecipe.disabled = false; refs.updateRecipe.disabled = !app.editor.savedId;}
  }
  function useEditor() {
    try {
      if (app.selectedRecipes.length >= MAX) throw Error('Deselect a recipe before adding another to this batch.');
      const recipe = editorValue(); delete recipe.savedId;
      const key = addLocalRecipe(recipe);
      if (app.slots.length === 1) app.slots[0].recipe = key;
      renderRecipeLibrary(); renderSlots(); notify(`“${recipe.name}” is selected for this batch.`);
    } catch (error) {notify(error.message, true);}
  }
  function doseVariants() {
    try {
      const recipe = editorValue();
      if (!recipe.pushes.length) throw Error('Add a push before making dose variants.');
      const factors = refs.variantFactors.value.split(',').map(value => value.trim()).filter(Boolean).map(Number);
      if (!factors.length || factors.length > MAX || factors.some(value => !Number.isFinite(value)))
        throw Error('Enter one to six numeric multipliers, separated by commas.');
      const variants = factors.map(factor => ({name: `${recipe.name} × ${factor}`, phase: recipe.phase,
        pushes: recipe.pushes.map(push => ({...push, dose: push.dose * factor}))}));
      if (variants.some(value => value.pushes.some(push => Math.abs(push.dose) > 4))) throw Error('A scaled dose exceeds ±4. Reduce the multiplier.');
      app.selectedRecipes = variants.map(value => addLocalRecipe(value, false));
      renderRecipeLibrary(); renderSlots(); notify(`Selected ${variants.length} dose variants. Arrange them with a prompt below.`);
    } catch (error) {notify(error.message, true);}
  }
  function arrange(recipeCount, promptCount) {
    if (app.selectedRecipes.length < recipeCount || app.selectedPrompts.length < promptCount) {
      notify(`This layout needs ${recipeCount} selected recipe${recipeCount === 1 ? '' : 's'} and ${promptCount} selected prompt${promptCount === 1 ? '' : 's'}.`, true); return;
    }
    app.slots = [];
    for (const recipe of app.selectedRecipes.slice(0, recipeCount))
      for (const prompt of app.selectedPrompts.slice(0, promptCount)) app.slots.push({recipe, prompt});
    renderSlots(); notify();
  }
  function pairedBaseline() {
    const recipe = app.selectedRecipes.find(key => app.recipes.get(key)?.pushes.length);
    if (!recipe || !app.selectedPrompts.length) {notify('Select a recipe with pushes and one prompt for the paired baseline preset.', true); return;}
    app.slots = [{recipe: BASE, prompt: app.selectedPrompts[0]}, {recipe, prompt: app.selectedPrompts[0]}]; renderSlots(); notify();
  }
  function renderSlots() {
    refs.slots.replaceChildren();
    app.slots.forEach((slot, index) => {
      const card = node('article', undefined, 'lab-slot');
      const remove = button('Remove', () => {app.slots.splice(index, 1); renderSlots();}); remove.disabled = app.slots.length === 1;
      card.append(row(node('strong', `Job ${index + 1}`), remove));
      const promptChoices = [...new Set([...app.selectedPrompts, slot.prompt].filter(Boolean))];
      const prompt = select([['', 'Choose a selected prompt'], ...promptChoices.map(key => [key, titleFor(app.prompts.get(key))])], `Prompt for job ${index + 1}`);
      prompt.value = slot.prompt; prompt.onchange = () => {slot.prompt = prompt.value; renderSlots();};
      const recipeChoices = [...new Set([BASE, ...app.selectedRecipes, slot.recipe].filter(Boolean))].filter(key => modelMatches(app.recipes.get(key)));
      const recipe = select(recipeChoices.map(key => [key, app.recipes.get(key).name]), `Recipe for job ${index + 1}`);
      if (!recipeChoices.includes(slot.recipe)) slot.recipe = BASE;
      recipe.value = slot.recipe; recipe.onchange = () => {slot.recipe = recipe.value; renderSlots();};
      card.append(field('Prompt', prompt), field('Recipe', recipe));
      const source = app.prompts.get(slot.prompt), selection = app.recipes.get(slot.recipe);
      if (source) card.append(hint(`${readable(source.corpus || 'custom')} · ${readable(source.split || 'manual')} · ${titleFor(source)}`));
      card.append(node('small', recipeDescription(selection), 'lab-dim')); refs.slots.append(card);
    });
    const ready = app.slots.filter(slot => app.prompts.has(slot.prompt) && app.recipes.has(slot.recipe)).length;
    refs.capacity.textContent = `${app.slots.length} / 6 jobs · ${ready} ready`;
    refs.addSlot.disabled = app.slots.length >= MAX;
    refs.run.disabled = app.submitting || ready !== app.slots.length || !app.model?.identity || app.available === 0;
    refs.run.textContent = app.submitting ? 'Submitting…' : `Run ${app.slots.length} job${app.slots.length === 1 ? '' : 's'}`;
    refs.previewNote.hidden = Number(refs.budget.value) >= 2048;
  }
  async function runBatch() {
    try {
      const seed = Number(refs.seed.value), maxTokens = Number(refs.budget.value), smoke = maxTokens < 2048;
      if (!refs.seed.value.trim() || !Number.isInteger(seed) || seed < 0 || seed > 4294967295) throw Error('Seed must be a whole number between 0 and 4,294,967,295.');
      if (!refs.budget.value.trim() || !Number.isInteger(maxTokens) || maxTokens < 256 || maxTokens > 4096) throw Error('Token budget must be a whole number from 256 to 4,096.');
      if (!app.model?.identity) throw Error('Wait for the model to finish loading.');
      if (app.contextTokens && maxTokens >= app.contextTokens) throw Error(`Choose fewer than ${app.contextTokens} answer tokens so the prompt also fits.`);
      const cells = app.slots.map(slot => {
        const prompt = app.prompts.get(slot.prompt), recipe = app.recipes.get(slot.recipe);
        if (!prompt || !recipe || !modelMatches(recipe)) throw Error('Choose a valid prompt and recipe for every job.');
        const cell = {label: `${smoke ? 'Preview · ' : ''}${recipe.name} · ${titleFor(prompt)}`.slice(0, 120)};
        if (prompt.custom) cell.prompt = prompt.prompt; else cell.prompt_id = prompt.id;
        if (recipe.id) cell.recipe_id = recipe.id;
        else cell.recipe = {name: recipe.name, pushes: copy(recipe.pushes), phase: recipe.phase};
        return cell;
      });
      app.submitting = true; renderSlots(); notify();
      const result = await request('/api/batches', {model_identity: app.model.identity, seed, max_tokens: maxTokens,
        thinking: refs.thinking.checked, smoke, cells});
      await showBatch(result.id); await loadHistory();
      refs.results.scrollIntoView({behavior: 'smooth', block: 'start'});
    } catch (error) {notify(error.message, true);}
    finally {app.submitting = false; renderSlots();}
  }
  async function loadHistory() {
    try {
      const data = await request('/api/batches'); refs.history.replaceChildren(node('option', 'Choose a batch'));
      refs.history.firstChild.value = '';
      for (const batch of data.batches) {
        const date = new Date(Number(batch.created) * 1000);
        const option = node('option', `${date.toLocaleString()} · ${batch.completed}/${batch.total} · ${readable(batch.state)}`);
        option.value = batch.id; refs.history.append(option);
      }
      refs.history.value = app.activeBatch || '';
    } catch (error) {notify(error.message, true);}
  }
  async function loadCapacity() {
    clearTimeout(capacityTimer);
    if (app.capacityLoading) return;
    app.capacityLoading = true;
    try {
      const data = await request('/api/capacity');
      refs.fleetStatus.textContent = `${data.available} / ${data.configured} workers reachable · up to ${data.max_parallel} concurrent jobs`;
      app.available = data.available; app.contextTokens = data.context_tokens || null;
      refs.thinking.disabled = data.thinking_supported === false;
      if (refs.thinking.disabled) refs.thinking.checked = false;
      refs.runtimeNote.textContent = `${data.context_tokens ? data.context_tokens.toLocaleString() + ' total tokens per request, including the prompt. Leave room for both prompt and response.' : 'Prompt and response must fit the worker context.'}${data.thinking_supported === false ? ' Thinking mode is unavailable on these workers.' : ''}`;
      refs.fleetCards.replaceChildren(); renderSlots();
      for (const card of data.cards || []) {
        const item = node('span', `${card.label} · ${card.available ? 'reachable' : 'unavailable'}`, 'lab-tag');
        item.classList.toggle('lab-reachable', Boolean(card.available));
        if (card.error) item.title = card.error;
        refs.fleetCards.append(item);
      }
    } catch (error) {refs.fleetStatus.textContent = 'Worker reachability could not be checked.'; refs.fleetCards.replaceChildren();}
    finally {app.capacityLoading = false; capacityTimer = setTimeout(pollCapacity, 10000);}
  }
  function pollCapacity() {
    if (!document.hidden && !document.getElementById('lab-view')?.classList.contains('hidden')) loadCapacity();
    else capacityTimer = setTimeout(pollCapacity, 10000);
  }
  function provenanceDetails(cell) {
    const details = node('details', undefined, 'lab-source'); details.append(node('summary', 'Prompt, recipe & provenance'));
    const prompt = cell.prompt || {}, recipe = cell.recipe || {};
    details.append(hint(`${readable(prompt.corpus || 'custom')} · ${readable(prompt.split || 'manual')}${prompt.id ? ' · ' + prompt.id : ''}`));
    details.append(node('pre', prompt.text || prompt.prompt || ''));
    details.append(node('strong', recipe.name || 'Recipe'), hint(phaseName(recipe.phase || 'all')));
    if (!recipe.pushes?.length) details.append(hint('Baseline · no activation intervention'));
    else for (const push of recipe.pushes) details.append(node('div', `${pushName(push)} · dose ${push.dose}`));
    const output = cell.output || {}, provenance = output.provenance || {};
    const facts = {Runtime: output.runtime_identity, 'Program hash': provenance.program_sha256,
      'Model identity': provenance.model_identity, 'Prepared identity': provenance.prepared_identity};
    for (const [name, value] of Object.entries(facts)) if (typeof value === 'string') details.append(node('p', `${name}: ${value}`, 'lab-hash'));
    return details;
  }
  function renderBatch(batch) {
    refs.resultCards.replaceChildren();
    const done = batch.cells.filter(cell => ['complete', 'incomplete', 'failed', 'interrupted'].includes(cell.state)).length;
    refs.resultStatus.textContent = `${readable(batch.state)} · ${done} / ${batch.cells.length} finished`;
    if (batch.input?.smoke) refs.resultStatus.append(node('span', 'Preview run · below 2,048 tokens', 'lab-tag lab-preview-tag'));
    for (const cell of batch.cells) {
      const card = node('article', undefined, 'lab-result'); card.dataset.state = cell.state;
      card.append(row(node('h4', cell.label || `Job ${cell.index + 1}`), node('span', readable(cell.state), 'lab-status')));
      card.append(hint(cell.endpoint_label ? `Card: ${cell.endpoint_label}` : 'Waiting for a shared card'));
      if (cell.error) card.append(node('p', typeof cell.error === 'string' ? cell.error : JSON.stringify(cell.error), 'lab-cell-error'));
      if (cell.output) {
        card.append(node('div', cell.output.answer || 'No answer text returned.', 'lab-answer'));
        const usage = cell.output.usage || {};
        card.append(hint(`${cell.output.complete ? 'Complete answer' : 'Incomplete answer'} · ${cell.output.finish_reason || 'finish reason unavailable'}${Number.isFinite(usage.completion_tokens) ? ` · ${usage.completion_tokens} generated tokens` : ''}`));
      } else if (!cell.error) card.append(hint(cell.state === 'running' ? 'Generating a response…' : 'Queued alongside other work.'));
      card.append(provenanceDetails(cell)); refs.resultCards.append(card);
    }
  }
  async function showBatch(id) {
    clearTimeout(batchTimer); app.activeBatch = id;
    const serial = ++app.batchSerial;
    try {
      const batch = await request('/api/batch?' + new URLSearchParams({id}));
      if (serial !== app.batchSerial) return;
      renderBatch(batch); refs.history.value = id;
      if (batch.cells.some(cell => ['queued', 'running', 'pending'].includes(cell.state)))
        batchTimer = setTimeout(() => showBatch(id), document.hidden ? 5000 : 1800);
      else loadHistory();
    } catch (error) {if (serial === app.batchSerial) {refs.resultStatus.textContent = 'Could not refresh this batch.'; notify(error.message, true);}}
  }
  async function loadSources() {
    notify();
    const results = await Promise.allSettled([request('/api/state'), request('/api/corpora'), request('/api/saved-recipes')]);
    if (results[0].status === 'fulfilled') {
      app.model = results[0].value; refs.model.textContent = app.model.model?.model_id || app.model.model?.model || 'Current calibrated model';
    }
    if (results[1].status === 'fulfilled') {
      const chosen = refs.corpus.value;
      refs.corpus.replaceChildren(node('option', 'All corpora')); refs.corpus.firstChild.value = '';
      for (const corpus of results[1].value.corpora) {
        const option = node('option', `${readable(corpus.name)} · ${Number(corpus.rows).toLocaleString()}`); option.value = corpus.name; refs.corpus.append(option);
      }
      if ([...refs.corpus.options].some(option => option.value === chosen)) refs.corpus.value = chosen;
    }
    if (results[2].status === 'fulfilled') for (const recipe of results[2].value.recipes) app.recipes.set('saved:' + recipe.id, recipe);
    const failures = results.filter(result => result.status === 'rejected');
    if (failures.length) notify(failures.map(result => result.reason.message).join(' · '), true);
    renderRecipeLibrary(); renderSlots(); await Promise.all([loadPrompts(), loadHistory(), loadCapacity()]);
  }
  function init() {
    if (app.ready) return true;
    const root = document.getElementById('lab-view'); if (!root) return false;
    app.ready = true; root.classList.add('persona-lab');
    app.recipes.set(BASE, {name: 'Baseline', pushes: [], phase: 'all'});
    const head = node('div', undefined, 'lab-heading');
    const heading = node('div'); heading.append(node('h2', 'Experiment workspace'), hint('Keep your recipes. Change the prompt. Compare up to six jobs.'));
    refs.model = node('span', 'Loading model…', 'lab-tag');
    const jumps = row(button('Arrange batch ↓', () => refs.composer?.scrollIntoView({behavior: 'smooth', block: 'start'})),
      button('Results ↓', () => refs.results?.scrollIntoView({behavior: 'smooth', block: 'start'})));
    jumps.classList.add('lab-jumps'); head.append(heading, refs.model, jumps, button('Refresh', loadSources)); root.append(head);
    refs.notice = node('div', '', 'lab-notice'); refs.notice.hidden = true; refs.notice.setAttribute('role', 'status'); refs.notice.setAttribute('aria-live', 'polite'); root.append(refs.notice);
    const sources = node('div', undefined, 'lab-sources'); root.append(sources);
    const prompts = panel('1 · Choose prompts', 'Search registered sources or add a custom prompt. Select up to six.');
    refs.search = input('search', '', 'Search prompt bank'); refs.search.placeholder = 'Search prompts…';
    refs.corpus = select([['', 'All corpora']], 'Corpus');
    refs.split = select([['discovery', 'Discovery'], ['heldout', 'Heldout'], ['all', 'All splits']], 'Source split');
    prompts.append(row(refs.corpus, refs.split), refs.search, hint('Heldout labels stay visible. These manual tests do not change the source split.'));
    refs.promptCount = node('span', '0 / 6 selected', 'lab-counter'); refs.selectedPrompts = node('div', undefined, 'lab-selection');
    prompts.append(refs.promptCount, refs.selectedPrompts);
    refs.promptRows = node('div', undefined, 'lab-prompt-list'); refs.promptStatus = node('small', 'Loading…', 'lab-dim');
    refs.previous = button('Previous', () => {app.offset = Math.max(0, app.offset - 12); loadPrompts();});
    refs.next = button('Next', () => {app.offset += 12; loadPrompts();});
    prompts.append(refs.promptRows, row(refs.previous, refs.promptStatus, refs.next));
    const custom = node('details', undefined, 'lab-custom'); custom.append(node('summary', 'Use a custom prompt'));
    refs.customName = input('text', '', 'Custom prompt label'); refs.customName.placeholder = 'Label (optional)';
    refs.customText = node('textarea'); refs.customText.rows = 4; refs.customText.placeholder = 'Write the task or conversation to compare…'; refs.customText.setAttribute('aria-label', 'Custom prompt text');
    custom.append(field('Label', refs.customName), field('Prompt', refs.customText), button('Add to selected prompts', () => {
      const text = refs.customText.value.trim(); if (!text) {notify('Write a custom prompt first.', true); return;}
      if (app.selectedPrompts.length >= MAX) {notify('Remove a selected prompt to make room.', true); return;}
      const key = `custom:${++app.serial}`; app.prompts.set(key, {id: key, custom: true, prompt: text,
        label: refs.customName.value.trim() || `Custom prompt ${app.serial}`, corpus: 'custom', split: 'manual'});
      toggleSelection('prompt', key); refs.customText.value = refs.customName.value = '';
    })); prompts.append(custom); sources.append(prompts);
    const recipes = panel('2 · Choose recipes', 'Select saved recipes or make a variation. Baseline adds no activation push.');
    refs.recipeCount = node('span', '', 'lab-counter'); refs.recipeLibrary = node('div', undefined, 'lab-recipe-list'); recipes.append(refs.recipeCount, refs.recipeLibrary);
    const editor = node('div', undefined, 'lab-editor'); editor.append(node('h4', 'Recipe editor'));
    refs.editing = hint(''); refs.recipeName = input('text', 'New recipe', 'Recipe name'); refs.recipeName.maxLength = 120;
    refs.recipeName.oninput = () => {app.editor.name = refs.recipeName.value;};
    refs.recipePhase = select(phases, 'Recipe phase'); refs.recipePhase.onchange = () => {app.editor.phase = refs.recipePhase.value;};
    refs.pushRows = node('div', undefined, 'lab-push-list'); editor.append(refs.editing, row(field('Name', refs.recipeName), field('Apply during', refs.recipePhase)), refs.pushRows);
    const add = node('details', undefined, 'lab-custom'); add.append(node('summary', 'Add a neuron or trait push'));
    const kind = select([['neuron', 'Neuron input'], ['column', 'Output column'], ['trait', 'Trait direction']], 'Push type');
    const layer = input('number', '15', 'Layer'); layer.min = 0; layer.max = 63;
    const neuron = input('number', '0', 'Neuron'); neuron.min = 0; neuron.max = 17407;
    const trait = select(traitNames.map(name => [name, readable(name)]), 'Trait'); trait.hidden = true;
    const dose = input('number', '.25', 'Relative dose'); dose.min = -4; dose.max = 4; dose.step = .05;
    kind.onchange = () => {neuron.hidden = kind.value === 'trait'; trait.hidden = kind.value !== 'trait';};
    add.append(row(field('Site', kind), field('Layer', layer)), row(field('Neuron / trait', row(neuron, trait)), field('Dose', dose)),
      button('Add push', () => {
        if (!layer.value.trim() || !dose.value.trim() || (kind.value !== 'trait' && !neuron.value.trim())) {
          notify('Fill in the layer, target and relative dose before adding a push.', true); return;
        }
        const push = {kind: kind.value, layer: Number(layer.value), dose: Number(dose.value)};
        if (kind.value === 'trait') push.trait = trait.value; else push.neuron = Number(neuron.value);
        app.editor.pushes.push(push); renderEditor();
      }));
    editor.append(add, hint('Relative doses keep each push tied to its layer calibration. Positive and negative signs are explicit.'));
    refs.saveRecipe = button('Save as new', () => saveRecipe(false)); refs.updateRecipe = button('Update saved', () => saveRecipe(true));
    editor.append(row(button('Use in batch', useEditor, 'lab-primary'), refs.saveRecipe, refs.updateRecipe));
    refs.variantFactors = input('text', '-1, 0, 0.5, 1, 1.5, 2', 'Dose multipliers');
    editor.append(field('Quick variants · multiply every dose by', refs.variantFactors), button('Select these variants', doseVariants),
      hint('Variants replace the recipe selection, not your saved recipes. A zero-dose recipe keeps its explicit zero pushes.'));
    recipes.append(editor); sources.append(recipes);
    const composer = panel('3 · Arrange your batch', 'Up to six concurrent jobs. Cards are shared with overnight sweeps; queued jobs wait for a card.');
    refs.composer = composer; composer.id = 'lab-composer';
    refs.fleetStatus = node('strong', 'Checking worker reachability…', 'lab-fleet-status');
    refs.fleetCards = node('div', undefined, 'lab-fleet-cards');
    composer.append(refs.fleetStatus, refs.fleetCards, hint('Reachable means the worker identity was verified. Busy / idle is unknown.'));
    const layouts = row(node('span', 'Recipes × prompts', 'lab-caption'));
    for (const [r, p] of [[6, 1], [1, 6], [2, 3], [3, 2]]) layouts.append(button(`${r} × ${p}`, () => arrange(r, p)));
    layouts.append(button('Selected combinations', () => {
      const n = app.selectedRecipes.length * app.selectedPrompts.length;
      if (!n || n > MAX) {notify('Select recipes and prompts whose total combinations are between one and six.', true); return;}
      arrange(app.selectedRecipes.length, app.selectedPrompts.length);
    }), button('Paired baseline', pairedBaseline));
    refs.slots = node('div', undefined, 'lab-slots'); refs.capacity = node('strong', '', 'lab-capacity');
    refs.addSlot = button('Add a job', () => {if (app.slots.length < MAX) app.slots.push({prompt: app.selectedPrompts[0] || '', recipe: app.selectedRecipes[0] || BASE}); renderSlots();});
    composer.append(layouts, row(refs.capacity, refs.addSlot), refs.slots);
    refs.seed = input('number', '7035', 'Shared batch seed'); refs.seed.min = 0; refs.seed.max = 4294967295;
    refs.budget = input('number', '2048', 'Token budget per job'); refs.budget.min = 256; refs.budget.max = 4096; refs.budget.step = 256; refs.budget.oninput = renderSlots;
    refs.thinking = input('checkbox', '', 'Enable thinking');
    refs.previewNote = hint('Preview run: budgets below 2,048 tokens are marked as smoke tests, not full evaluations.'); refs.previewNote.classList.add('lab-preview-tag');
    refs.run = button('Run batch', runBatch, 'lab-primary');
    refs.runtimeNote = hint('Checking worker context and thinking support…');
    composer.append(row(field('Shared seed', refs.seed), field('Tokens per job', refs.budget), field('Enable thinking', refs.thinking), refs.run),
      hint('Every job uses the same seed. Baselines occupy a slot; no extra runs are added behind the scenes.'), refs.runtimeNote, refs.previewNote); root.append(composer);
    refs.results = panel('Batch results', 'Each job keeps its prompt, recipe, completion status and provenance.');
    refs.results.id = 'lab-results';
    refs.history = select([['', 'Choose a batch']], 'Batch history'); refs.history.onchange = () => {if (refs.history.value) showBatch(refs.history.value);};
    refs.resultStatus = node('p', 'Run a batch or choose one from history.', 'lab-dim');
    refs.resultCards = node('div', undefined, 'lab-results');
    refs.results.append(row(refs.history, button('Refresh results', () => app.activeBatch ? showBatch(app.activeBatch) : loadHistory())), refs.resultStatus, refs.resultCards); root.append(refs.results);
    const filter = () => {app.offset = 0; loadPrompts();}; refs.corpus.onchange = refs.split.onchange = filter;
    refs.search.oninput = () => {clearTimeout(searchTimer); searchTimer = setTimeout(filter, 280);};
    renderEditor(); renderSelections(); renderRecipeLibrary(); renderSlots(); loadSources(); return true;
  }
  window.PersonaLab = {
    open() {
      // tab('lab') calls open(); only the map handoff calls tab(), so this
      // entry point must never navigate again.
      const mounted = app.ready;
      if (init() && mounted) loadCapacity();
    },
    useCurrentRecipe(recipe) {
      if (!init()) return;
      const value = {name: recipe.name || 'Map recipe', pushes: copy(recipe.pushes || []), phase: recipe.phase || 'all'};
      const key = addLocalRecipe(value, false); app.selectedRecipes = [key];
      loadEditor(value); app.slots.forEach(slot => {slot.recipe = key;}); renderRecipeLibrary(); renderSlots();
      if (typeof window.tab === 'function') window.tab('lab');
      else if (typeof tab === 'function') tab('lab');
      document.getElementById('lab-view').classList.remove('hidden');
      notify('Map recipe selected. Choose prompts, arrange your jobs and run the batch.');
    },
    refresh() {if (init()) return loadSources();}
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, {once: true}); else init();
})();
