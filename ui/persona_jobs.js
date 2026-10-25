/* App-owned schedules. Times are resolved by the server in the chosen IANA zone. */
(() => {
  'use strict';
  const ZONE = 'America/Los_Angeles';
  const names = {big_five_expand: 'Big Five corpus expansion', neighborhood_hillclimb: 'Neighborhood hillclimb', studio_session: 'Interactive persona studio'};
  const terminal = new Set(['complete', 'partial', 'failed', 'cancelled', 'expired']);
  const app = {ready: false, enabled: false, loadError: '', jobs: [], selected: null, detail: null,
    submitting: false, cancelling: false, listSerial: 0, detailSerial: 0};
  const refs = {};
  let timer;
  function node(tag, text, cls) {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    if (cls) element.className = cls;
    return element;
  }
  function button(text, action, cls = '') {
    const element = node('button', text, cls); element.type = 'button'; element.onclick = action; return element;
  }
  function input(type, value) {const element = node('input'); element.type = type; element.value = value; return element;}
  function field(label, control, help) {
    const element = node('label', undefined, 'pj-field'); element.append(node('span', label), control);
    if (help) element.append(node('small', help)); return element;
  }
  function row(...children) {const element = node('div', undefined, 'pj-row'); element.append(...children); return element;}
  function hint(text) {return node('p', text, 'pj-hint');}
  function number(value) {return Number.isFinite(value) ? value.toLocaleString() : '—';}
  function readable(value) {return String(value || 'unknown').replaceAll('_', ' ');}
  function units(job) {return job.kind === 'studio_session' ? 'ready sessions' : job.counts_unit === 'observations' || job.kind === 'neighborhood_hillclimb' ? 'observations' : 'synthetic entries';}
  function localClock(zone, when = new Date()) {
    const parts = new Intl.DateTimeFormat('en-CA', {timeZone: zone, year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', hourCycle: 'h23'}).formatToParts(when);
    const values = Object.fromEntries(parts.map(part => [part.type, part.value]));
    return `${values.year}-${values.month}-${values.day}T${values.hour}:${values.minute}`;
  }
  function nextStart(zone) {
    const now = localClock(zone), day = now.slice(0, 10);
    if (now < `${day}T02:00`) return `${day}T02:00`;
    const [year, month, date] = day.split('-').map(Number);
    // Calendar arithmetic only. This is deliberately not an offset conversion.
    return `${new Date(Date.UTC(year, month - 1, date + 1)).toISOString().slice(0, 10)}T02:00`;
  }
  function date(value, zone) {
    if (!value) return 'Not recorded';
    if (typeof value === 'object' && Number.isFinite(value.time)) value = value.time;
    if (typeof value === 'number') value *= 1000;
    try {
      const instant = new Date(value);
      if (!Number.isFinite(instant.valueOf())) return String(value);
      return new Intl.DateTimeFormat('en-US', {timeZone: zone || ZONE, month: 'short', day: 'numeric', year: 'numeric',
        hour: 'numeric', minute: '2-digit', timeZoneName: 'shortOffset'}).format(instant);
    } catch (_) {return String(value);}
  }
  async function request(url, payload) {
    const response = await fetch(url, payload === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    const value = await response.json().catch(() => ({error: `Request failed (${response.status})`}));
    if (!response.ok) throw Error(value.error || response.statusText || 'Request failed.');
    return value;
  }
  function notice(text = '', error = false) {
    refs.notice.textContent = text; refs.notice.hidden = !text; refs.notice.classList.toggle('pj-error', error);
  }
  function ensure() {
    if (app.ready) return true;
    const root = document.getElementById('jobs-view'); if (!root) return false;
    app.ready = true; refs.root = root; root.classList.add('persona-jobs');
    const heading = node('div', undefined, 'pj-heading'), titles = node('div');
    titles.append(node('p', 'LOCAL RESEARCH · DURABLE SCHEDULES', 'pj-eyebrow'), node('h2', 'Jobs & schedules'),
      hint('Plan a future run, follow its progress, and stop only the work that belongs to that job.'));
    heading.append(titles, button('Refresh status', refresh));
    refs.service = node('div', undefined, 'pj-service'); refs.service.setAttribute('role', 'status');
    refs.notice = node('p', undefined, 'pj-notice'); refs.notice.hidden = true; refs.notice.setAttribute('role', 'status');
    const layout = node('div', undefined, 'pj-layout');
    refs.form = node('form', undefined, 'pj-panel pj-form');
    const formHead = row(node('h3', 'Plan a job'), button('Reset form', reset, 'pj-text-button'));
    formHead.classList.add('pj-spread'); refs.form.append(formHead);
    refs.kind = node('select');
    for (const [value, text] of Object.entries(names)) {const option = node('option', text); option.value = value; refs.kind.append(option);}
    refs.kind.value = 'big_five_expand'; refs.kind.onchange = () => {renderKind(); preview();};
    refs.name = input('text', 'Big Five discovery expansion'); refs.name.maxLength = 120; refs.name.required = true;
    refs.form.append(field('Job type', refs.kind), field('Name', refs.name));
    refs.kindNote = hint(''); refs.kindNote.classList.add('pj-kind-note'); refs.form.append(refs.kindNote);
    refs.start = input('datetime-local', nextStart(ZONE)); refs.start.required = true; refs.start.step = '60';
    refs.zone = input('text', ZONE); refs.zone.required = true; refs.zone.spellcheck = false;
    refs.start.oninput = preview; refs.zone.oninput = preview;
    refs.form.append(field('Scheduled start', refs.start), field('Time zone', refs.zone,
      'Use an IANA zone, such as America/Los_Angeles. The server checks daylight-saving transitions.'));
    refs.duration = input('number', 8); refs.duration.min = '0.25'; refs.duration.max = '16'; refs.duration.step = '0.25'; refs.duration.required = true; refs.duration.oninput = preview;
    refs.seed = input('number', 7035); refs.seed.min = '0'; refs.seed.max = '4294967295'; refs.seed.step = '1'; refs.seed.required = true;
    refs.form.append(row(field('Duration (hours)', refs.duration), field('Seed', refs.seed)));
    refs.target = input('number', 35000); refs.target.min = '1'; refs.target.max = '40000'; refs.target.step = '1'; refs.target.required = true;
    refs.maximum = input('number', 40000); refs.maximum.min = '1'; refs.maximum.max = '40000'; refs.maximum.step = '1'; refs.maximum.required = true;
    refs.targetField = field('Target entries', refs.target); refs.maxField = field('Hard entry cap', refs.maximum);
    refs.form.append(row(refs.targetField, refs.maxField));
    refs.neighborhood = node('div', undefined, 'pj-neighborhood');
    refs.targets = node('textarea'); refs.targets.rows = 3; refs.targets.placeholder = 'L15 N242\nL28 N11876'; refs.targets.spellcheck = false;
    refs.neighborhood.append(field('Target neurons', refs.targets, 'One coordinate per line: L15 N242 or 15:242. Coordinates outside the current map can be investigated at calibrated layers.'));
    refs.targetNote = hint('Unmapped target: new measurements required. Missing or flat activity is a question to test, not an inferred link.');
    refs.neighborhood.append(refs.targetNote);
    const corpora = node('fieldset', undefined, 'pj-corpora'); corpora.append(node('legend', 'Discovery corpora'));
    refs.corpora = [];
    for (const [value, label] of [['big_five', 'Big Five'], ['scotus', 'SCOTUS'], ['skippy', 'Skippy books']]) {
      const check = input('checkbox', value); check.checked = true; refs.corpora.push(check);
      const wrap = node('label'); wrap.append(check, node('span', label)); corpora.append(wrap);
    }
    refs.neighborhood.append(corpora); refs.form.append(refs.neighborhood);
    refs.preview = node('div', undefined, 'pj-preview'); refs.preview.setAttribute('aria-live', 'polite');
    const fleet = node('div', undefined, 'pj-fleet');
    fleet.append(node('strong', 'John · local 27B manager'), node('span', 'Six Quincies · shared GPU fleet'),
      hint('Start depends on workboard admission. Thermal, memory and ownership guards remain active; scheduled time is not a promise that all cards are idle.'));
    refs.submit = node('button', 'Schedule job', 'pj-primary'); refs.submit.type = 'submit'; refs.submit.disabled = true;
    refs.form.append(refs.preview, fleet, refs.submit, hint('Schedules persist in the app. Held-out prompts stay excluded from adaptive feedback.'));
    refs.form.onsubmit = event => {event.preventDefault(); submit();};
    const workspace = node('div', undefined, 'pj-workspace');
    const history = node('section', undefined, 'pj-panel'), historyHead = row(node('h3', 'Scheduled & recent jobs'));
    refs.listCount = node('span', 'Loading…', 'pj-hint'); historyHead.append(refs.listCount); historyHead.classList.add('pj-spread');
    refs.list = node('div', undefined, 'pj-job-list'); history.append(historyHead, refs.list);
    refs.detail = node('section', undefined, 'pj-panel pj-detail'); refs.detail.append(hint('Select a job to view progress, the resolved start time and its event history.'));
    workspace.append(history, refs.detail); layout.append(refs.form, workspace); root.append(heading, refs.service, refs.notice, layout);
    renderKind(); preview(); return true;
  }
  function reset() {
    refs.kind.value = 'big_five_expand'; refs.name.value = 'Big Five discovery expansion'; refs.zone.value = ZONE;
    refs.start.value = nextStart(ZONE); refs.duration.value = '8'; refs.seed.value = '7035'; refs.target.value = '35000';
    refs.maximum.value = '40000'; refs.targets.value = ''; refs.corpora.forEach(item => {item.checked = true;});
    renderKind(); preview(); notice();
  }
  function renderKind() {
    const neighborhood = refs.kind.value === 'neighborhood_hillclimb';
    const studio = refs.kind.value === 'studio_session';
    refs.targetField.hidden = studio; refs.maxField.hidden = studio;
    refs.neighborhood.hidden = !neighborhood;
    refs.targets.required = neighborhood;
    refs.targetField.firstElementChild.textContent = neighborhood ? 'Valid observation target' : 'Accepted entry target';
    refs.maxField.firstElementChild.textContent = neighborhood ? 'Hard observation cap' : 'Hard entry cap';
    refs.kindNote.textContent = studio ? 'Start John and the six Quincies for interactive persona experiments. No automatic battery runs; models stay available until the session ends or you stop it.' : neighborhood ?
      'Adapt discovery experiments around the selected neurons. Count valid observations; newly observed co-activity still needs separate validation.' :
      'Generate and deduplicate new Big Five themed discovery prompts. These are synthetic task entries, not validated psychometric items or personality scores.';
    if (['Big Five discovery expansion', 'Neighborhood discovery search', 'Interactive persona studio'].includes(refs.name.value)) {
      refs.name.value = studio ? 'Interactive persona studio' : neighborhood ? 'Neighborhood discovery search' : 'Big Five discovery expansion';
    }
  }
  function preview() {
    refs.preview.replaceChildren();
    try {
      const zone = refs.zone.value.trim(); localClock(zone);
      if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(refs.start.value)) throw Error('Choose a start date and time.');
      refs.preview.append(node('span', 'PROPOSED START', 'pj-eyebrow'), node('strong', `${refs.start.value.replace('T', ' at ')} · ${zone}`),
        node('small', `${refs.duration.value || '—'} hours · server resolves the exact offset when saved`));
    } catch (error) {refs.preview.append(node('span', error instanceof RangeError ? 'Enter a valid IANA time zone.' : error.message, 'pj-error-text'));}
  }
  function integer(control, label, min, max) {
    const value = Number(control.value);
    if (!String(control.value).trim() || !Number.isInteger(value) || value < min || value > max) throw Error(`${label} must be a whole number from ${min} to ${max}.`);
    return value;
  }
  function targets() {
    const values = [], seen = new Set();
    const current = typeof state !== 'undefined' ? state : window.state;
    for (const line of refs.targets.value.split(/[\n;]+/).map(value => value.trim()).filter(Boolean)) {
      const match = line.match(/^L\s*(\d+)\s*N\s*(\d+)$/i) || line.match(/^(\d+)[\s,:/]+(\d+)$/);
      if (!match) throw Error(`Cannot read target “${line}”. Use one L15 N242 coordinate per line.`);
      const layer = Number(match[1]), neuron = Number(match[2]), key = `${layer}:${neuron}`;
      if (!Number.isSafeInteger(layer) || !Number.isSafeInteger(neuron)) throw Error('Neuron coordinates must be whole numbers.');
      if (layer >= (current?.model?.layers || 64) || neuron >= (current?.model?.intermediate || 17408)) throw Error(`${coordinate({layer, neuron})} is outside this model’s dimensions.`);
      if (Array.isArray(current?.calibrated_layers) && !current.calibrated_layers.includes(layer)) throw Error(`Layer ${layer} has no calibration yet.`);
      if (!seen.has(key)) {seen.add(key); values.push({layer, neuron});}
    }
    if (!values.length) throw Error('Choose at least one target neuron.');
    return values;
  }
  function coordinate(value) {return `L${value.layer} N${value.neuron}`;}
  function payload() {
    const timezone = refs.zone.value.trim(); localClock(timezone);
    if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(refs.start.value)) throw Error('Choose a start date and time.');
    const duration = Number(refs.duration.value) * 60;
    if (!String(refs.duration.value).trim() || !Number.isInteger(duration) || duration < 15 || duration > 960 || duration % 15) throw Error('Duration must be in 15-minute increments from 15 minutes to 16 hours.');
    const result = {kind: refs.kind.value, name: refs.name.value.trim(), local_start: refs.start.value, timezone,
      duration_minutes: duration, seed: integer(refs.seed, 'Seed', 0, 4294967295),
      target_entries: refs.kind.value === 'studio_session' ? 1 : integer(refs.target, 'Target', 1, 40000),
      max_entries: refs.kind.value === 'studio_session' ? 1 : integer(refs.maximum, 'Hard cap', 1, 40000)};
    if (!result.name || result.name.length > 120) throw Error('Give this job a name of 1 to 120 characters.');
    if (result.target_entries > result.max_entries) throw Error('The target cannot exceed the hard cap.');
    if (result.kind === 'neighborhood_hillclimb') {
      result.targets = targets(); result.corpora = refs.corpora.filter(item => item.checked).map(item => item.value);
      if (!result.corpora.length) throw Error('Select at least one discovery corpus.');
    }
    return result;
  }
  function controls() {
    refs.submit.disabled = app.submitting || !app.enabled || Boolean(app.loadError);
    refs.submit.textContent = app.submitting ? 'Saving schedule…' : 'Schedule job';
  }
  async function submit() {
    if (app.submitting || !app.enabled || app.loadError) return;
    try {
      if (refs.form.reportValidity && !refs.form.reportValidity()) return;
      const body = payload(); app.submitting = true; controls(); notice();
      const job = await request('/api/jobs', body);
      if (!job.id) throw Error('The server did not return a saved job ID. Refresh before trying again.');
      app.selected = job.id; app.detail = null;
      refs.detail.replaceChildren(hint('Loading saved job…'));
      notice(`Schedule saved: ${body.name}. Its start and admission status are shown in the job details.`);
      await Promise.all([loadJobs(), loadDetail(job.id)]);
      refs.detail.scrollIntoView({behavior: 'smooth', block: 'nearest'});
    } catch (error) {notice(error.message || 'Unable to save the schedule.', true);}
    finally {app.submitting = false; controls();}
  }
  function renderService(value) {
    refs.service.replaceChildren();
    if (!value.enabled) {
      refs.service.dataset.state = 'offline'; refs.service.append(node('strong', 'Scheduling is not enabled'),
        node('span', 'Job history is readable. An operator must configure the local scheduler before new jobs can be saved.')); return;
    }
    const alive = value.scheduler?.alive;
    refs.service.dataset.state = alive === true ? 'online' : 'waiting';
    refs.service.append(node('strong', alive === true ? 'Scheduler online' : 'Schedules enabled · waiting for scheduler'),
      node('span', alive === true ? 'Jobs start after their scheduled time when compute admission succeeds.' : 'Saved jobs remain durable, but cannot begin until the scheduler is running.'));
    if (value.scheduler?.last_heartbeat) refs.service.append(node('small', `Last heartbeat: ${date(value.scheduler.last_heartbeat, value.timezone || ZONE)}`));
  }
  async function loadJobs() {
    const serial = ++app.listSerial;
    try {
      const value = await request('/api/jobs'); if (serial !== app.listSerial) return;
      if (!Array.isArray(value.jobs)) throw Error('The server returned an invalid job list.');
      app.enabled = value.enabled === true; app.loadError = ''; app.jobs = value.jobs;
      renderService(value); renderList();
    } catch (error) {
      if (serial !== app.listSerial) return;
      app.loadError = error.message; refs.service.dataset.state = 'offline';
      refs.service.replaceChildren(node('strong', 'Could not refresh schedules'), node('span', error.message));
    } finally {if (serial === app.listSerial) controls();}
  }
  function renderList() {
    const focusedId = document.activeElement?.dataset?.jobId;
    refs.list.replaceChildren(); refs.listCount.textContent = `${app.jobs.length} job${app.jobs.length === 1 ? '' : 's'}`;
    if (!app.jobs.length) {refs.list.append(node('p', 'No saved jobs yet. Set a future start time and schedule the first one.', 'pj-empty')); return;}
    for (const job of app.jobs) {
      const item = button('', () => choose(job.id), 'pj-job'); item.classList.toggle('pj-selected', job.id === app.selected);
      item.dataset.jobId = job.id;
      item.setAttribute('aria-pressed', String(job.id === app.selected));
      const title = node('div', undefined, 'pj-job-title'); title.append(node('strong', job.name || names[job.kind] || 'Research job'), badge(job.state));
      const start = node('span', date(job.start_at, job.timezone), 'pj-job-time');
      const progress = node('small', `${number(job.counts?.accepted_unique)} / ${number(job.target_entries)} ${units(job)} · ${job.timezone || ZONE}`);
      item.append(title, start, progress); refs.list.append(item);
      if (job.id === focusedId) item.focus({preventScroll: true});
    }
  }
  function badge(state) {const value = node('span', readable(state), 'pj-badge'); value.dataset.state = state || 'unknown'; return value;}
  async function choose(id) {
    if (!id) return; app.selected = id; app.detail = null; renderList();
    refs.detail.replaceChildren(hint('Loading job details…')); await loadDetail(id);
  }
  async function loadDetail(id) {
    if (!id) return;
    const serial = ++app.detailSerial;
    try {
      const value = await request(`/api/job?${new URLSearchParams({id})}`);
      if (serial !== app.detailSerial || app.selected !== id) return;
      if (value.id !== id) throw Error('The returned job does not match the selection.');
      const changed = JSON.stringify(app.detail) !== JSON.stringify(value);
      app.detail = value; if (changed) renderDetail(value);
    } catch (error) {
      if (serial !== app.detailSerial || app.selected !== id) return;
      refs.detail.replaceChildren(node('p', error.message, 'pj-error-text'), button('Retry details', () => loadDetail(id)));
    }
  }
  function renderDetail(job) {
    const focused = document.activeElement, wasCancel = focused?.dataset?.jobCancel === job.id;
    refs.detail.replaceChildren();
    const heading = row(node('h3', job.name || names[job.kind] || 'Research job'), badge(job.state)); heading.classList.add('pj-spread');
    refs.detail.append(heading, hint(`${names[job.kind] || readable(job.kind)} · ${job.id}`));
    const times = node('dl', undefined, 'pj-facts');
    for (const [label, value] of [['Scheduled start', date(job.start_at, job.timezone)], ['Time zone', job.timezone],
      ['Deadline', date(job.deadline, job.timezone)], ['Started', job.started ? date(job.started, job.timezone) : 'Not started'],
      ['Finished', job.finished ? date(job.finished, job.timezone) : 'Not finished'], ['Compute', 'John · local 27B manager + six Quincies']]) {
      if (value) times.append(node('dt', label), node('dd', value));
    }
    refs.detail.append(times);
    const counts = job.counts || {}, statistics = node('div', undefined, 'pj-statistics');
    for (const [label, value] of [[job.kind === 'studio_session' ? 'Session ready' : job.kind === 'neighborhood_hillclimb' ? 'Valid observations' : 'Accepted unique', counts.accepted_unique],
      ['Requested', counts.requested], ['Rejected', counts.rejected], ['Errors', counts.errors]]) {
      const stat = node('div'); stat.append(node('strong', number(value)), node('span', label)); statistics.append(stat);
    }
    refs.detail.append(statistics);
    const progress = node('progress'); progress.max = Math.max(1, Number(job.target_entries) || 1);
    if (Number.isFinite(counts.accepted_unique)) progress.value = Math.max(0, Math.min(progress.max, counts.accepted_unique));
    progress.setAttribute('aria-label', `${units(job)} completed toward target`);
    refs.detail.append(progress, hint(`Target ${number(job.target_entries)} ${units(job)} · hard cap ${number(job.max_entries)} · ${number(counts.requests)} requests`));
    refs.detail.append(hint(job.kind === 'studio_session' ? 'A ready session provides the model workers. Persona Studio and recipe experiments record their own prompts and results separately. This session ends at its deadline or on your stop request.' : job.kind === 'neighborhood_hillclimb' ?
      'Completed observations are not proven connections. Flat or missing activity remains an experimental finding; held-out data is not used for adaptive feedback.' :
      'Accepted means a new, deduplicated synthetic discovery prompt. It does not imply a validated psychometric item or a demonstrated personality effect.'));
    if (Array.isArray(job.targets) && job.targets.length) refs.detail.append(hint(`Targets: ${job.targets.map(coordinate).join(' · ')}`));
    if (Array.isArray(job.corpora) && job.corpora.length) refs.detail.append(hint(`Discovery corpora: ${job.corpora.map(readable).join(', ')}`));
    if (job.error) refs.detail.append(node('p', String(job.error), 'pj-notice pj-error'));
    if (!terminal.has(job.state)) {
      const stopping = job.cancel_requested || job.state === 'stopping';
      const cancel = button(stopping ? 'Stop requested' : job.state === 'queued' ? 'Cancel scheduled job' : 'Stop this job’s work', () => cancelJob(job), 'pj-stop');
      cancel.dataset.jobCancel = job.id; cancel.disabled = Boolean(stopping || app.cancelling);
      refs.detail.append(cancel, hint('Cancelling a queued job prevents its start. A running job stops cooperatively; unrelated jobs remain untouched.'));
      if (wasCancel && !cancel.disabled) cancel.focus({preventScroll: true});
    }
    const events = node('section', undefined, 'pj-events'); events.append(node('h4', 'Job events'));
    if (!Array.isArray(job.events) || !job.events.length) events.append(hint('No events recorded yet.'));
    else {
      const list = node('ol');
      for (const event of job.events.slice(-40).reverse()) {
        const item = node('li'), line = row(node('strong', readable(event.event)), node('time', date(event.time, job.timezone)));
        line.classList.add('pj-spread'); item.append(line);
        const description = eventDescription(event.data);
        if (description) item.append(node('p', description));
        list.append(item);
      }
      events.append(list, hint('Showing the latest 40 recorded events. Full request payloads remain in the local job records.'));
    }
    refs.detail.append(events);
  }
  function eventDescription(data) {
    if (!data || typeof data !== 'object') return '';
    const values = [];
    for (const key of ['message', 'reason', 'detail', 'error', 'state', 'unit', 'host', 'accepted_unique', 'requested', 'rejected', 'requests', 'errors', 'observations']) {
      const value = data[key]; if (typeof value === 'string' || typeof value === 'number') values.push(`${readable(key)}: ${String(value).slice(0, 700)}`);
    }
    return values.join(' · ');
  }
  async function cancelJob(job) {
    if (app.cancelling || job.cancel_requested || terminal.has(job.state)) return;
    app.cancelling = true; renderDetail(job);
    try {
      const value = await request('/api/job/cancel', {id: job.id});
      if (value.id !== job.id) throw Error('The returned job does not match the stop request. Refresh its status before retrying.');
      if (app.selected === job.id) {app.detail = value; renderDetail(value);}
      notice(value.state === 'cancelled' ? 'The scheduled job was cancelled.' : 'Stop requested for this job’s own work.');
      await loadJobs();
    } catch (error) {notice(error.message || 'Unable to request a stop.', true);}
    finally {app.cancelling = false; if (app.detail) renderDetail(app.detail);}
  }
  function schedulePoll() {
    clearTimeout(timer); timer = setTimeout(async () => {
      if (!document.hidden && !refs.root.classList.contains('hidden')) await refresh();
      else schedulePoll();
    }, 5000);
  }
  async function refresh() {
    if (!ensure()) return;
    await Promise.all([loadJobs(), app.selected ? loadDetail(app.selected) : Promise.resolve()]); schedulePoll();
  }
  function open() {if (ensure()) refresh();}
  function fromNeuron(target) {
    if (!target || !Number.isInteger(target.layer) || target.layer < 0 || !Number.isInteger(target.neuron) || target.neuron < 0 || !ensure()) return;
    refs.kind.value = 'neighborhood_hillclimb'; refs.name.value = `Neighborhood search · ${coordinate(target)}`;
    refs.targets.value = coordinate(target); renderKind(); preview();
    window.PersonaNeighbors?.close();
    const changeTab = typeof tab === 'function' ? tab : window.tab;
    if (typeof changeTab === 'function') changeTab('jobs');
    else {refs.root.classList.remove('hidden'); open();}
    notice('Target selected. Review the discovery corpora, budget and future start before scheduling.');
    refs.targets.focus({preventScroll: true}); refs.form.scrollIntoView({behavior: 'smooth', block: 'start'});
  }
  window.PersonaJobs = {open, fromNeuron};
})();
