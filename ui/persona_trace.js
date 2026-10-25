/* Actual consumed-token playback and first-next-token intervention lens. */
(()=>{'use strict';
let refs={},campaign=null,poll=null,playing=null,index=0,armIndex=0,flat=[];
const n=(tag,text,cls)=>{const x=document.createElement(tag);if(text!==undefined)x.textContent=text;if(cls)x.className=cls;return x;};
const b=(label,fn)=>{const x=n('button',label);x.type='button';x.onclick=fn;return x;};
const api=async(path,value)=>{const r=await fetch(path,value?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)}:undefined);const v=await r.json();if(!r.ok)throw Error(v.error||'Request failed');return v;};
function field(label,control){const l=n('label',label);l.append(control);return l;}
function ensure(){if(refs.root)return;
 const root=refs.root=n('section',undefined,'details hidden pt-root');root.id='trace-view';document.querySelector('footer').before(root);
 root.append(n('h2','Live tokens & intervention lens'),n('p','Watch selected neurons through the prompt and generated continuation, then compare the full next-token distribution at the shared final prompt position. Inference uses the active leased fleet.','pt-muted'));
 const config=n('div',undefined,'pt-config');refs.mode=n('select');for(const [value,label]of [['dose_sweep','One prompt × six recipe strengths'],['three_prompts','Three prompts × baseline and recipe']]){const o=n('option',label);o.value=value;refs.mode.append(o);}
 refs.prompt=n('textarea');refs.prompt.rows=5;refs.prompt.value='A friend is excited about a plan that has an obvious flaw. How would you respond?';refs.prompt.placeholder='For three prompts, separate each with a line containing ---';
 refs.recipe=n('textarea');refs.recipe.rows=3;refs.recipe.value='[]';refs.recipe.spellcheck=false;
 refs.neurons=n('textarea');refs.neurons.rows=2;refs.neurons.value='L14N8949 L13N7956';
 refs.budget=n('input');refs.budget.type='number';refs.budget.min=1;refs.budget.max=512;refs.budget.value=128;
 refs.phase=n('select');for(const v of ['all','prefill','decode']){const o=n('option',v);o.value=v;refs.phase.append(o);}
 refs.seed=n('input');refs.seed.type='number';refs.seed.value=7035;
 config.append(field('Six-card arrangement',refs.mode),field('Prompt(s); separate three with ---',refs.prompt),field('Recipe interventions (JSON)',refs.recipe),field('Observed neurons, e.g. L14N8949',refs.neurons));
 const options=n('div',undefined,'pt-controls');options.append(field('Generated token budget',refs.budget),field('Steering phase',refs.phase),field('Seed',refs.seed));
 const copy=b('Use current map recipe',()=>{refs.recipe.value=JSON.stringify(typeof pushes!=='undefined'?pushes:[],null,2);const coords=typeof pushes!=='undefined'?pushes.filter(p=>Number.isInteger(p.neuron)).map(p=>'L'+p.layer+'N'+p.neuron):[];if(typeof selectedUnit!=='undefined'&&selectedUnit)coords.push('L'+selectedUnit.layer+'N'+selectedUnit.neuron);if(coords.length)refs.neurons.value=[...new Set(coords)].join(' ');});
 refs.run=b('Run six comparisons',start);refs.status=n('p','Start a fleet session in Jobs if the workers are asleep.','pt-muted');
 const corpora=n('select');for(const v of ['big_five','scotus','skippy']){const o=n('option',v);o.value=v;corpora.append(o);}
 const corpusButton=b('Use corpus prompts',async()=>{try{const v=await api('/api/prompts?corpus='+corpora.value+'&limit=3');const rows=v.prompts||v.rows||[];refs.prompt.value=rows.slice(0,refs.mode.value==='three_prompts'?3:1).map(p=>p.prompt||p.text).join('\n---\n');}catch(e){refs.status.textContent=e.message;}});
 root.append(config,options,copy,field('Prompt corpus',corpora),corpusButton,refs.run,refs.status);
 const search=n('input');search.placeholder='Search recorded token effects';search.maxLength=100;search.setAttribute('aria-label','Search recorded token effects');const found=n('div',undefined,'pt-history');
 const searchButton=b('Find effects',async()=>{try{const result=await api('/api/trace/search?q='+encodeURIComponent(search.value));found.replaceChildren(n('p',result.scope,'pt-muted'));for(const hit of result.matches)found.append(b(JSON.stringify(hit.text)+' · Δ '+hit.delta_logp.toFixed(3)+' · '+hit.label,()=>{armIndex=hit.arm;load(hit.campaign_id);}));}catch(e){refs.status.textContent=e.message;}});
 root.append(search,searchButton,found);refs.history=n('div',undefined,'pt-history');root.append(refs.history);
 refs.arms=n('div',undefined,'pt-arms');refs.playback=n('section',undefined,'pt-playback');
 const playbackControls=n('div',undefined,'pt-controls');refs.slider=n('input');refs.slider.type='range';refs.slider.min=0;refs.slider.max=0;refs.slider.value=0;refs.slider.setAttribute('aria-label','Consumed token position');refs.slider.oninput=()=>{index=Number(refs.slider.value);refs.follow.checked=false;drawFrame();};
 refs.follow=n('input');refs.follow.type='checkbox';refs.follow.checked=true;
 playbackControls.append(b('Play / pause',()=>{if(playing){clearInterval(playing);playing=null;return;}refs.follow.checked=false;playing=setInterval(()=>{if(index>=flat.length-1){clearInterval(playing);playing=null;return;}index++;drawFrame();},140);}),field('Follow live',refs.follow),refs.slider);
 refs.token=n('p');refs.heat=n('canvas');refs.heat.width=1100;refs.heat.height=250;refs.heat.setAttribute('aria-label','Selected-neuron token activation heatmap');refs.heat.onclick=e=>{const r=refs.heat.getBoundingClientRect();index=Math.max(0,Math.min(flat.length-1,Math.floor((e.clientX-r.left)/r.width*flat.length)));refs.follow.checked=false;drawFrame();};
 refs.heat.onmousemove=e=>{if(!flat.length)return;const r=refs.heat.getBoundingClientRect(),x=Math.min(flat.length-1,Math.max(0,Math.floor((e.clientX-r.left)/r.width*flat.length))),y=Math.min(campaign.input.neurons.length-1,Math.max(0,Math.floor((e.clientY-r.top)/r.height*campaign.input.neurons.length))),f=flat[x],p=campaign.input.neurons[y];refs.heat.title='Position '+f.position+' '+JSON.stringify(f.token.text)+' · L'+p.layer+' N'+p.neuron+' = '+f.values[y].toFixed(5);};
 refs.values=n('div',undefined,'pt-values');refs.answer=n('pre');refs.lens=n('div');refs.playback.append(playbackControls,refs.token,refs.heat,refs.values,n('h3','Generated continuation'),refs.answer,refs.lens);root.append(refs.arms,refs.playback);
}
function coordinates(){const raw=refs.neurons.value.trim().split(/[\s,;]+/);return raw.map(v=>{const m=/^L(\d+)N(\d+)$/i.exec(v);if(!m)throw Error('Use coordinates such as L14N8949, separated by spaces.');return {layer:Number(m[1]),neuron:Number(m[2])};});}
async function start(){refs.run.disabled=true;try{const state=await api('/api/state');const value={model_identity:state.identity,prompts:refs.mode.value==='three_prompts'?refs.prompt.value.split(/^\s*---\s*$/m).map(s=>s.trim()):[refs.prompt.value],pushes:JSON.parse(refs.recipe.value),phase:refs.phase.value,neurons:coordinates(),seed:Number(refs.seed.value),max_tokens:Number(refs.budget.value),mode:refs.mode.value};const result=await api('/api/traces',value);refs.follow.checked=true;armIndex=0;await load(result.id);}catch(e){refs.status.textContent=e.message;refs.run.disabled=false;}}
async function load(id){clearTimeout(poll);try{campaign=await api('/api/trace?id='+id);refs.status.textContent=campaign.state+' · '+campaign.id+' · continuous SwiGLU values, not binary firing.';render();const active=['queued','running'].includes(campaign.state);refs.run.disabled=active;if(active)poll=setTimeout(()=>load(id),900);else history();}catch(e){refs.status.textContent=e.message;refs.run.disabled=false;}}
function render(){refs.arms.replaceChildren();campaign.arms.forEach((a,i)=>{const x=b('Prompt '+(a.prompt_index+1)+' · '+a.label+' · '+a.state,()=>{armIndex=i;render();});x.classList.toggle('selected',i===armIndex);if(a.error)x.title=a.error;refs.arms.append(x);});
 const arm=campaign.arms[armIndex];flat=[];for(const e of arm?.events||[])for(let i=0;i<e.positions.length;i++)flat.push({position:e.positions[i],token:e.tokens[i],values:e.values[i],norms:e.residual_l2[i],kind:e.kind});if(refs.follow.checked)index=flat.length-1;index=Math.max(0,Math.min(index,flat.length-1));drawFrame();refs.answer.textContent=arm?.answer||arm?.error||'Waiting for generated continuation…';
 refs.lens.replaceChildren();if(arm?.lens_error)refs.lens.append(n('p','Lens unavailable: '+arm.lens_error));if(arm?.finish_reason==='length')refs.lens.append(n('p','Diagnostic token budget reached. This continuation is incomplete and is not a scored personality answer.','pt-muted'));
 if(arm?.lens){const l=arm.lens;refs.lens.append(n('h3','Empirical next-token lens'),n('p','KL(baseline ∥ intervention) '+l.kl_baseline_to_intervention.toPrecision(5)+' · KL(intervention ∥ baseline) '+l.kl_intervention_to_baseline.toPrecision(5)),n('p',l.semantics,'pt-muted'));
 const cols=n('div',undefined,'pt-deltas');for(const key of ['increases','decreases']){const section=n('section');section.append(n('h4',key==='increases'?'Largest Δlog p increases':'Largest Δlog p decreases'));const table=n('table');for(const row of l[key].slice(0,12)){const tr=n('tr');tr.append(n('td',JSON.stringify(row.text)),n('td',row.delta_logp.toFixed(4)),n('td',(100*Math.exp(row.baseline_logp)).toPrecision(3)+'% → '+(100*Math.exp(row.intervention_logp)).toPrecision(3)+'%'));table.append(tr);}section.append(table);cols.append(section);}refs.lens.append(cols);
 const top=n('details');top.append(n('summary','Most probable next tokens'));
 for(const key of ['baseline_top','intervention_top'])top.append(n('p',key.replace('_top','')+': '+(l[key]||[]).map(t=>JSON.stringify(t.text)+' '+(100*t.probability).toFixed(2)+'%').join(' · ')));
 refs.lens.append(top);
 if(arm.scale===0)refs.lens.append(n('p',arm.baseline_equal?'Zero-dose control: output and captured activations exactly match baseline.':'Zero-dose control differs from baseline; inspect this run before interpreting effects.'));
 }
}
function drawFrame(){refs.slider.max=Math.max(0,flat.length-1);refs.slider.value=index;const f=flat[index],c=refs.heat.getContext('2d'),w=refs.heat.width,h=refs.heat.height; c.fillStyle='#161c20';c.fillRect(0,0,w,h);refs.values.replaceChildren();if(!f){refs.token.textContent='No observed tokens yet.';return;}
 const neurons=campaign.input.neurons;refs.token.textContent='Consumed position '+f.position+' · '+f.kind+' · token '+f.token.id+': '+JSON.stringify(f.token.text);
 let max=0;for(const row of flat)for(const v of row.values)max=Math.max(max,Math.abs(v));max=Math.max(max,1e-10);
 for(let x=0;x<flat.length;x++)for(let y=0;y<neurons.length;y++){const v=flat[x].values[y],a=Math.sqrt(Math.min(1,Math.abs(v)/max));c.fillStyle=v>=0?'rgba(134,194,50,'+a+')':'rgba(222,144,180,'+a+')';c.fillRect(x*w/flat.length,y*h/neurons.length,Math.max(1,w/flat.length),h/neurons.length);}
 c.strokeStyle='#fff';c.lineWidth=2;c.strokeRect(index*w/flat.length,0,Math.max(1,w/flat.length),h);
 neurons.forEach((n,i)=>{refs.values.append(window.document.createTextNode('L'+n.layer+' N'+n.neuron+': '+f.values[i].toFixed(4)+' · residual ‖h‖ '+f.norms[n.layer].toFixed(2)+'  '));});
 refs.heat.title='Columns are consumed tokens; rows follow the listed neuron order. Green = positive; pink = negative. Brightness scales to this arm’s largest absolute observed value. Click to scrub.';
}
async function history(){try{const rows=await api('/api/traces');refs.history.replaceChildren();for(const r of rows)refs.history.append(b(new Date(r.created*1000).toLocaleTimeString()+' · '+r.state,()=>load(r.id)));}catch(e){refs.status.textContent=e.message;}}
function open(){ensure();history();}
ensure();window.PersonaTrace={open};
})();
