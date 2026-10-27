/* Shared metadata graph and full-text browser. Never launches inference. */
(() => {
 'use strict';
 const names={personality_diagnostics:'Personality diagnostics · shared input',big_five:'Big Five',adult_sexuality:'Adult sexuality · 18+',scotus:'SCOTUS',skippy:'Skippy books'};
 const human=s=>String(s).replaceAll('_',' '), n=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e;};
 const button=(text,fn)=>{const e=n('button',text);e.type='button';e.onclick=fn;return e;};
 function mount(root,{onSelect=()=>{},corpus='big_five'}={}){
  root.classList.add('prompt-browser');
  const state={corpus,split:'discovery',q:'',group:'trait',trait:'',facet:'',domain:'',category:'',origin:'',offset:0,limit:24,spread:false}, refs={}, filters=['trait','facet','domain','category','origin'];let serial=0,timer;
  const controls=n('div');controls.className='pb-controls';
  function select(key,label,options){const wrap=n('label',label),el=n('select');el.setAttribute('aria-label',label);for(const [value,text] of options){const o=n('option',text);o.value=value;el.append(o);}el.value=String(state[key]);el.onchange=()=>{state[key]=key==='limit'?Number(el.value):el.value;state.offset=0;if(key==='corpus'){filters.forEach(f=>state[f]='');state.group='trait';refs.group.value='trait';}load();};wrap.append(el);refs[key]=el;controls.append(wrap);}
  select('corpus','Corpus',Object.entries(names));select('split','Split',[['discovery','Discovery'],['heldout','Held-out'],['all','All splits']]);
  select('group','Group nodes by',filters.map(f=>[f,human(f)]));select('limit','Prompts per page',[24,48,96].map(v=>[v,String(v)]));
  const search=n('input');search.type='search';search.placeholder='Find a concept, scenario, facet or domain…';search.setAttribute('aria-label','Search prompts and category labels');search.oninput=()=>{clearTimeout(timer);timer=setTimeout(()=>{state.q=search.value;state.offset=0;load();},250);};
  const crumbs=n('div'),status=n('p'),graph=n('div'),groups=n('details'),groupList=n('div'),cards=n('div'),pager=n('div');status.setAttribute('role','status');graph.className='pb-graph';cards.className='pb-cards';crumbs.className='pb-crumbs';pager.className='pb-pager';groups.append(n('summary','All category nodes'),groupList);
  const notice=n('p','Node links show the recorded category hierarchy. Design labels are not measured personality traits.');notice.className='pb-note';
  const adult=n('p','Adult-only, non-graphic research scenarios about sexuality, desire, intimacy and consent. Authored prompts; not a validated assessment.');adult.className='pb-note';
  const previous=button('← Previous',()=>{state.offset=Math.max(0,state.offset-state.limit);state.spread=false;load();}),next=button('Next →',()=>{state.offset+=state.limit;state.spread=false;load();});
  const page=n('input');page.type='number';page.min=1;page.value=1;page.setAttribute('aria-label','Jump to page');
  const jump=button('Go to page',()=>{const value=Number(page.value);if(Number.isInteger(value)&&value>=1&&value<=Number(page.max)){state.offset=(value-1)*state.limit;state.spread=false;load();}});
  const sample=button('Sample across selection',()=>{state.spread=true;state.offset=0;load();});
  pager.append(previous,page,jump,next,sample);root.append(controls,search,crumbs,notice,adult,status,graph,groups,pager,cards);
  const sv=(tag,attrs,text)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs||{}).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;return e;};
  function choose(key){state[state.group]=key;const order=['trait','facet','domain'];const index=order.indexOf(state.group);if(index>=0&&index<2){state.group=order[index+1];refs.group.value=state.group;}state.offset=0;state.spread=false;load();}
  function renderGraph(data){graph.replaceChildren();groupList.replaceChildren();const entries=data.groups.slice(0,16),height=Math.max(320,entries.length*55),svg=sv('svg',{viewBox:`0 0 760 ${height}`,role:'group','aria-label':`${human(state.group)} category nodes`});
   svg.append(sv('rect',{x:12,y:height/2-40,width:195,height:80,rx:18,class:'pb-hub'}));svg.append(sv('text',{x:110,y:height/2-5,'text-anchor':'middle'},names[state.corpus]||'Prompts'));svg.append(sv('text',{x:110,y:height/2+20,'text-anchor':'middle'},`${data.total.toLocaleString()} prompts`));
   entries.forEach((g,i)=>{const y=(i+.5)*height/entries.length;svg.append(sv('path',{d:`M207 ${height/2} C300 ${height/2} 300 ${y} 370 ${y}`,class:'pb-edge'}));const item=sv('g',{role:'button',tabindex:0,'aria-label':`${human(g.key)}: ${g.count} prompts`});item.append(sv('rect',{x:370,y:y-22,width:374,height:44,rx:12,class:'pb-node'}));item.append(sv('title',{},`${human(g.key)} · ${g.count.toLocaleString()} prompts`));item.append(sv('text',{x:386,y:y+5},human(g.key).length>37?human(g.key).slice(0,35)+'…':human(g.key)));item.append(sv('text',{x:726,y:y+5,'text-anchor':'end'},g.count.toLocaleString()));item.onclick=()=>choose(g.key);item.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(g.key);}};svg.append(item);});graph.append(svg);
   if(data.groups.length>16)graph.append(n('p',`Showing the 16 largest nodes; all ${data.groups.length} are available below.`));
   data.groups.forEach(g=>groupList.append(button(`${human(g.key)} · ${g.count.toLocaleString()}`,()=>choose(g.key))));
  }
  async function load(){const token=++serial;status.textContent='Loading prompt categories…';previous.disabled=next.disabled=true;try{const response=await fetch('/api/prompt-browser?'+new URLSearchParams(state));const data=await response.json();if(!response.ok)throw Error(data.error||'Unable to load prompts');if(token!==serial)return;
   adult.hidden=state.corpus!=='adult_sexuality';crumbs.replaceChildren(button('Clear category filters',()=>{filters.forEach(f=>state[f]='');state.offset=0;state.group='trait';refs.group.value='trait';load();}));filters.filter(f=>state[f]).forEach(f=>crumbs.append(button(`${human(f)}: ${human(state[f])} ×`,()=>{state[f]='';state.offset=0;load();})));
   state.offset=data.offset;const pages=Math.max(1,Math.ceil(data.total/state.limit));page.max=pages;page.value=Math.floor(data.offset/state.limit)+1;
   status.textContent=data.spread?`${data.rows.length} evenly spaced examples from ${data.total.toLocaleString()} matching prompts`:`${data.total?data.offset+1:0}–${data.offset+data.rows.length} of ${data.total.toLocaleString()} · ${pages.toLocaleString()} pages`;
   previous.disabled=data.offset===0;next.disabled=data.offset+data.rows.length>=data.total||data.spread;renderGraph(data);cards.replaceChildren();
   for(const prompt of data.rows){const card=n('article'),head=n('div');head.className='pb-card-head';const add=button('Add to selection',()=>{try{onSelect(prompt);add.textContent='Added ✓';}catch(e){status.textContent=e.message;}});head.append(n('strong',prompt.label),add);card.append(head,n('small',`${names[prompt.corpus]||prompt.corpus} · ${prompt.split} · ${human(prompt.facets.facet)} · ${human(prompt.facets.domain)}`),n('p',prompt.prompt));if(prompt.system){const detail=n('details');detail.append(n('summary','Profile system instruction'),n('p',prompt.system));card.append(detail);}cards.append(card);}
   if(!data.total)cards.append(n('p','No matches. Clear a category filter or change the search.'));
  }catch(e){if(token===serial)status.textContent=e.message;}}
  load();return {refresh:load};
 }
 window.PersonaPromptBrowser={mount};
})();
