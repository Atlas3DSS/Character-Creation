/* Independent checkpoint lens and weight-affinity explorer, inspired by Lyra. */
(()=>{
'use strict';
let target=null,serial=0,dialog,body,title,compact,returnFocus;
const controls={hops:2,fanout:4,direction:'both',threshold:0,bundle:.2,layout:'topology'};
const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
const button=(text,fn)=>{const b=node('button',text);b.type='button';b.onclick=fn;return b;};
const fetchJSON=async path=>{const r=await fetch(path),v=await r.json();if(!r.ok)throw Error(v.error||'Request failed');return v;};
function ensure(){
 if(dialog)return;
 dialog=node('dialog',undefined,'pa-dialog');dialog.setAttribute('aria-labelledby','pa-title');
 const head=node('header');title=node('h2','Checkpoint atlas');title.id='pa-title';
 head.append(title,button('Close',()=>dialog.close()));dialog.append(head,node('p','Static weight evidence · independently implemented from Lyra’s neuron-explorer workflow.','pa-muted'));
 const filters=node('div',undefined,'pa-controls');
 for(const [key,label,values] of [ ['hops','Hops',[1,2,3]],['fanout','Edges per direction',[2,4,6,8]],['direction','Direction',['both','in','out']],['threshold','Minimum |cosine|',[0,.01,.02,.05,.1]],['bundle','Curve bundling',[0,.2,.6]],['layout','3D geometry',['topology','layers']] ]){
  const labelNode=node('label',label),s=node('select');for(const v of values){const o=node('option',String(v));o.value=v;s.append(o);}s.value=controls[key];s.onchange=()=>{controls[key]=['direction','layout'].includes(key)?s.value:Number(s.value);load();};labelNode.append(s);filters.append(labelNode);
 }
 const coordinate=node('form',undefined,'pa-controls'),layer=node('input'),neuron=node('input');layer.type=neuron.type='number';layer.min=neuron.min=0;layer.max=63;neuron.max=17407;layer.value=0;neuron.value=0;layer.id='pa-layer';neuron.id='pa-neuron';layer.setAttribute('aria-label','Layer');neuron.setAttribute('aria-label','Neuron');
 coordinate.append(node('span','Open coordinate'),layer,neuron,button('Open',()=>{target={layer:Number(layer.value),neuron:Number(neuron.value)};load();}));coordinate.onsubmit=e=>e.preventDefault();
 const search=node('form',undefined,'pa-search'),input=node('input'),results=node('div',undefined,'pa-search-results');input.placeholder='Search stored top tokens…';input.maxLength=100;input.setAttribute('aria-label','Search token text');const searchButton=button('Find neurons',()=>search.requestSubmit());
 search.append(input,searchButton);search.onsubmit=async e=>{e.preventDefault();searchButton.disabled=true;results.replaceChildren(node('p','Searching checkpoint token projections…'));try{const v=await fetchJSON('/api/lens/search?q='+encodeURIComponent(input.value));results.replaceChildren(node('p',v.scope+' '+v.missing_layers.length+' missing layers.','pa-muted'));for(const m of v.matches)results.append(button(`L${m.layer} N${m.neuron} ${m.direction==='positive'?'+':'−'} · ${JSON.stringify(m.token)} · ${m.logit.toFixed(2)}`,()=>{target=m;load();}));}catch(e){results.replaceChildren(node('p',e.message));}finally{searchButton.disabled=false;}};
 body=node('div');dialog.append(coordinate,filters,search,results,body);document.body.append(dialog);dialog.addEventListener('close',()=>returnFocus?.focus({preventScroll:true}));
}
function tokenTable(data){
 const columns=node('div',undefined,'pa-lenses');for(const sign of ['positive','negative']){
  const column=node('section');column.append(node('h3',(sign==='positive'?'+':'−')+' output direction'));
  const table=node('table');for(const row of data[sign]||[]){const tr=node('tr'),text=node('td',JSON.stringify(row.text)),score=node('td',row.logit.toFixed(2));text.title='Token ID '+row.id;tr.append(text,score);table.append(tr);}column.append(table);columns.append(column);
 }return columns;
}
async function load(){
 if(!target)return;ensure();const id=++serial,t={layer:target.layer,neuron:target.neuron};title.textContent=`L${t.layer} · N${t.neuron} — checkpoint atlas`;
 document.getElementById('pa-layer').value=t.layer;document.getElementById('pa-neuron').value=t.neuron;
 body.replaceChildren(node('p','Loading checkpoint evidence…'));
 const q=new URLSearchParams(t),webq=new URLSearchParams({...t,...controls});
 try{
  const [lens,web]=await Promise.all([fetchJSON('/api/lens?'+q),fetchJSON('/api/weight-web?'+webq)]);if(id!==serial)return;
  body.replaceChildren();if(!lens.available){body.append(node('p',lens.message));return;}
  const add=button('Add this neuron to map recipe',()=>{if(typeof pushes==='undefined')return;const dose=Number(document.getElementById('dose').value);if(!Number.isFinite(dose))return;pushes.push({kind:'neuron',layer:t.layer,neuron:t.neuron,dose});renderPushes();add.textContent='Added · L'+t.layer+' N'+t.neuron;});body.append(add);
  body.append(node('p',lens.source.semantics.lens,'pa-muted'),tokenTable(lens));
  body.append(node('h3','Weight-path web'));
  if(web.available){body.append(node('p',web.semantics.web,'pa-muted'));body.append(node('p',controls.layout==='topology'?web.topology_semantics:web.semantics.layout,'pa-muted'));
   body.append(window.PersonaGraph.render(web.graph,{layout:controls.layout,normalizeEdges:true,threshold:0,bundle:controls.bundle,onSelect:n=>{target=n;load();}}));
   body.append(node('p',`${web.graph.nodes.length} nodes · ${web.graph.edges.length} retained affinities · 3D coordinates = selected geometry · arrows = source down → destination gate/up · dashed pink = negative · widths relative to strongest displayed affinity. ${web.semantics.coverage}`,'pa-muted'));
   const pooled=node('details');pooled.append(node('summary','Tokens recurring across this weight cone'));
   for(const sign of ['positive','negative'])pooled.append(node('p',sign+': '+(web.pooled_tokens?.[sign]||[]).map(t=>JSON.stringify(t.text)+' ('+t.neurons+' neurons)').join(' · ')));
   pooled.append(node('p','Counts of stored top-token projections among displayed neurons, unweighted by path or direction. These are not predicted generation effects.','pa-muted'));body.append(pooled);
   if(web.missing_layers.length)body.append(node('p','Missing layer data: '+web.missing_layers.join(', ')));
  }else body.append(node('p',web.message));
  const details=node('details');details.append(node('summary','Checkpoint provenance and limitations'),node('pre',JSON.stringify(lens.source,null,2)));body.append(details);
 }catch(e){if(id===serial)body.replaceChildren(node('p',e.message));}
}
function open(){ensure();returnFocus=document.activeElement;if(!dialog.open)dialog.showModal();load();}
function select(n){target={layer:n.layer,neuron:n.neuron};ensure();compact=node('section',undefined,'pa-compact');compact.append(node('h3','Logit lens & weight paths'),node('p','Explore both signs, upstream/downstream affinities and token search. Static weights are separate from measured co-activity.','pa-muted'),button('Open checkpoint atlas ↗',open));document.getElementById('unit')?.append(compact);if(dialog.open)load();}
window.PersonaAtlas={select,open};
})();
