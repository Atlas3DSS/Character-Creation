"""Activation workflow DOM contract; all server calls mocked, no inference."""
from pathlib import Path
import subprocess
import unittest

class ActivationWorkflowUITests(unittest.TestCase):
 def test_drafts_candidates_comparisons_and_reset_races(self):
  path=Path(__file__).resolve().parents[1]/'ui/persona_activation_search.js'
  source=path.read_text().replace('window.PersonaActivationSearch={mount,open,preloadText};','window.PersonaActivationSearch={mount,open,preloadText,_test:{refs:()=>refs,start,load,startComparison,reset,labels,knownLabels,selection:()=>selected}};')
  script=r'''
const assert=require('node:assert/strict');
class Element{
 constructor(tag,text){this.tag=tag;this.textContent=text||'';this.children=[];this.value='';this.classList={add(){},contains(){return true},toggle(){}};}
 append(...v){this.children.push(...v);}replaceChildren(...v){this.children=v;}setAttribute(){}focus(){}scrollIntoView(){}getContext(){return null;}
}
const root=new Element('section');global.window={};global.document={hidden:false,body:root,createElement:t=>new Element(t),createTextNode:t=>new Element('text',t),getElementById:()=>root,querySelector:()=>null};
const calls=[];let holdSearch=null,delaySearch=false,labelModel='model';
const job={starter_recipes:[{id:'single',label:'First candidate trial',neurons:[{layer:3,neuron:0}],dose:.1,reason:'Largest current-input peak',status:'untested',evidence:'current_input_activity'},{id:'bad',label:'Invalid candidate trial',neurons:[{layer:99,neuron:0}],dose:.1,status:'untested',evidence:'current_input_activity'}],id:'search',state:'complete',input:{text:'fear of the green dog',top_k:32,model_identity:'model'},output:{model_identity:'model',tokens:[{id:1,text:'green',position:4}],candidates:Array.from({length:7},(_,i)=>({layer:3,neuron:i,peak:i+1,peak_position:4,mean_abs:1,values:[i+1]})),coverage:{tokens:1}},comparisons:[]};
global.fetch=async(url,options)=>{
 calls.push({url,method:options?.method||'GET',body:options?.body&&JSON.parse(options.body)});
 let result;
 if(url==='/api/activation-searches'&&options?.method==='POST'){
  if(delaySearch)return new Promise(resolve=>{holdSearch=()=>resolve({ok:true,json:async()=>({id:'search'})});});result={id:'search'};
 }else if(url.startsWith('/api/activation-search?'))result=job;
 else if(url==='/api/activation-searches')result=[job];
 else if(url==='/api/neuron-labels')result={model_identity:labelModel,labels:[{layer:3,neuron:0,label:'green',direction:'positive'}]};
 else if(url==='/api/activation-comparisons')result={id:'batch'};
 else if(url.startsWith('/api/batch?'))result={id:'batch',state:'complete',cells:[{state:'complete',label:'baseline',output:{answer:'scene',complete:true}}]};
 else throw Error('Unexpected mocked request '+url);
 return {ok:true,json:async()=>result};
};
'''+source+r'''
(async()=>{
 const app=window.PersonaActivationSearch;app.mount();const t=app._test,r=t.refs();
 assert.equal(calls.length,0);assert.equal(r.text.maxLength,20000);assert.equal(r.compare.disabled,true);
 app.preloadText('arbitrary thought');assert.equal(calls.length,0);assert.equal(r.text.value,'arbitrary thought');
 r.text.value='whole green phrase';r.text.selectionStart=6;r.text.selectionEnd=11;
 function nodes(e){return [e,...e.children.flatMap(nodes)];}
 nodes(root).find(x=>x.textContent==='Use highlighted text only').onclick();assert.equal(r.text.value,'green');assert.equal(calls.length,0);
 await t.start();assert.equal(calls.find(x=>x.method==='POST').body.text,'green');assert.equal(r.prompt.value,job.input.text);
 await t.labels();assert.equal(t.knownLabels(job.output.candidates[0]).length,1);
 labelModel='other';await t.labels();assert.equal(t.knownLabels(job.output.candidates[0]).length,0);
 const beforeStarter=calls.length;const starter=nodes(r.starters).find(x=>x.textContent==='First candidate trial');assert.ok(starter);assert.equal(nodes(r.starters).some(x=>x.textContent==='Invalid candidate trial'),false);starter.onclick();assert.equal(t.selection().size,1);assert.equal(Number(r.dose.value),.1);assert.equal(calls.length,beforeStarter);assert.match(nodes(r.starters).map(x=>x.textContent).join(' '),/not proven safe/);const chosen=nodes(r.table).find(x=>x.type==='checkbox'&&x.checked);chosen.checked=false;chosen.onchange();r.dose.value=.5;
 const boxes=()=>nodes(r.table).filter(x=>x.type==='checkbox');
 for(let i=0;i<6;i++){const check=boxes().find(x=>!x.checked&&!x.disabled);check.checked=true;check.onchange();}
 assert.equal(t.selection().size,6);assert.equal(boxes().filter(x=>x.disabled).length,1);assert.equal(r.compare.disabled,false);
 nodes(root).find(x=>x.textContent==='Draft a scene around this input').onclick();assert.match(r.prompt.value,/Act out a short fictional scene/);assert.match(r.prompt.value,/fear of the green dog/);
 r.prompt.value='Shared scene exact';await t.startComparison();
 const comparison=calls.find(x=>x.url==='/api/activation-comparisons');assert.equal(comparison.body.prompt,'Shared scene exact');assert.equal(comparison.body.neurons.length,6);assert.equal(comparison.body.dose,.5);
 r.text.value='edited idea';r.text.oninput();assert.match(r.draftStatus.textContent,/saved source/);
 app.preloadText('new idea');assert.equal(t.selection().size,0);assert.equal(r.answers.children.length,0);assert.equal(r.compare.disabled,true);
 delaySearch=true;const inflight=t.start();t.reset();holdSearch();await inflight;
 assert.equal(r.text.value,'');assert.equal(r.prompt.value,'');assert.equal(r.results.hidden,true);assert.equal(t.selection().size,0);assert.equal(r.compare.disabled,true);
 delete job.starter_recipes;await t.load('search',true);assert.equal(r.starters.children.length,0);assert.equal(r.compare.disabled,true);
 assert.equal(calls.filter(x=>x.method==='POST').length,3); // two deliberate searches, one deliberate comparison
})().catch(e=>{console.error(e);process.exit(1)});
'''
  result=subprocess.run(['node','-e',script],capture_output=True,text=True)
  self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
