"""CPU-only browser-control contract checks; no requests or inference."""
import json
from pathlib import Path
import subprocess
import unittest

class LiteraryKVUITests(unittest.TestCase):
    def test_kind_copy_selection_and_qualification_gate(self):
        source=(Path(__file__).resolve().parents[1]/'ui/persona_literature.js').read_text()
        start=source.index(' function kvPresentation(')
        end=source.index(' function promptHarness(',start)
        script=r'''
const assert=require('node:assert/strict');
class Element {
 constructor(tag,text){this.tag=tag;this.textContent=text;this.children=[];this.value='';this.classList={toggle(){}};}
 append(...rows){this.children.push(...rows);}
 replaceChildren(...rows){this.children=rows;this.value='';}
 add(row){this.children.push(row);}
 closest(){return null;}
}
const n=(tag,text)=>new Element(tag,text),button=(text,fn)=>Object.assign(n('button',text),{onclick:fn}),field=(label,e)=>{const x=n('label',label);x.append(e);return x;};
const Option=function(text,value){return Object.assign(n('option',text),{value});};
let available=true,requests=0;
async function experimentRequest(){requests++;return {recipes:[{id:'recipe',name:'Recipe',pushes:[{kind:'neuron'}]}],kv:{available,reason:'Qualification checked',artifacts:[{label:'Learned',artifact:{kind:'learned_attention_kv',sha256:'learned'}},{label:'Text',artifact:{kind:'hybrid_text_prefix',sha256:'text'}}]},history:[],plan:{}};}
'''+source[start:end]+r'''
(async()=>{
 const root=literaryExperiment({id:'prompt',provenance:{split:'heldout'}});
 const all=[];function walk(x){all.push(x);for(const c of x.children)walk(c);}walk(root);
 const selects=all.filter(x=>x.tag==='select'),[recipe,artifact]=selects;
 const run=all.find(x=>x.textContent==='Run four matched arms');
 const refresh=all.find(x=>x.textContent==='Refresh availability');
 assert.equal(requests,0);assert.equal(run.disabled,true);
 await refresh.onclick();assert.equal(requests,1);assert.equal(run.disabled,true);
 assert.equal(artifact.value,'');assert.equal(recipe.value,'');
 recipe.value='recipe';artifact.value='learned';artifact.onchange();
 assert.equal(run.disabled,false);assert.match(root.children[0].textContent,/learned attention-kv/);
 assert.match(root.children[1].textContent,/recurrent state starts fresh/);
 assert.match(root.children[1].textContent,/not continuous dose/);
 artifact.value='text';artifact.onchange();
 assert.match(root.children[0].textContent,/cached text prefix/);
 assert.match(root.children[1].textContent,/not a learned graft/);
 artifact.value='unknown';artifact.onchange();assert.equal(run.disabled,true);
 available=false;await refresh.onclick();recipe.value='recipe';artifact.value='learned';artifact.onchange();assert.equal(run.disabled,true);
 assert.equal(requests,2); // Availability reads only; selecting controls never generates.
})().catch(e=>{console.error(e);process.exit(1)});
'''
        result=subprocess.run(['node','-e',script],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
