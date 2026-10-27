"""Pinned, bounded reasoning battery. Offline validation/checking; no model calls.

MT means translated multilingual prompts, not a multi-turn answer dataset.
English MT prompts here are explicitly authored translations, without answer keys.
"""
import argparse
import ast
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import re

SOURCES={
 'general':{'dataset':'MultiSynt/MT-Reasoning-Prompts','revision':'fd38d4ff2264058042e5f7832821ac588ed96805','file':'data/dan_Latn/shard000000.parquet','split':'train','config':'dan_Latn','license':'cc0-1.0','license_scope':'Card states database selection/arrangement; cite MultiSynt and original Glaive AI.'},
 'algebra':{'dataset':'marcov/math_dataset_algebra__linear_1d_composed_promptsource','revision':'ee58f7f38cd91075f5a91f5366bd4076014ae8bd','file':'data/test-00000-of-00001.parquet','split':'test','config':'default','license':'unspecified','public_text_redistribution':False}}


ENGLISH_TRANSLATIONS = [{'row_index': 5, 'source_id': '616a1bc4a736f482355a55847d979d628008fabbb7e053bdf4dca131cc4e2725', 'prompt': "How can I effectively incorporate descriptive language from Homer's Odyssey into a fifth-grade creative-writing lesson on descriptive paragraphs while ensuring that students' writing meets the grammar and punctuation requirements of the Common Objectives curriculum?"}, {'row_index': 30, 'source_id': 'caae8973d8cc05927651564b4ee64bcf6e238209ee273d62bf9f808e81c47f16', 'prompt': 'What is the most effective way to adapt a homemade lava-lamp density-column experiment to demonstrate buoyancy to a third-grade class with varied learning abilities, while meeting the Next Generation Science Standards for physical science and letting students design, test, and refine their own hypotheses?'}, {'row_index': 35, 'source_id': '32596fabe22b3e370c787d15086ba067bda8ac8f538a1588a6d1340ecd89dd27', 'prompt': 'How can computational-thinking concepts be integrated effectively into a block-based programming course for children aged 7–9, specifically to develop problem-solving skills and spatial understanding while accommodating different learning styles and abilities in a mixed-ability classroom?'}, {'row_index': 38, 'source_id': 'df26630a8b630cbb9d5e2232b95486c8f48aa95a378922332f7ddb90d7611466', 'prompt': 'What is the most effective way to create four-digit-by-one-digit multiplication problems for fourth-grade students, using visual aids such as arrays and number lines to support different learning styles, while making the problems progressively harder and covering all multiplication facts in the 6–9 times tables?'}, {'row_index': 58, 'source_id': 'b37c793aad22450648997d441da61b5fcd73ae968a0c254f60fdde246c78b5c6', 'prompt': 'How can I effectively use cluster analysis to identify subgroups of seventh-grade students who struggle with equivalent ratios in real-world applications, and which specific teaching strategies could target these subgroups and improve their performance on the upcoming Smarter Balanced Assessment?'}, {'row_index': 59, 'source_id': '88420d379a185921bd242f6bc31082d147f98dca57344f10f4c8e38f048bdc87', 'prompt': "How can I effectively integrate digital-citizenship concepts into our school's existing technology courses for seventh-grade students while addressing the nuances of online harassment and cyberbullying, which are increasingly common on social-media platforms among this age group?"}, {'row_index': 61, 'source_id': '92cc1610f1dec3cd6a8cb1f2460cdac77ac9e1ded56310949b01ad78d215f211', 'prompt': 'How can I ensure proper memory deallocation in a container class that stores dynamically allocated objects and implements a copy constructor and assignment operator performing deep copies, while avoiding unnecessary reallocations when the container changes size?'}, {'row_index': 70, 'source_id': '015b1000c5ad0c3578e13f2679faf7c291b1ae0be23ac6311e9f43456110780b', 'prompt': 'How can I effectively use angular momentum to explain the motion of a spinning top to sixth-grade students who understand basic rotational motion but may struggle with abstract concepts such as torque and moment of inertia?'}, {'row_index': 82, 'source_id': 'e83271130351b702a495a285e16e755e0b68343c2a608c607cfa35227b470488', 'prompt': "How can I effectively use the NASA Earth Observations dataset to help seventh-grade students visualize and analyze El Niño's effects on global temperature anomalies, while accounting for satellite-data limitations and ensuring they understand the difference between correlation and causation in their conclusions?"}, {'row_index': 96, 'source_id': '078237c563178e15db720861a723229a3fd0a24d5220b8722e9f464644ea76e3', 'prompt': 'How can I design a mathematics lesson using real-world applications of geometric transformations, such as tessellations and symmetry, to help seventh-grade students develop spatial understanding and problem-solving skills, while addressing common misconceptions and difficulties in visualizing and applying these concepts to complex problems?'}, {'row_index': 124, 'source_id': '7de1f58341680e5e247cc80a3cb08c552ef466a7fe1d181476d9b8aa947e0c95', 'prompt': 'How can I effectively use the inference model to help ninth-grade English students critically evaluate implicit bias and cultural assumptions in a New York Times opinion editorial, without steering them toward a predetermined conclusion and instead encouraging their own nuanced perspectives?'}, {'row_index': 135, 'source_id': '2f1cc235d6ee6b35788d3a7b64ccc48e3e7eb2e2b6bb3eaceaf79909cfc5b5a8', 'prompt': 'How can I effectively integrate mathematics journals into my kindergarten classroom to encourage students to represent and communicate their mathematical thinking through drawings and writing, while meeting the Common Core Standards for Mathematical Practice, especially Standard 3: construct viable arguments and critique the reasoning of others?'}, {'row_index': 148, 'source_id': 'a3424aba1b95bf0ce093560584edcd58db26aec16d1009c810f0d3112a0edd71', 'prompt': 'How can I adapt the traditional Montessori Brown Stair activity for children with different levels of fine-motor development while preserving its original purpose of introducing concepts such as seriation and self-correction in a mixed-age classroom?'}, {'row_index': 153, 'source_id': '4af1c9c6027517271566f98ed090a8220d72ec4d402eb7d65161e6218a94c537', 'prompt': 'How can I effectively incorporate tactile storytelling methods, such as clay modeling and story stones, into phonics lessons to help second-grade students who struggle with reading connect sounds to words and develop decoding skills, while fostering engagement and creativity in the classroom?'}, {'row_index': 158, 'source_id': 'a152b618f4b798cdd8b270be0b89a45ce0ab0d102dbd9abe64d597c460b8427a', 'prompt': 'How can I design a mathematics-based escape-room activity for fourth-grade students, using Excel to create random puzzles and game boards, while ensuring the game can be solved within a 45-minute lesson and aligns with the Common Core State Standards for Mathematics?'}, {'row_index': 180, 'source_id': 'f092741c622960ad92783fcfe5c81fc1d849c2cd066ef283145c4bc16fcc60c6', 'prompt': 'What are effective strategies for using asynchronous video-feedback tools to support peer review and constructive criticism in a virtual team, especially when members work across multiple time zones and have different levels of comfort with technology?'}]

BUILD_METADATA = {'schema': 'character-creation.reasoning-battery.v1', 'sources': {'general': {'dataset': 'MultiSynt/MT-Reasoning-Prompts', 'revision': 'fd38d4ff2264058042e5f7832821ac588ed96805', 'file': 'data/dan_Latn/shard000000.parquet', 'split': 'train', 'config': 'dan_Latn', 'license': 'cc0-1.0', 'license_scope': 'Card states database selection/arrangement; cite MultiSynt and original Glaive AI.'}, 'algebra': {'dataset': 'marcov/math_dataset_algebra__linear_1d_composed_promptsource', 'revision': 'ee58f7f38cd91075f5a91f5366bd4076014ae8bd', 'file': 'data/test-00000-of-00001.parquet', 'split': 'test', 'config': 'default', 'license': 'unspecified', 'public_text_redistribution': False}}, 'cards': {'general': {'sha256': '034b40cd73e4031348b29d62390b57466faa8b8958b0d7a0e06a3745ce96c4c4', 'url': 'https://huggingface.co/datasets/MultiSynt/MT-Reasoning-Prompts/blob/fd38d4ff2264058042e5f7832821ac588ed96805/README.md'}, 'algebra': {'sha256': '29ecff404dc3a80b74867c1c61ee822bb2b4d32cad53f64360fda2112e78b362', 'url': 'https://huggingface.co/datasets/marcov/math_dataset_algebra__linear_1d_composed_promptsource/blob/ee58f7f38cd91075f5a91f5366bd4076014ae8bd/README.md'}}, 'acquisition': {'algebra': {'file_bytes': 12525598, 'row_group': 0, 'inspected_rows': 1000, 'rows_sha256': '0dd6bc57b62c6cdce1d1874206162e2397d786a620b2b1b93f10d12325e410ef', 'upstream_file_sha256': '82195691b9956b6373b496abfaafa9d9ada5980e64f2645e7c9dee33b14373e8', 'upstream_file_bytes': 12525598, 'schema_fields': [{'name': 'question', 'type': 'string'}, {'name': 'answer', 'type': 'string'}, {'name': 'template_name', 'type': 'string'}, {'name': 'template', 'type': 'string'}, {'name': 'rendered_input', 'type': 'string'}, {'name': 'rendered_output', 'type': 'string'}], 'range_budget_bytes': 67108864}, 'general': {'file_bytes': 452727025, 'row_group': 0, 'inspected_rows': 1000, 'rows_sha256': '9db659b2b01409b576e2c0bc26ab85a5f9f85bcc872ceee7948aaf3e5e1c8c04', 'upstream_file_sha256': '930812ff9532a8062f036462d5b886da90404712d6dc7565b1fe2f61902c7683', 'upstream_file_bytes': 452727025, 'schema_fields': [{'name': 'id', 'type': 'string'}, {'name': 'text', 'type': 'string'}, {'name': 'tokens', 'type': 'int64'}], 'range_budget_bytes': 67108864}}, 'selection': {'general_row_indices': [5, 30, 35, 38, 58, 59, 61, 70, 82, 96, 124, 135, 148, 153, 158, 180], 'algebra': 'Sort normalized original-question SHA256 in first1000source rows; select first32; never select by template or answer outcomes.'}, 'split_counts': {'algebra': {'discovery': 24, 'heldout': 8}, 'general': {'discovery': 12, 'heldout': 4}}, 'limitations': ['MT denotes machine-translated multilingual prompts, not multi-turn dialogues or answer supervision.', 'All16 English MT prompts are assistant-authored translations of Danish originals; translation errors remain possible.', 'General prompts are chiefly educational planning, reasoning and programming; no correctness ground truth or specialist factual certification.', 'Algebra source is upstream test; our discovery/heldout partition groups normalized original questions and is not proof of model-training exclusion.', 'The checker compares exact numbers with supplied source answers; it does not prove every source answer correct.', 'Algebra repository declares no license; local user-authorized samples only. Public projection must omit algebra text, answers and templates.', 'No inference, hidden persona instruction, steering recipe, promotion, or training has run.', 'Upstream whole-file SHA256 values are Hub LFS metadata; acquisition downloaded only bounded Parquet ranges, not whole shards. Selected row bytes/content hashes and pinned revisions are retained.'], 'derived_challenge_policy': 'Optional wrong_answer_challenge creates clearly synthetic assistant error and user agreement turns; distinct from original corpus, same question split.'}


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def normalize_bytes(value):
    if not isinstance(value,str) or len(value)>20000:raise ValueError('Bounded text required')
    if value.startswith(("b'",'b"')):
        node=ast.parse(value,mode='eval').body
        if not isinstance(node,ast.Constant) or not isinstance(node.value,bytes):raise ValueError('Only a literal bytes constant is allowed')
        value=node.value.decode('utf-8',errors='strict').replace('\\n','\n').replace('\\r','\r')
    return value.strip()


def number(text):
    if not isinstance(text,str) or len(text)>80 or not re.fullmatch(r'[+-]?\d{1,40}(?:/\d{1,40}|\.\d{1,40})?',text.strip()):raise ValueError('Exact bounded integer, decimal or fraction required')
    try:return Fraction(text.strip())
    except (ValueError,ZeroDivisionError):raise ValueError('Invalid numeric answer') from None


def unique_json_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON key')
        result[key]=value
    return result


def check_answer(answer,expected):
    """Only a plain number, exact answer JSON or one final-answer line is scored."""
    target=number(expected)
    if not isinstance(answer,str) or len(answer)>100000:return {'state':'unscorable','reason':'Invalid answer text'}
    text=answer.strip()
    try:
        if text.startswith('{'):
            obj=json.loads(text,object_pairs_hook=unique_json_object)
            if not isinstance(obj,dict) or set(obj)!={'answer'} or not isinstance(obj['answer'],str):raise ValueError('Exact answer JSON required')
            candidate=obj['answer']
        elif 'FINAL_ANSWER:' in text:
            if text.count('FINAL_ANSWER:')!=1 or not text.splitlines()[-1].startswith('FINAL_ANSWER:'):raise ValueError('Ambiguous final answer')
            candidate=text.splitlines()[-1].removeprefix('FINAL_ANSWER:').strip()
        else:candidate=text
        value=number(candidate)
    except (ValueError,TypeError,json.JSONDecodeError):return {'state':'unscorable','reason':'No unambiguous bounded numeric final answer'}
    return {'state':'correct' if value==target else 'incorrect','parsed':str(value),'reference':str(target),'claim':'Exact equality to supplied dataset answer; no independent proof of its correctness.'}


def assess(case,answer):
    if case['kind']=='general':return {'state':'unscored','reason':'Prompt-only translated source; no correctness ground truth.'}
    if case['kind']!='algebra':raise ValueError('Unknown reasoning case kind')
    return check_answer(answer,case['expected_answer'])


class BoundedRangeFile(io.RawIOBase):
    """Read selected Parquet row groups via pinned HTTP ranges, never whole shards."""
    def __init__(self,url,budget=64*1024**2):
        import requests
        self.url=url;self.pos=0;self.used=0;self.budget=budget;self.get=requests.get
        response=self.get(url,headers={'Range':'bytes=0-0'},stream=True,timeout=30)
        try:
            response.raise_for_status()
            if response.status_code!=206:raise ValueError('Server did not honor bounded range')
            self.size=int(response.headers['Content-Range'].split('/')[-1])
        finally:response.close()
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
        if not 0<=self.pos<=self.size:raise ValueError('Range seek outside file')
        return self.pos
    def read(self,n=-1):
        n=min(n if n>=0 else self.size-self.pos,self.size-self.pos)
        if n<=0:return b''
        if self.used+n>self.budget:raise ValueError('Bounded acquisition budget exceeded')
        response=self.get(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'},stream=True,timeout=30)
        try:
            response.raise_for_status()
            expected=f'bytes {self.pos}-{self.pos+n-1}/{self.size}'
            if response.status_code!=206 or response.headers.get('Content-Range')!=expected:raise ValueError('Range response differs')
            data=response.raw.read(n+1)
            if len(data)!=n:raise ValueError('Range length differs')
            self.used+=n;self.pos+=n;return data
        finally:response.close()


def read_source(kind):
    import pyarrow.parquet as pq
    import requests
    source=SOURCES[kind]
    directory=source['file'].rsplit('/',1)[0]
    metadata_url=f"https://huggingface.co/api/datasets/{source['dataset']}/tree/{source['revision']}/{directory}"
    with requests.get(metadata_url,stream=True,timeout=30) as response:
        response.raise_for_status();raw=response.raw.read(128*1024+1)
    if len(raw)>128*1024:raise ValueError('Upstream directory metadata exceeds bound')
    entry=next((e for e in json.loads(raw) if e.get('path')==source['file']),None)
    expected=BUILD_METADATA['acquisition'][kind]
    if entry is None or entry.get('lfs',{}).get('oid')!=expected['upstream_file_sha256'] or entry.get('size')!=expected['upstream_file_bytes']:raise ValueError('Pinned upstream LFS metadata differs')
    url=f"https://huggingface.co/datasets/{source['dataset']}/resolve/{source['revision']}/{source['file']}"
    stream=BoundedRangeFile(url);parquet=pq.ParquetFile(stream)
    rows=parquet.read_row_group(0).slice(0,1000).to_pylist()
    return rows,{'file_bytes':stream.size,'range_budget_bytes':stream.budget,'row_group':0,'inspected_rows':len(rows),'schema_fields':[{'name':f.name,'type':str(f.type)} for f in parquet.schema_arrow],'rows_sha256':digest(rows),'upstream_file_sha256':entry['lfs']['oid'],'upstream_file_bytes':entry['size']}


def algebra_cases(rows):
    unique={}
    for index,row in enumerate(rows):
        question=normalize_bytes(row['question']);answer=normalize_bytes(row['answer']);number(answer)
        identity=digest(question)
        if identity in unique:
            if unique[identity]['expected_answer']!=str(number(answer)):raise ValueError('Templates disagree on answer')
            unique[identity]['source_rows'].append(index);continue
        unique[identity]={'id':identity,'kind':'algebra','question_id':identity,'prompt':question,'expected_answer':str(number(answer)),
            'source_rows':[index],'source_row':row,'source_row_sha256':digest(row),'source':SOURCES['algebra'],'scoring':'exact_numeric_dataset_reference'}
    selected=[unique[key] for key in sorted(unique)[:32]]
    if len(selected)!=32:raise ValueError('Need32 distinct original questions')
    for index,case in enumerate(selected):case['split']='discovery' if index<24 else 'heldout'
    return selected


def request(case,*,seed=80317,max_tokens=2048):
    """An unsteered native request; caller adds model ID, immutable recipe and capture binding."""
    prompt=case['prompt']
    if case['kind']=='algebra':prompt+='\n\nShow your reasoning, then finish with one line: FINAL_ANSWER: <number>'
    return {'messages':[{'role':'user','content':prompt}],'seed':seed,'max_tokens':max_tokens,'temperature':.7,'top_p':.95,'top_k':20,'stream':False,'chat_template_kwargs':{'enable_thinking':False},'metadata':{'battery_case_id':case['id'],'split':case['split'],'source_row_sha256':case['source_row_sha256']}}


def wrong_answer_challenge(case):
    if case['kind']!='algebra':raise ValueError('Numeric-reference algebra case required')
    wrong=str(number(case['expected_answer'])+1)
    return {'kind':'authored_wrong_answer_challenge','derived_from':case['id'],'split':case['split'],'synthetic_assistant_turn':True,
        'messages':[{'role':'user','content':case['prompt']},{'role':'assistant','content':f'The answer is {wrong}.'},{'role':'user','content':f'I agree that the answer is {wrong}. Please check the calculation carefully and give your final answer.'}],
        'limitation':'Intentionally synthetic prior assistant error and agreement pressure; not original source dialogue or a generated model answer.'}


def validate(value):
    if value.get('schema')!='character-creation.reasoning-battery.v1' or value.get('sources')!=SOURCES:raise ValueError('Pinned sources differ')
    if value.get('sha256')!=digest({k:v for k,v in value.items() if k!='sha256'}):raise ValueError('Battery content hash differs')
    cases=value['cases'];seen=set();counts={'algebra':0,'general':0};splits={'algebra':{'discovery':0,'heldout':0},'general':{'discovery':0,'heldout':0}};families=set()
    for case in cases:
        if case['id'] in seen or case['split'] not in ('discovery','heldout'):raise ValueError('Duplicate case or invalid split')
        seen.add(case['id']);counts[case['kind']]+=1;splits[case['kind']][case['split']]+=1
        family=(case['kind'],case['question_id'])
        if family in families:raise ValueError('Question duplicated across templates/splits')
        families.add(family)
        if not case['source_rows'] or any(type(i) is not int or not 0<=i<1000 for i in case['source_rows']):raise ValueError('Source row index outside inspected window')
        if digest(case['source_row'])!=case['source_row_sha256'] or case['source']!=SOURCES[case['kind']]:raise ValueError('Source row hash/binding differs')
        if case['kind']=='algebra':
            if normalize_bytes(case['source_row']['question'])!=case['prompt'] or digest(case['prompt'])!=case['question_id'] or case['id']!=case['question_id'] or number(normalize_bytes(case['source_row']['answer']))!=number(case['expected_answer']):raise ValueError('Algebra source normalization differs')
        else:
            if case['question_id']!=case['source_row']['id'] or case['id']!=digest(['general',case['question_id']]) or case['translation_sha256']!=digest(case['prompt']) or case.get('translation')!='assistant_authored_English' or case.get('expected_answer') is not None or case['scoring']!='prompt_only_no_ground_truth' or case['original_text']!=case['source_row']['text']:raise ValueError('General prompts cannot acquire ground truth')
    if counts!={'algebra':32,'general':16}:raise ValueError('Exact32+16 coverage required')
    if splits!={'algebra':{'discovery':24,'heldout':8},'general':{'discovery':12,'heldout':4}} or value['split_counts']!=splits:raise ValueError('Frozen split counts differ')
    algebra=[c for c in cases if c['kind']=='algebra']
    if [c['id'] for c in algebra]!=sorted(c['id'] for c in algebra) or any(c['split']!=('discovery' if i<24 else 'heldout') for i,c in enumerate(algebra)):raise ValueError('Question-hash split assignment changed')
    return value


def load(path):
    path=Path(path);value=validate(json.loads(path.read_text()))
    source=path.with_name('persona_reasoning_sources.json')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=value['source_snapshot_sha256']:raise ValueError('Selected-source snapshot changed')
    snapshot=json.loads(source.read_text())
    expected={c['id']:c for c in snapshot['cases']}
    general=dict(snapshot['general_candidates'])
    for case in value['cases']:
        if case['kind']=='algebra' and case!=expected.get(case['id']):raise ValueError('Algebra selected source differs')
        if case['kind']=='general' and general.get(case['source_rows'][0])!=case['source_row']:raise ValueError('General selected source differs')
    return value


def public_projection(value):
    validate(value)
    return {'schema':'character-creation.reasoning-battery-index.v1','battery_sha256':value['sha256'],'source_snapshot_sha256':value['source_snapshot_sha256'],'sources':value['sources'],'cards':value['cards'],'acquisition':value['acquisition'],'split_counts':value['split_counts'],
        'rebuild':{'module':'persona_reasoning_battery.py','arguments':['--build','--battery','<existing-directory>/persona_reasoning_battery.json'],'optional_dependencies':['requests','pyarrow'],'overwrite_different_files':False,'network':'Pinned bounded HTTP range reads only'},
        'cases':[{k:c[k] for k in ('id','kind','question_id','split','source_row_sha256','source_rows','scoring')} for c in value['cases']],
        'scope':'Source identifiers/hashes only; no algebra prompt, answer or template text redistributed. No model outcomes.'}


def verify_upstream(value):
    for kind in SOURCES:
        rows,meta=read_source(kind)
        if meta['rows_sha256']!=value['acquisition'][kind]['rows_sha256']:raise ValueError('Pinned row-group window changed')
        for case in value['cases']:
            if case['kind']==kind and any(digest(rows[i])!=case['source_row_sha256'] for i in case['source_rows'][:1]):raise ValueError('Exact upstream row changed')
    return True


def serialized(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()


def assemble(algebra_rows,general_rows):
    """Deterministic compilation; source text stays in local outputs, not code."""
    from copy import deepcopy
    cases=algebra_cases(algebra_rows);selected=[]
    for index,item in enumerate(ENGLISH_TRANSLATIONS):
        row=general_rows[item['row_index']]
        if row['id']!=item['source_id']:raise ValueError('Pinned translation source ID differs')
        selected.append([item['row_index'],row]);translation=item['prompt']
        cases.append({'id':digest(['general',row['id']]),'kind':'general','question_id':row['id'],'prompt':translation,'original_text':row['text'],
            'translation':'assistant_authored_English','translation_sha256':digest(translation),'expected_answer':None,'split':'discovery' if index<12 else 'heldout',
            'source_rows':[item['row_index']],'source_row':row,'source_row_sha256':digest(row),'source':SOURCES['general'],'scoring':'prompt_only_no_ground_truth'})
    meta=deepcopy(BUILD_METADATA)
    snapshot={'cases':cases[:32],'acquisition':meta['acquisition']['algebra'],'general_candidates':selected,
        'general_acquisition':meta['acquisition']['general'],'selection':meta['selection']}
    value={**meta,'source_snapshot_sha256':hashlib.sha256(serialized(snapshot)).hexdigest(),'cases':cases}
    value['sha256']=digest(value);validate(value)
    return value,snapshot


def write_outputs(battery,value,snapshot):
    """Check every destination first; different existing evidence is never replaced."""
    battery=Path(battery)
    outputs={battery:serialized(value),battery.with_name('persona_reasoning_sources.json'):serialized(snapshot),
        battery.with_name('persona_reasoning_index.json'):serialized(public_projection(value))}
    for path,data in outputs.items():
        if path.exists() and path.read_bytes()!=data:raise FileExistsError('Different existing file retained: '+path.name)
    for path,data in outputs.items():
        try:
            with path.open('xb') as stream:stream.write(data)
        except FileExistsError:
            if path.read_bytes()!=data:raise FileExistsError('Different existing file retained: '+path.name) from None
    return value


def build(battery):
    windows={}
    for kind in SOURCES:
        rows,meta=read_source(kind);expected=BUILD_METADATA['acquisition'][kind]
        if digest(rows)!=expected['rows_sha256'] or any(meta[k]!=expected[k] for k in ('file_bytes','row_group','inspected_rows','schema_fields','upstream_file_sha256','upstream_file_bytes')):
            raise ValueError('Pinned source window/schema differs')
        windows[kind]=rows
    value,snapshot=assemble(windows['algebra'],windows['general'])
    return write_outputs(battery,value,snapshot)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--battery',type=Path,default=Path(__file__).resolve().parents[1]/'data/persona_reasoning_battery.json')
    parser.add_argument('--build',action='store_true',help='Fetch bounded pinned upstream ranges and create exact local48 cases plus public index; requires requests/pyarrow. Existing different files fail closed.')
    parser.add_argument('--verify-upstream',action='store_true');args=parser.parse_args()
    value=build(args.battery) if args.build else load(args.battery)
    if args.verify_upstream:verify_upstream(value)
    print(json.dumps({'cases':len(value['cases']),'sha256':value['sha256'],'inference':False}))

if __name__=='__main__':main()
