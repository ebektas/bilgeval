"""TD's private reader. The canonical schema union is intentionally unchanged."""
from dataclasses import dataclass
import json
from .reader import open_dataset_text

@dataclass(frozen=True)
class TDExample:
    id: str
    state: object
    questions: dict
    target: dict
    metadata: dict
    @property
    def raw(self):
        return {'id':self.id,'metadata':self.metadata}

def labels_for(q):
    if q['type']=='noul':return ['false','true']
    if q['type']=='choice':return list(q['criteria'])
    return [str(i) for i in range(len(q['criteria']))]

def normalize_td(row):
    if row.get('schema_version')!='td.instance/v1' or row.get('task_type')!='typed_decision':raise ValueError('Expected td.instance/v1 typed_decision')
    if not isinstance(row.get('id'),str) or not row['id']:raise ValueError('Missing TD id')
    inp=row['input'];qs=inp['questions'];gold=row['target']
    if not isinstance(inp['state'],(str,dict,list)):raise ValueError('Invalid state')
    if not isinstance(qs,dict) or not qs or set(qs)!=set(gold):raise ValueError('Question/gold key mismatch')
    for name,q in qs.items():
        if not isinstance(name,str) or not name or q.get('type') not in ('noul','choice','score'):raise ValueError('Invalid TD question')
        if not isinstance(q.get('instructions'),str):raise ValueError('Instructions must be text')
        c=q.get('criteria')
        if q['type']=='choice' and (not isinstance(c,dict) or not 1<=len(c)<=255 or not all(isinstance(k,str) and isinstance(v,str) for k,v in c.items())):raise ValueError('Invalid choice criteria')
        if q['type']=='score' and (not isinstance(c,list) or not 1<=len(c)<=10 or not all(isinstance(v,str) for v in c)):raise ValueError('Invalid score criteria')
        if q['type']=='noul':
            if type(gold[name]) is not bool:raise ValueError('Noul gold must be boolean')
            if c is not None and (not isinstance(c,dict) or set(c)!={'false','true'}):raise ValueError('Invalid noul criteria')
        elif q['type']=='score':
            if type(gold[name]) is not int or not 0<=gold[name]<len(c):raise ValueError('Invalid score gold')
        elif gold[name] not in c:raise ValueError('Invalid choice gold')
    return TDExample(row['id'],inp['state'],qs,gold,row.get('metadata',{}))

def read_td_dataset(config,max_examples=None):
    seen=set();rows=[]
    with open_dataset_text(config.path) as f:
        for line in f:
            if not line.strip():continue
            e=normalize_td(json.loads(line))
            if e.id in seen:raise ValueError('Duplicate TD id: '+e.id)
            seen.add(e.id);rows.append(e)
    return rows


def select_task(examples,task_id):
    """Project metadata.tasks before sampling; never send another task's gold/questions."""
    selected=[]
    for e in examples:
        names=[q for q in e.questions if e.metadata.get('tasks',{}).get(q)==task_id]
        if not names:continue
        meta=dict(e.metadata);meta['tasks']={q:task_id for q in names}
        selected.append(TDExample(e.id,e.state,{q:e.questions[q] for q in names},{q:e.target[q] for q in names},meta))
    return selected
