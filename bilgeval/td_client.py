"""System One transport; exactly one POST per state, no hidden retry, no gold."""
import math
from urllib.parse import urlsplit
import httpx
from .td_reader import labels_for

def number(x):
    if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x):raise ValueError('Probability/value must be finite number')
    return float(x)

def validate_answer(q,a):
    if not isinstance(a,dict) or a.get('type')!=q['type']:raise ValueError('Answer type mismatch')
    labels=labels_for(q)
    if q['type']=='noul':
        p=number(a.get('noul'))
        if not 0<=p<=1:raise ValueError('Noul outside [0,1]')
        return {'labels':labels,'probabilities':[1-p,p],'prediction':'true' if p>=.5 else 'false'}
    probs=a.get('probabilities')
    if not isinstance(probs,dict) or set(probs)!=set(labels):raise ValueError('Probability keys mismatch')
    p=[number(probs[k]) for k in labels]
    if min(p)<0 or max(p)>1 or abs(sum(p)-1)>1e-3:raise ValueError('Invalid probability distribution; not normalized silently')
    conf=number(a.get('confidence'))
    if not 0<=conf<=1:raise ValueError('Invalid confidence')
    pred=labels[max(range(len(p)),key=lambda i:p[i])]
    if q['type']=='choice' and a.get('choice')!=pred:raise ValueError('Choice is not ordered argmax')
    if q['type']=='score':
        if a.get('legend')!={str(i):v for i,v in enumerate(q['criteria'])}:raise ValueError('Score legend mismatch')
        if abs(number(a.get('score'))-sum(i*v for i,v in enumerate(p)))>1e-6:raise ValueError('Score expectation mismatch')
    return {'labels':labels,'probabilities':p,'prediction':pred}

class TDClient:
    def __init__(self,base_url,model,api_key=None,timeout=60,transport=None):
        if not isinstance(model,str) or not model.strip():raise ValueError('TD model is required')
        self.model=model;self.calls=0;self.failed=0
        base=base_url.rstrip('/');self.url=base+'/systemone' if base.endswith('/v1') else base+'/v1/systemone'
        headers={}
        if api_key and urlsplit(base).hostname not in ('localhost','127.0.0.1','::1'):headers['Authorization']='Bearer '+api_key
        self.http=httpx.Client(timeout=timeout,headers=headers,transport=transport)
    def close(self):self.http.close()
    def decide(self,state,questions):
        self.calls+=1
        try:
            qs={name:{k:q[k] for k in ('type','instructions','criteria') if k in q} for name,q in questions.items()}
            response=self.http.post(self.url,json={'model':self.model,'state':state,'questions':qs})
            response.raise_for_status();body=response.json()
            if not isinstance(body,dict) or not isinstance(body.get('model'),str) or not isinstance(body.get('answers'),dict):raise ValueError('Malformed TD response')
            usage=body.get('usage',{})
            if type(usage.get('input_tokens')) is not int or usage['input_tokens']<0 or usage.get('output_tokens')!=0:raise ValueError('Invalid usage')
            if set(body['answers'])-set(questions):raise ValueError('Unexpected answer question IDs')
            return body
        except Exception:
            self.failed+=1;raise
    def probe(self):
        qs={'n':{'type':'noul','instructions':'Is two even?'},'c':{'type':'choice','instructions':'Choose','criteria':{'yes':'Yes','no':'No'}},'s':{'type':'score','instructions':'Rate','criteria':['low','high']}}
        b=self.decide('Two is even.',qs)
        if set(b['answers'])!=set(qs):raise ValueError('C-02: missing answer')
        for k,q in qs.items():validate_answer(q,b['answers'][k])
        return b
