import json
from types import SimpleNamespace
import httpx
import pytest
from bilgeval.td_reader import TDExample
from bilgeval.td_client import TDClient
from bilgeval.typed_decision import evaluate_examples
from bilgeval.td_metrics import metrics

def mock_client():
 def handle(request):
  body=json.loads(request.content);assert set(body)=={'model','state','questions'};answers={}
  for name,q in body['questions'].items():
   if q['type']=='noul':answers[name]={'type':'noul','noul':.9}
   else:
    keys=list(q['criteria']) if q['type']=='choice' else [str(i) for i in range(len(q['criteria']))]
    p={k:float(i==0) for i,k in enumerate(keys)}
    answers[name]={'type':q['type'],'probabilities':p,'confidence':1.}
    if q['type']=='choice':answers[name]['choice']=keys[0]
    else:answers[name].update(score=0.,legend=dict(zip(keys,q['criteria'])))
  return httpx.Response(200,json={'model':'mock','answers':answers,'usage':{'input_tokens':1,'output_tokens':0}})
 return TDClient('http://test','mock',transport=httpx.MockTransport(handle))

def test_mixed_task_splits_types_and_keeps_one_request_per_state():
 qs={'n':{'type':'noul','instructions':'Is it true?'},'c':{'type':'choice','instructions':'Choose','criteria':{'A':'first','B':'second'}},'s':{'type':'score','instructions':'Rate','criteria':['low','high']}}
 e=TDExample('mixed','state',qs,{'n':True,'c':'A','s':0},{'tasks':dict.fromkeys(qs,'example.mixed')})
 c=mock_client();r=evaluate_examples([e],c,lambda e:None,full_set=True)
 assert c.calls==1 and r['td_metrics']['N']==3
 sub=r['td_metrics']['subreports'];assert set(sub)=={'example.mixed.noul','example.mixed.choice','example.mixed.score'}
 for m in sub.values():assert m['n']==m['n_valid']==1 and m['coverage']==m['accuracy']==1

def test_variable_choice_and_score_inventories():
 rows=[]
 for i,(kind,criteria) in enumerate([('choice',{'A':'a','B':'b'}),('choice',{'A':'a','B':'b','C':'c'}),('score',['low','high']),('score',['low','mid','high'])]):
  q={'type':kind,'instructions':'Select','criteria':criteria};gold='A' if kind=='choice' else 0
  rows.append(TDExample(str(i),'s',{'q':q},{'q':gold},{'tasks':{'q':'example.variable'}}))
 r=evaluate_examples(rows,mock_client(),lambda e:None,full_set=True)
 for m in r['td_metrics']['subreports'].values():
  assert m['macro_f1']=='N/A (karma sınıf kümesi)' and m['accuracy']==1 and m['n']==2 and m['ece']==0 and m['nll']==0

def test_stsb_raw_gold_is_carried_and_spearman_uses_it():
 q={'type':'score','instructions':'Rate','criteria':['0','1','2','3','4','5']}
 e=TDExample('stsb','s',{'q':q},{'q':3},{'tasks':{'q':'example.similarity'},'source_fields':{'score_raw':'2.500'}})
 r=evaluate_examples([e],mock_client(),lambda e:None,full_set=True);assert r['td_rows'][0]['gold_raw']==2.5
 labels=['0','1','2','3','4','5'];rows=[]
 for i,(gold,raw) in enumerate([('2',2.1),('2',2.2),('3',3.0)]):
  p=[0.]*6;p[i]=1.;rows.append(dict(type='score',labels=labels,gold=gold,gold_raw=raw,prediction=str(i),probabilities=p,status='ok',length_eligible=True,correct=False))
 assert metrics(rows,labels,full_set=True)['spearman']==pytest.approx(1.)




def _mixed_example():
 qs={'n':{'type':'noul','instructions':'Is it true?'},'c':{'type':'choice','instructions':'Choose','criteria':{'A':'first','B':'second'}},'s':{'type':'score','instructions':'Rate','criteria':['low','high']}}
 return TDExample('mixed','state',qs,{'n':True,'c':'A','s':0},{'tasks':dict.fromkeys(qs,'example.mixed')})

def test_per_question_mode_sends_one_request_per_question():
 c=mock_client();r=evaluate_examples([_mixed_example()],c,lambda e:None,full_set=True,request_mode='per_question')
 assert c.calls==3 and r['td_metrics']['N']==3
 for m in r['td_metrics']['subreports'].values():
  assert m['latency']['unit']=='per_question' and m['latency']['n_latency']==1 and m['latency']['questions_per_request']==1

def test_per_state_latency_counts_one_request_for_all_questions():
 c=mock_client();r=evaluate_examples([_mixed_example()],c,lambda e:None,full_set=True)
 lat=r['td_metrics']['latency']
 assert c.calls==1 and lat['unit']=='per_state_request' and lat['n_latency']==1 and lat['questions_per_request']==3

def test_latency_stats_values_and_exclusions():
 from bilgeval.td_metrics import latency_stats
 rows=[{'id':str(i),'request_id':str(i),'request_ok':True,'latency_ms':v,'request_mode':'per_question'} for i,v in enumerate((10.,20.,30.,40.))]
 rows.append({'id':'x','request_id':'x','request_ok':False,'latency_ms':999.,'request_mode':'per_question'})
 s=latency_stats(rows)
 assert s['n_latency']==4 and s['mean_ms']==25. and s['p50_ms']==25. and abs(s['p95_ms']-38.5)<1e-9
