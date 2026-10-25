import math
from types import SimpleNamespace
import httpx
import pytest
from bilgeval.td_client import TDClient, validate_answer
from bilgeval.td_reader import TDExample, read_td_dataset
from bilgeval.td_metrics import metrics, selective
from bilgeval.typed_decision import evaluate_examples, common_table

Q={'type':'choice','instructions':'Choose','criteria':{'a':'first','b':'second'}}
def answer(p=.8):return {'type':'choice','choice':'a' if p>=.5 else 'b','confidence':max(p,1-p),'probabilities':{'a':p,'b':1-p}}
def ex(i='x',questions=None,target=None):return TDExample(i,'state',questions or {'q':Q},target or {'q':'a'},{})
def client(handler):return TDClient('http://test/v1','model@1',transport=httpx.MockTransport(handler))
def row(g='a',p=(.8,.2),kind='choice',i='x'):
 labels=['false','true'] if kind=='noul' else ['0','1'] if kind=='score' else ['a','b']
 pred=labels[max(range(len(p)),key=lambda j:p[j])]
 if kind=='noul':pred='true' if p[1]>=.5 else 'false'
 return dict(id=i,state_id=i,question_id='q',type=kind,labels=labels,gold=g,prediction=pred,probabilities=list(p),status='ok',length_eligible=True,correct=pred==g)

def test_payload_gold_and_one_request():
 calls=[]
 def handler(request):
  import json
  body=json.loads(request.content);calls.append(body)
  assert set(body)=={'model','state','questions'} and body['model']=='model@1'
  return httpx.Response(200,json={'model':'model@1','answers':{'q':answer()},'usage':{'input_tokens':3,'output_tokens':0}})
 r=evaluate_examples([ex()],client(handler),lambda e:None,full_set=True)
 assert len(calls)==1 and r['td_metrics']['accuracy']==1 and r['results'][0]['state_id']=='x'

@pytest.mark.parametrize('p',[-.1,float('nan'),float('inf'),1.1])
def test_invalid_probability(p):
 a=answer();a['probabilities']['a']=p
 with pytest.raises(ValueError):validate_answer(Q,a)

def test_id_alignment_partial_error():
 c=client(lambda req:httpx.Response(200,json={'model':'model@1','answers':{'second':answer()},'usage':{'input_tokens':1,'output_tokens':0}}))
 r=evaluate_examples([ex(questions={'first':Q,'second':Q},target={'first':'a','second':'a'})],c,lambda e:None,full_set=True)
 assert [(x['question_id'],x['status']) for x in r['results']]==[('first','error'),('second','ok')]
 assert r['td_metrics']['accuracy']==.5 and r['td_metrics']['coverage']==.5
 assert r['td_metrics']['ece']=='N/A (coverage düşük)'

def test_length_rejection_no_request():
 c=client(lambda req:pytest.fail('must not send'))
 r=evaluate_examples([ex()],c,lambda e:'head_would_truncate',full_set=True)
 assert r['results'][0]['status']=='işlenmedi' and r['results'][0]['reason']=='head_would_truncate'
 assert r['td_metrics']['N']==0 and r['td_metrics']['model_coverage']==0

def test_hand_calibration():
 m=metrics([row(),row(g='b',p=(.6,.4),i='y')],['a','b'],full_set=True)
 assert m['accuracy']==.5 and m['macro_f1']==pytest.approx(1/3)
 assert m['ece']==pytest.approx(.4)
 assert m['brier']==pytest.approx(.4) and m['categorical_brier']==pytest.approx(.4)
 assert m['nll']==pytest.approx(-(math.log(.8)+math.log(.4))/2)
 assert m['wrong_answer_confidence']==.6 and m['hc_errors_90']==0
 assert m['aurc']==.25 and m['augrc']==.25
 assert m['acc_at_50']==1 and m['acc_at_80']==pytest.approx(.625)

def test_noul_and_score():
 m=metrics([row('true',(.1,.9),'noul'),row('false',(.8,.2),'noul','y')],['false','true'],full_set=True)
 assert m['positive_f1']==m['recall']==m['specificity']==1
 assert m['brier']==pytest.approx(.025) and m['categorical_brier']==pytest.approx(.05)
 s=metrics([row('1',(.2,.8),'score'),row('0',(.7,.3),'score','y')],['0','1'],full_set=True)
 assert s['exact']==s['within_1']==1
 assert s['spearman']==pytest.approx(1)
 assert s['argmax_mae']==0 and s['expected_mae']==pytest.approx(.25)

def test_ties_fractional_and_max200():
 s=selective([.7,.7],[0,1]);assert s['aurc']==.5 and s['augrc']==.375 and s['acc_at_50']==.5
 m=metrics([row()],['a','b'],full_set=False)
 for key in ['ece','brier','categorical_brier','nll','aurc','augrc','acc_at_80','acc_at_50']:assert key not in m

def test_common_empty():
 r=row();r.update(length_eligible=False,status='işlenmedi')
 t=common_table([{'model':'a','rows':[row()]},{'model':'b','rows':[r]}],['a','b'],full_set=True)
 assert t['status']=='ortak küme boş' and t['common_N']==0

def test_reader_duplicate_and_gold_domain(tmp_path):
 import json
 row={'schema_version':'td.instance/v1','task_type':'typed_decision','id':'i','input':{'state':'s','questions':{'q':Q}},'target':{'q':'a'}}
 p=tmp_path/'data.jsonl';p.write_text(json.dumps(row)+'\n'+json.dumps(row))
 with pytest.raises(ValueError,match='Duplicate'):read_td_dataset(SimpleNamespace(path=p))
 row['target']['q']='unknown';p.write_text(json.dumps(row))
 with pytest.raises(ValueError,match='gold'):read_td_dataset(SimpleNamespace(path=p))

def test_http_error_and_extra_id():
 for response in [httpx.Response(503),httpx.Response(200,json={'model':'m','answers':{'wrong_id':answer()},'usage':{'input_tokens':1,'output_tokens':0}})]:
  c=client(lambda request:response)
  r=evaluate_examples([ex()],c,lambda e:None,full_set=True)
  assert r['td_metrics']['accuracy']==0 and c.calls==1 and r['results'][0]['status']=='error'

def test_score_contract_and_noul_boundary():
 q={'type':'score','instructions':'Rate','criteria':['low','high']}
 a={'type':'score','score':.7,'confidence':.7,'legend':{'0':'low','1':'high'},'probabilities':{'0':.3,'1':.7}}
 assert validate_answer(q,a)['prediction']=='1'
 a['score']=1
 with pytest.raises(ValueError):validate_answer(q,a)
 assert validate_answer({'type':'noul'},{'type':'noul','noul':.5})['prediction']=='true'
 assert validate_answer(Q,answer(.5))['prediction']=='a'

def test_coverage_gate_boundary_and_nll_clipping():
 good=[row(i=str(i)) for i in range(100)]
 good[-1].update(status='error',prediction=None,probabilities=None,correct=False)
 m=metrics(good,['a','b'],full_set=True);assert m['coverage']==.99 and isinstance(m['nll'],float)
 good[-2].update(status='error',prediction=None,probabilities=None,correct=False)
 m=metrics(good,['a','b'],full_set=True);assert m['nll']=='N/A (coverage düşük)'
 m=metrics([row('b',(1.,0.))],['a','b'],full_set=True)
 assert m['nll_clipped']==1 and m['nll']==pytest.approx(-math.log(1e-12))
 assert m['hc_errors_90']==m['hc_errors_95']==1 and m['wrong_answer_confidence']==1

def test_noul_error_is_wrong_in_specificity_and_recall():
 rows=[row('true',(.1,.9),'noul','1'),row('false',(.9,.1),'noul','2')]
 rows[1].update(status='error',probabilities=None,prediction=None,correct=False)
 m=metrics(rows,['false','true'],full_set=True)
 assert m['accuracy']==.5 and m['specificity']==0 and m['recall']==1

def test_pair_metrics_and_absent_class():
 a=row(i='a');b=row('b',(.6,.4),i='b');a['pair_id']=b['pair_id']='pair'
 m=metrics([a,b],['a','b','c'],full_set=False)
 assert m['robust_accuracy']==0 and m['pair_consistency']==1 and m['macro_f1']==pytest.approx(2/9)



def test_preflight_c01_c06():
 def handler(request):
  import json
  qs=json.loads(request.content)['questions'];answers={}
  for k,q in qs.items():
   if q['type']=='noul':answers[k]={'type':'noul','noul':.7}
   elif q['type']=='choice':answers[k]={'type':'choice','choice':'yes','confidence':.7,'probabilities':{'yes':.7,'no':.3}}
   else:answers[k]={'type':'score','score':.7,'confidence':.7,'probabilities':{'0':.3,'1':.7},'legend':{'0':'low','1':'high'}}
  return httpx.Response(200,json={'model':'model@1','answers':answers,'usage':{'input_tokens':1,'output_tokens':0}})
 c=client(handler);c.probe();assert c.calls==1

@pytest.fixture
def td_config(tmp_path):
 import hashlib
 tokenizers=pytest.importorskip('tokenizers')
 Tokenizer,models,pre_tokenizers=tokenizers.Tokenizer,tokenizers.models,tokenizers.pre_tokenizers
 t=Tokenizer(models.WordLevel({'[UNK]':0,'word':1,'[CLS]':2},unk_token='[UNK]'));t.pre_tokenizer=pre_tokenizers.Whitespace()
 p=tmp_path/'tokenizer.json';t.save(str(p))
 return {'length':{'family':'laya','tokenizer_path':str(p),'tokenizer_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},'identity':{k:None for k in ['model_version','adapter_version','adapter_code_sha256','engine','engine_version','image_digest','dtype','head_dtype','flags','temperature_file_sha256','noul_criteria_policy']}}

def test_length_budgets(td_config):
 from bilgeval.td_lengths import LengthCheck
 check=LengthCheck(td_config['length'])
 assert check(ex()) is None
 assert check(TDExample('i','word '*1100,{'q':Q},{'q':'a'},{}))=='sequence_gt_1024'
 huge={'type':'choice','instructions':'Choose','criteria':{str(i):'word '*8 for i in range(60)}}
 assert check(ex(questions={'q':huge},target={'q':'0'}))=='head_or_option_would_truncate'
 huge={'type':'choice','instructions':'Choose','criteria':{'a':'word '*49,'b':'word'}}
 assert check(ex(questions={'q':huge}))=='head_or_option_would_truncate'
 td_config['length']['family']='bilge';check=LengthCheck(td_config['length'])
 assert check(TDExample('i','word '*2100,{'q':Q},{'q':'a'},{}))=='sequence_gt_2048'




def test_common_nonempty_counts_errors_as_wrong():
 a=[row(i='a'),row(i='b')];b=[row(i='a'),row(i='b')]
 a[1].update(length_eligible=False,status='işlenmedi');b[0].update(status='error',correct=False,prediction=None,probabilities=None)
 c=common_table([{'model':'a','rows':a},{'model':'b','rows':b}],['a','b'],full_set=True)
 assert c['common_N']==1 and c['models'][0]['model_coverage']==.5
 assert c['models'][0]['td_metrics']['accuracy']==1 and c['models'][1]['td_metrics']['accuracy']==0


def test_score_nonexact_and_all_wrong_selective():
 r=row('1',(.8,.2),'score');m=metrics([r],['0','1'],full_set=True)
 assert m['exact']==0 and m['within_1']==1 and m['argmax_mae']==1 and m['expected_mae']==pytest.approx(.8)
 assert m['spearman'] is None
 s=selective([.9,.8],[1,1]);assert s['aurc']==1 and s['augrc']==.75

def test_probability_tolerance_does_not_normalize():
 a={'type':'choice','choice':'b','confidence':.6,'probabilities':{'a':.4,'b':.5995}}
 assert validate_answer(Q,a)['probabilities']==[.4,.5995]
 a['probabilities']['b']=.59
 with pytest.raises(ValueError):validate_answer(Q,a)

