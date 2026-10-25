import httpx
from types import SimpleNamespace
from bilgeval.td_reader import TDExample
from bilgeval.td_client import TDClient
from bilgeval.typed_decision import evaluate_context

def test_task_projection_does_not_leak_other_questions():
 from bilgeval.td_reader import select_task
 qa={'type':'noul','instructions':'A'};qb={'type':'score','instructions':'B','criteria':['0','1']}
 a=TDExample('a','s',{'a':qa,'b':qb},{'a':True,'b':1},{'tasks':{'a':'basic.a','b':'basic.b'}})
 b=TDExample('b','t',{'b':qb},{'b':0},{'tasks':{'b':'basic.b'}})
 selected=select_task([a,b],'basic.a')
 assert len(selected)==1 and selected[0].id=='a' and selected[0].questions=={'a':qa} and selected[0].target=={'a':True}
 assert set(a.questions)=={'a','b'}

def test_results_include_actual_p03_policy():
 q={'type':'noul','instructions':'A'};a=TDExample('a','s',{'q':q},{'q':True},{'tasks':{'q':'basic.a'}})
 c=TDClient('http://test/v1','model@1',transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'model':'model@1','answers':{'q':{'type':'noul','noul':.8}},'usage':{'input_tokens':1,'output_tokens':0}})))
 check=SimpleNamespace(config={'family':'bilge','noul_criteria_policy':'ignored'})
 class Check:
  config=check.config
  def __call__(self,e):return None
 ctx=SimpleNamespace(examples=[a],client=lambda role:c,td_length_check=Check(),run=SimpleNamespace(max_examples=None,suite_name=None),progress_callback=None)
 result=evaluate_context(ctx)
 assert result['td_rows'][0]['noul_criteria_policy']=='ignored'
