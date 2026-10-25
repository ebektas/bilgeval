"""Question-level TD evaluation, typed subreports and common-set comparisons."""
import json,time
from .td_reader import labels_for
from .td_client import validate_answer
from .td_metrics import metrics, latency_stats, METRICS_VERSION, MIXED

def evaluate_examples(examples,client,length_check,*,full_set,progress_callback=None,noul_criteria_policy=None,request_mode='per_state'):
    if request_mode not in ('per_state','per_question'):raise ValueError('request_mode must be per_state or per_question')
    rows=[];start_calls=client.calls;start_failed=client.failed
    check=getattr(length_check,'check',None)
    for index,e in enumerate(examples):
        units=[list(e.questions)] if request_mode=='per_state' else [[qid] for qid in e.questions]
        for unit in units:
            sub={qid:e.questions[qid] for qid in unit}
            reason=check(e.state,sub) if (check and request_mode=='per_question') else length_check(e)
            body=None;error=None;elapsed=None
            if reason is None:
                start=time.perf_counter()
                try:body=client.decide(e.state,sub)
                except Exception as exc:error=f'{type(exc).__name__}: {exc}'
                elapsed=(time.perf_counter()-start)*1000
            request_id=e.id if request_mode=='per_state' else json.dumps([e.id,unit[0]],ensure_ascii=False)
            for qid in unit:
                q=sub[qid];gold=e.target[qid];gold=str(gold).lower() if q['type']=='noul' else str(gold)
                r={'id':json.dumps([e.id,qid],ensure_ascii=False),'state_id':e.id,'question_id':qid,'task_id':e.metadata.get('tasks',{}).get(qid,qid),'noul_criteria_policy':noul_criteria_policy,'type':q['type'],'labels':labels_for(q),'gold':gold,'prediction':None,'probabilities':None,'correct':False,'length_eligible':reason is None,'status':'işlenmedi' if reason else 'error','reason':reason or error,'latency_ms':elapsed,'request_id':request_id,'request_ok':body is not None,'request_mode':request_mode,'questions_per_request':len(unit),'pair_id':e.metadata.get('pair_id')}
                raw_gold=e.metadata.get('source_fields',{}).get('score_raw')
                if q['type']=='score' and raw_gold is not None:r['gold_raw']=float(raw_gold)
                if body is not None:
                    try:
                        a=validate_answer(q,body['answers'].get(qid));r.update(a,status='ok',reason=None,correct=a['prediction']==gold,served_model=body['model'])
                    except ValueError as exc:r['reason']=str(exc)
                rows.append(r)
        if progress_callback:progress_callback(index+1,len(examples),sum(r['status']=='error' for r in rows))
    m=summarize(rows,full_set=full_set)
    return {'metric_name':'accuracy','score':m['accuracy'],'score_scale':{'min':0.,'max':1.,'higher_is_better':True},'td_metrics':m,'td_rows':rows,'results':rows,'num_examples':m['N'],'num_scored_examples':m['n_valid'],'num_failed_examples':m['N']-m['n_valid'],'coverage':m['coverage'],'num_candidate_scores':m['n_valid'],'num_requests':client.calls-start_calls,'td_failed_requests':client.failed-start_failed,'failure_policy':'Endpoint errors count as incorrect; length rejects are outside N'}

def summarize(rows,*,full_set):
    groups={}
    for row in rows:groups.setdefault(row['type'],[]).append(row)
    subreports={}
    for kind,group in groups.items():
        task=group[0].get('task_id','typed_decision')
        subreports[task+'.'+kind]=metrics(group,group[0]['labels'],full_set=full_set)
    if len(groups)==1:
        result=dict(next(iter(subreports.values())))
    else:
        eligible=[r for r in rows if r.get('length_eligible',True)]
        valid=[r for r in eligible if r['status']=='ok'];n=len(eligible);nv=len(valid)
        result={'N':n,'n':n,'n_valid':nv,'n_selected':len(rows),'coverage':nv/n if n else 0.,'model_coverage':n/len(rows) if rows else 0.,'accuracy':sum(r['correct'] for r in valid)/n if n else None,'macro_f1':MIXED,'metric_scope':'full' if full_set else 'subset','metrics_version':METRICS_VERSION}
    result['latency']=latency_stats(rows)
    result['subreports']=subreports
    return result

def evaluate_context(ctx):
    r=evaluate_examples(ctx.examples,ctx.client('td'),ctx.td_length_check,full_set=ctx.run.max_examples is None and 'max200' not in (ctx.run.suite_name or ''),progress_callback=ctx.progress_callback,request_mode=getattr(ctx.td_length_check,'request_mode','per_state'),noul_criteria_policy=ctx.td_length_check.config.get("noul_criteria_policy", "used" if ctx.td_length_check.config.get("family")=="laya" else None))
    return r

def common_table(model_rows,labels,*,full_set):
    if not model_rows:return {'status':'ortak küme boş','common_N':0,'models':[]}
    maps=[]
    for m in model_rows:
        keyed={(r['state_id'],r['question_id']):r for r in m['rows']}
        if len(keyed)!=len(m['rows']):raise ValueError('Duplicate state/question identity')
        maps.append(keyed)
    if any(set(m)!=set(maps[0]) for m in maps[1:]):raise ValueError('Common-set reports require identical selected IDs')
    for k in maps[0]:
        if any((m[k]['gold'],m[k]['labels'],m[k]['type'])!=(maps[0][k]['gold'],maps[0][k]['labels'],maps[0][k]['type']) for m in maps[1:]):raise ValueError('Common-set gold/schema mismatch')
    common=set.intersection(*[{k for k,r in m.items() if r['length_eligible']} for m in maps]);out=[]
    for model,m in zip(model_rows,maps):
        out.append({'model':model['model'],'model_coverage':sum(r['length_eligible'] for r in m.values())/len(m) if m else 0.,'td_metrics':summarize([m[k] for k in sorted(common)],full_set=full_set) if common else None})
    return {'status':'ok' if common else 'ortak küme boş','common_N':len(common),'models':out}

def common_reports(reports):
    tables={}
    runs=set.intersection(*[set(r['datasets']) for r in reports]) if reports else set()
    for run in sorted(runs):
        entries=[r['datasets'][run] for r in reports]
        if not all('td_rows' in e for e in entries):continue
        sig=[(e['task_version'],e['sample_seed'],e['max_examples'],e['td_identity']['data_hash'],e['td_identity']['metrics'],e['td_identity'].get('request_mode','per_state')) for e in entries]
        if len(set(sig))!=1:raise ValueError('Incompatible TD comparison selection')
        model_rows=[{'model':r['model'],'rows':e['td_rows']} for r,e in zip(reports,entries)]
        labels=entries[0]['td_rows'][0]['labels']
        tables[run]=common_table(model_rows,labels,full_set=entries[0]['td_metrics']['metric_scope']=='full')
    return tables
