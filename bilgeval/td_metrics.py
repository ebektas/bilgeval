"""Question-level metrics, including heterogeneous class inventories. No CI."""
import math
import numpy as np
LOW='N/A (coverage düşük)'
MIXED='N/A (karma sınıf kümesi)'
METRICS_VERSION='typed-decision-metrics/v2'

def selective(confidence,errors):
    n=len(errors)
    if not n:return {'N':0,'aurc':None,'augrc':None,'acc_at_80':None,'acc_at_50':None}
    pairs=sorted(zip(confidence,errors),reverse=True);ordered=[];i=0
    while i<n:
        j=i+1
        while j<n and pairs[j][0]==pairs[i][0]:j+=1
        ordered.extend([sum(e for _,e in pairs[i:j])/(j-i)]*(j-i));i=j
    prefix=np.cumsum(ordered)
    def acc(frac):
        mass=frac*n;whole=int(mass);partial=mass-whole
        error=sum(ordered[:whole])+(ordered[whole]*partial if whole<n else 0)
        return 1-error/mass
    return {'N':n,'aurc':float(np.mean(prefix/np.arange(1,n+1))),'augrc':float(np.mean(prefix/n)),'acc_at_80':acc(.8),'acc_at_50':acc(.5)}

def ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);r=[0.]*len(values);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        for k in order[i:j]:r[k]=(i+j-1)/2
        i=j
    return r

def latency_stats(rows):
    seen={}
    for r in rows:
        if r.get('request_ok') and r.get('latency_ms') is not None and r.get('length_eligible',True):
            seen.setdefault(r.get('request_id',r['id']),(r['latency_ms'],r.get('questions_per_request',1)))
    unit='per_question' if rows and rows[0].get('request_mode')=='per_question' else 'per_state_request'
    out={'unit':unit,'n_latency':len(seen),'mean_ms':None,'p50_ms':None,'p95_ms':None,'questions_per_request':None,'concurrency':1,'scope':'client-side; HTTP + JSON decode; failed and rejected requests excluded'}
    if seen:
        v=[x[0] for x in seen.values()]
        out.update(mean_ms=float(np.mean(v)),p50_ms=float(np.percentile(v,50)),p95_ms=float(np.percentile(v,95)),questions_per_request=float(np.mean([x[1] for x in seen.values()])))
    return out

def metrics(rows,labels,*,full_set):
    eligible=[r for r in rows if r.get('length_eligible',True)];valid=[r for r in eligible if r['status']=='ok'];n=len(eligible);nv=len(valid)
    coverage=nv/n if n else 0.;correct=sum(r['correct'] for r in valid)
    m={'N':n,'n_valid':nv,'coverage':coverage,'n_selected':len(rows),'model_coverage':n/len(rows) if rows else 0.,'accuracy':correct/n if n else None,'metric_scope':'full' if full_set else 'subset','confusion':{k:{**{p:0 for p in labels},'__error__':0} for k in labels}}
    m.update(n=n,metrics_version=METRICS_VERSION)
    m['latency']=latency_stats(eligible)
    uniform=len({frozenset(r['labels']) for r in rows})<=1
    if uniform:
        for r in eligible:m['confusion'][r['gold']][r['prediction'] if r['status']=='ok' else '__error__']+=1
        f1=[];class_acc={}
        for k in labels:
            tp=m['confusion'][k][k];support=sum(m['confusion'][k].values());fp=sum(m['confusion'][other][k] for other in labels if other!=k);fn=support-tp
            f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
            class_acc[k]=tp/support if support else None
        m['macro_f1']=sum(f1)/len(labels) if n else None;m['class_accuracy']=class_acc
    else:
        m.update(macro_f1=MIXED,class_accuracy=MIXED,confusion=None)
    kind=eligible[0]['type'] if eligible else (rows[0]['type'] if rows else None)
    if kind=='noul':
        tp=m['confusion']['true']['true'];tn=m['confusion']['false']['false'];pos=sum(m['confusion']['true'].values());neg=sum(m['confusion']['false'].values());fp=m['confusion']['false']['true']
        m.update(positive_f1=2*tp/(pos+tp+fp) if pos+tp+fp else 0.,recall=tp/pos if pos else None,specificity=tn/neg if neg else None,predicted_true=sum(r['prediction']=='true' for r in valid),predicted_false=sum(r['prediction']=='false' for r in valid))
    if kind=='score':
        m.update(exact=m['accuracy'],within_1=sum(abs(int(r['prediction'])-int(r['gold']))<=1 for r in valid)/n if n else None)
        if coverage<.99:m.update(argmax_mae=LOW,expected_mae=LOW,spearman=LOW)
        else:
            gold=[float(r.get('gold_raw',r['gold'])) for r in valid];expected=[sum(i*p for i,p in enumerate(r['probabilities'])) for r in valid]
            gr=ranks(gold);pr=ranks(expected)
            rho=float(np.corrcoef(gr,pr)[0,1]) if len(set(gr))>1 and len(set(pr))>1 else None
            m.update(argmax_mae=sum(abs(int(r['prediction'])-int(r['gold'])) for r in valid)/nv,expected_mae=sum(abs(p-int(r['gold'])) for p,r in zip(expected,valid))/nv,spearman=rho)
    pairs={}
    for r in eligible:
        if r.get('pair_id'):pairs.setdefault(r['pair_id'],[]).append(r)
    if pairs:
        complete=[p for p in pairs.values() if len(p)==2]
        m['pair_count']=len(complete);m['incomplete_pairs']=sum(len(p)!=2 for p in pairs.values())
        m['robust_accuracy']=sum(all(r['status']=='ok' and r['correct'] for r in p) for p in complete)/len(complete) if complete else None
        m['pair_consistency']=sum(all(r['status']=='ok' for r in p) and p[0]['prediction']==p[1]['prediction'] for p in complete)/len(complete) if complete else None
    if not full_set:return m
    keys=['ece','brier','categorical_brier','nll','nll_clipped','hc_errors_90','hc_errors_95','wrong_answer_confidence','aurc','augrc','acc_at_80','acc_at_50']
    if coverage<.99:
        m.update({k:LOW for k in keys});return m
    conf=[max(r['probabilities']) for r in valid];err=[int(not r['correct']) for r in valid];ece=0.;brier=[];cat=[];nll=[];clipped=0
    for b in range(10):
        ix=[i for i,c in enumerate(conf) if min(9,int(c*10))==b]
        if ix:ece+=len(ix)/nv*abs(sum(conf[i] for i in ix)/len(ix)-sum(1-err[i] for i in ix)/len(ix))
    for r in valid:
        target=r['labels'].index(r['gold']);p=r['probabilities'];v=sum((x-(j==target))**2 for j,x in enumerate(p));cat.append(v);brier.append((p[1]-(target==1))**2 if r['type']=='noul' else v)
        clipped+=p[target]<1e-12;nll.append(-math.log(max(p[target],1e-12)))
    m.update(ece=ece,brier=sum(brier)/nv,categorical_brier=sum(cat)/nv,nll=sum(nll)/nv,nll_clipped=clipped,hc_errors_90=sum(e and c>=.90 for e,c in zip(err,conf)),hc_errors_95=sum(e and c>=.95 for e,c in zip(err,conf)),wrong_answer_confidence=sum(c for e,c in zip(err,conf) if e)/sum(err) if sum(err) else None)
    m.update({k:v for k,v in selective(conf,err).items() if k!='N'});m['selective_N']=nv;m['selective_estimator']='right-endpoint discrete mean; tie-average errors'
    return m
