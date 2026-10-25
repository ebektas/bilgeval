"""Offline, nontruncating preflight. No model weights or remote tokenizer lookup."""
import hashlib,json
from pathlib import Path

class LengthCheck:
    def __init__(self,config):
        from tokenizers import Tokenizer
        self.config=config;path=Path(config['tokenizer_path'])
        self.sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        if self.sha256!=config['tokenizer_sha256']:raise ValueError('TD tokenizer SHA-256 mismatch')
        self.tok=Tokenizer.from_file(str(path));self.tok.no_truncation();self.tok.no_padding()
        if config['family'] not in ('laya','bilge'):raise ValueError('Unknown TD length profile')
    def encode(self,s):return self.tok.encode(s,add_special_tokens=False).ids
    def __call__(self,example):
        return self.check(example.state,example.questions)
    def check(self,state,questions):
        cfg=self.config
        if cfg['family']=='bilge':
            if len(questions)>64:return 'question_count_gt_64'
            state=state if isinstance(state,str) else json.dumps(state,ensure_ascii=False,sort_keys=True)
            for q in questions.values():
                prefix=f"Context:\n{state}\n\nQuestion: {q['instructions']}\n"
                if q['type']=='noul':
                    if q.get('criteria') and cfg.get('noul_criteria_policy')!='ignored':raise ValueError('Noul criteria policy is unresolved')
                    seqs=[prefix+'Answer (Yes or No):']
                else:
                    cr=q['criteria']
                    opts=[k+': '+v if v else k for k,v in cr.items()] if q['type']=='choice' else cr
                    if q['type']=='choice' and not 2<=len(opts)<=64:return 'choice_option_count_out_of_range'
                    if q['type']=='score' and not 1<=len(opts)<=10:return 'score_level_count_out_of_range'
                    seqs=[prefix+f'Proposed answer: {o}\nIs this proposed answer correct? Answer (Yes or No):' for o in opts]
                if max(1+len(self.encode(s)) for s in seqs)>2048:return 'sequence_gt_2048'
            return None
        state=state if isinstance(state,str) else json.dumps(state,ensure_ascii=False)
        st=len(self.encode(state.replace('<mask>',' ')))
        for q in questions.values():
            cr=q.get('criteria') or {};kind=q['type']
            if kind=='choice':opts=[k+': '+v if v else k for k,v in cr.items()]
            elif kind=='score':opts=[f'level {i}: {v}' for i,v in enumerate(cr)]
            else:opts=['false: '+(cr.get('false') or 'no, the statement does not hold'),'true: '+(cr.get('true') or 'yes, the statement holds')]
            head=len(self.encode(f"{kind} question: {q['instructions']}".replace('<mask>',' ')))
            lens=[len(self.encode(' '+o.replace('<mask>',' '))) for o in opts]
            capped=[min(n,48)+1 for n in lens];budget=256-sum(capped)
            if budget<16:
                cap=max(4,(256-16)//len(capped));capped=[min(n,cap) for n in capped];budget=256-sum(capped)
            if head>max(8,budget) or any(n+1>c for n,c in zip(lens,capped)):return 'head_or_option_would_truncate'
            if 4+head+sum(n+1 for n in lens)+st>1024:return 'sequence_gt_1024'
        return None


class NoLocalLengthCheck:
    """No tokenizer profile was supplied; the endpoint must enforce its own limits."""
    config={'family':'unspecified','noul_criteria_policy':None}
    def __call__(self,example):return None


def prepare_runtime(options,model):
    from .td_metrics import METRICS_VERSION
    options=(options or {}).get('models',{}).get(model,options or {})
    profile=options.get('length')
    check=LengthCheck(profile) if profile else NoLocalLengthCheck()
    mode=options.get('request_mode','per_state')
    if mode not in ('per_state','per_question'):raise ValueError('request_mode must be per_state or per_question')
    check.request_mode=mode
    identity={key:None for key in ('model_version','adapter_version','adapter_code_sha256','engine','engine_version','image_digest','dtype','head_dtype','flags','temperature_file_sha256','noul_criteria_policy')}
    identity.update(options.get('identity',{}))
    length={k:v for k,v in (profile or {}).items() if k!='tokenizer_path'}
    if profile:
        family=profile['family'];length['max_input_tokens_per_sequence']=1024 if family=='laya' else 2048
        if family=='laya':length.update(head_max_len=256,option_max_len=48)
    else:length['status']='not_configured; endpoint_enforced'
    return check,{**identity,'model':model,'length':length,'protocol':'systemone-v1','request_mode':mode,'metrics':METRICS_VERSION,'ci':False}
