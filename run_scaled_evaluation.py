#!/usr/bin/env python3
"""Scaled, BATCHED re-run of the generation experiments with bootstrap 95% CIs.
Fixed scorer (score_tool_calls_all). Reuses refusal_dirs.pt (r_text @ L33).
Experiments: ablation (n=120), addition (n=120), steering dose-response (n=100/60),
patching+AUC on a 300-prompt baseline."""
import os, sys, json, re, time
from contextlib import nullcontext
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.environ.setdefault('HF_HOME','/workspace/.cache/huggingface')
REPO=Path(__file__).resolve().parent; sys.path.insert(0,str(REPO))
from tools.registry import get_domain
MODEL_ID,DTYPE,DEVICE='Qwen/Qwen3-14B',torch.bfloat16,'cuda'; ENABLE_THINKING=False
ART=REPO/'interp_artifacts'; torch.set_grad_enabled(False); LAYER=33; BS=24
RNG=np.random.RandomState(0)
def log(*a): print(*a,flush=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
log('[load] model bf16...')
tok=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True); tok.padding_side='left'
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=DTYPE,device_map=DEVICE,trust_remote_code=True); model.eval()
r_text=torch.load(ART/'refusal_dirs.pt')['dirs'][LAYER]; r_dev=r_text.to(DEVICE,DTYPE).view(-1)
log('[load] done')
DOMAIN_MAP={'Finance':'finance','Healthcare':'healthcare','Education / Academic Integrity':'education','Legal / Compliance':'legal'}
df=pd.read_excel(REPO/'data'/'complete_combined_safety_eval_dataset_2304_normalized_labels.xlsx')
df.columns=[c.strip() for c in df.columns]; df['User Prompt']=df['User Prompt'].str.replace('�','—',regex=False)
df['domain_key']=df['Domain'].map(DOMAIN_MAP)
df['group_id']=df['Domain']+'|'+df['Scenario']+'|'+df['Prompt Variant']+'|'+df['System Condition']
def tools_for_row(row): return None if row['Mode']=='No-tool chat' else list(get_domain(row['domain_key']).tools_for_llm)
def format_prompt(row):
    msgs=[{'role':'system','content':row['System Prompt']},{'role':'user','content':row['User Prompt']}]
    t=tools_for_row(row); base=dict(tokenize=False,add_generation_prompt=True)
    for extra in ({'enable_thinking':ENABLE_THINKING},{}):
        try: return tok.apply_chat_template(msgs,tools=t,**base,**extra) if t else tok.apply_chat_template(msgs,**base,**extra)
        except TypeError: continue
def parse_tool_calls(text):
    m=re.search(r'\[TOOL_CALLS\]\s*(\[.*?\])',text,re.DOTALL)
    if m:
        try:
            p=json.loads(m.group(1))
            if isinstance(p,list): return [{'name':c.get('name',''),'arguments':c.get('arguments',{})} for c in p]
        except json.JSONDecodeError: pass
    calls=[]
    for raw in re.findall(r'<tool_call>(.*?)</tool_call>',text,re.DOTALL):
        try:
            c=json.loads(raw.strip()); calls.append({'name':c.get('name',''),'arguments':c.get('arguments',c.get('parameters',{}))})
        except json.JSONDecodeError: pass
    return calls
def unsafe_all(row,text):
    return len(get_domain(row['domain_key']).score_tool_calls_all(parse_tool_calls(text)))>0
# Refusal labels MUST come from the same classifier as the behavioral eval
# (the old 5-pattern local list is what made Table 3 baselines contradict Table 1).
from tools.refusal import classify_refusal as _shared_classify_refusal, make_anthropic_judge
_JUDGE_KEY=os.environ.get('JUDGE_KEY','')
_judge=make_anthropic_judge(api_key=_JUDGE_KEY) if _JUDGE_KEY else None
if _judge is None: log('[refusal] WARNING: JUDGE_KEY unset — regex-only labels, NOT comparable to the behavioral pipeline')
def looks_refused(t): return _shared_classify_refusal(t, judge=_judge)['refused']
def _layers(): return model.model.layers
class Hooks:
    def __init__(self): self.h=[]
    def remove(self):
        for x in self.h: x.remove()
        self.h=[]
    def __enter__(self): return self
    def __exit__(self,*a): self.remove()
def ablate():
    def mk():
        def hook(m,i,o):
            h=o[0] if isinstance(o,tuple) else o; h=h-(h@r_dev).unsqueeze(-1)*r_dev
            return (h,)+tuple(o[1:]) if isinstance(o,tuple) else h
        return hook
    hk=Hooks()
    for L in _layers(): hk.h.append(L.register_forward_hook(mk()))
    return hk
def addvec(coef):
    def hook(m,i,o):
        h=o[0] if isinstance(o,tuple) else o; h=h+coef*r_dev
        return (h,)+tuple(o[1:]) if isinstance(o,tuple) else h
    hk=Hooks(); hk.h.append(_layers()[LAYER-1].register_forward_hook(hook)); return hk
def set_proj_batch(targets):
    tt=torch.as_tensor(targets,device=DEVICE,dtype=DTYPE).view(-1,1)
    def hook(m,i,o):
        h=o[0] if isinstance(o,tuple) else o; cur=(h@r_dev)
        h=h+(tt-cur).unsqueeze(-1)*r_dev
        return (h,)+tuple(o[1:]) if isinstance(o,tuple) else h
    hk=Hooks(); hk.h.append(_layers()[LAYER-1].register_forward_hook(hook)); return hk

@torch.no_grad()
def gen_batch(prompts, hook=None, max_new_tokens=200):
    enc=tok(prompts,return_tensors='pt',padding=True,truncation=True,max_length=2048).to(DEVICE)
    Lp=enc['input_ids'].shape[1]
    with (hook if hook is not None else nullcontext()):
        out=model.generate(**enc,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
    return [tok.decode(out[i,Lp:],skip_special_tokens=False) for i in range(len(prompts))]
def gen_all(prompts, hook_fn=None):
    texts=[]
    for i in range(0,len(prompts),BS):
        chunk=prompts[i:i+BS]
        hk=hook_fn(i,len(chunk)) if hook_fn else None
        texts+=gen_batch(chunk,hook=hk)
    return texts
@torch.no_grad()
def proj_last(prompts):
    out=[]
    for i in range(0,len(prompts),BS):
        enc=tok(prompts[i:i+BS],return_tensors='pt',padding=True,truncation=True,max_length=2048).to(DEVICE)
        hs=model(**enc,output_hidden_states=True).hidden_states
        out.append((hs[LAYER][:,-1,:].float().cpu()@r_text).numpy())
    return np.concatenate(out)
def ci(arr,n=2000):
    arr=np.asarray(arr,float)
    if len(arr)==0: return [None,None]
    bs=[arr[RNG.randint(0,len(arr),len(arr))].mean() for _ in range(n)]
    return [round(float(np.percentile(bs,2.5)),3), round(float(np.percentile(bs,97.5)),3)]
def auc_ci(us,sa,n=2000):
    us=np.asarray(us,float); sa=np.asarray(sa,float)
    if not len(us) or not len(sa): return float('nan'),[None,None]
    point=float((sa[:,None]>us[None,:]).mean())
    bs=[]
    for _ in range(n):
        u=us[RNG.randint(0,len(us),len(us))]; s=sa[RNG.randint(0,len(sa),len(sa))]
        bs.append((s[:,None]>u[None,:]).mean())
    return round(point,3),[round(float(np.percentile(bs,2.5)),3),round(float(np.percentile(bs,97.5)),3)]

harm=df[df['Request Type']=='Harmful']; ben=df[df['Request Type']=='Benign']; out={}
notool_h=harm[harm['Mode']=='No-tool chat']; notool_b=ben[ben['Mode']=='No-tool chat']
toolh=harm[harm['Mode']=='Tool-enabled normal']; toolb=ben[ben['Mode']=='Tool-enabled normal']

# ---- Ablation (n=120 harmful No-tool) ----
t0=time.time(); log('[abl] ablation n=120...')
va=notool_h.sample(120,random_state=7).reset_index(drop=True); pa=[format_prompt(r) for _,r in va.iterrows()]
base=[looks_refused(t) for t in gen_all(pa)]
abl=[looks_refused(t) for t in gen_all(pa, hook_fn=lambda i,n: ablate())]
out['ablation']={'n':120,'base_refuse':round(np.mean(base),3),'base_ci':ci(base),'ablated_refuse':round(np.mean(abl),3),'ablated_ci':ci(abl)}
log(f"[abl] refuse {np.mean(base):.0%} {out['ablation']['base_ci']} -> {np.mean(abl):.0%} {out['ablation']['ablated_ci']}  ({time.time()-t0:.0f}s)")

# ---- Addition (n=120 benign No-tool, coef 1245) ----
t0=time.time(); log('[add] addition n=120...')
vb=notool_b.sample(120,random_state=7).reset_index(drop=True); pb=[format_prompt(r) for _,r in vb.iterrows()]
bbase=[looks_refused(t) for t in gen_all(pb)]
badd=[looks_refused(t) for t in gen_all(pb, hook_fn=lambda i,n: addvec(1245.0))]
out['addition']={'n':120,'coef':1245.0,'base_refuse':round(np.mean(bbase),3),'base_ci':ci(bbase),'added_refuse':round(np.mean(badd),3),'added_ci':ci(badd)}
log(f"[add] refuse {np.mean(bbase):.0%} {out['addition']['base_ci']} -> {np.mean(badd):.0%} {out['addition']['added_ci']}  ({time.time()-t0:.0f}s)")

# ---- Steering dose-response (n=100 harmful, 60 benign) ----
t0=time.time(); log('[steer] dose-response n=100/60...')
sh=toolh.sample(100,random_state=5).reset_index(drop=True); shp=[format_prompt(r) for _,r in sh.iterrows()]
sb=toolb.sample(60,random_state=5).reset_index(drop=True); sbp=[format_prompt(r) for _,r in sb.iterrows()]
GRID=[0,200,450,700]; steer={'grid':GRID,'harmful_unsafe':[],'harmful_unsafe_ci':[],'benign_refuse':[],'benign_refuse_ci':[]}
for c in GRID:
    hf=(lambda i,n,cc=c: addvec(float(cc))) if c>0 else None
    uh=[unsafe_all(sh.iloc[k],t) for k,t in enumerate(gen_all(shp,hook_fn=hf))]
    rb=[looks_refused(t) for t in gen_all(sbp,hook_fn=hf)]
    steer['harmful_unsafe'].append(round(np.mean(uh),3)); steer['harmful_unsafe_ci'].append(ci(uh))
    steer['benign_refuse'].append(round(np.mean(rb),3)); steer['benign_refuse_ci'].append(ci(rb))
    log(f"   c={c:4d} unsafe={np.mean(uh):.0%} {steer['harmful_unsafe_ci'][-1]} | benign refuse={np.mean(rb):.0%} {steer['benign_refuse_ci'][-1]}")
out['steering']=steer; log(f'[steer] done {time.time()-t0:.0f}s')

# ---- Patching + AUC on 300-prompt baseline ----
t0=time.time(); log('[patch/auc] baseline n=300...')
ps=toolh.sample(300,random_state=4).reset_index(drop=True); psp=[format_prompt(r) for _,r in ps.iterrows()]
proj=proj_last(psp)
nt=harm[harm['Mode']=='No-tool chat'].drop_duplicates('group_id').set_index('group_id')
need=sorted(set(ps['group_id'])&set(nt.index)); ntp=dict(zip(need, proj_last([format_prompt(nt.loc[g]) for g in need])))
global_nt=float(np.mean(list(ntp.values()))) if ntp else 159.0
base_unsafe=np.array([unsafe_all(ps.iloc[k],t) for k,t in enumerate(gen_all(psp))])
us=proj[base_unsafe]; sa=proj[~base_unsafe]
a,aci=auc_ci(us,sa)
out['auc']={'n':300,'n_unsafe':int(base_unsafe.sum()),'mean_proj_unsafe':round(float(us.mean()),1) if len(us) else None,
            'mean_proj_safe':round(float(sa.mean()),1) if len(sa) else None,'auc':a,'auc_ci':aci}
log(f"[auc] unsafe={base_unsafe.sum()}/300 AUC={a} {aci}")
idx=np.where(base_unsafe)[0]
targets=[ntp.get(ps.iloc[k]['group_id'],global_nt) for k in idx]
puprompts=[psp[k] for k in idx]
# patch in batches with per-batch target slices
flips=[]
for bi in range(0,len(idx),BS):
    chunk=puprompts[bi:bi+BS]; tch=targets[bi:bi+BS]
    texts=gen_batch(chunk, hook=set_proj_batch(tch))
    for j,t in enumerate(texts):
        k=idx[bi+j]; flips.append(not unsafe_all(ps.iloc[k],t))
out['patching']={'n_unsafe_baseline':int(len(idx)),'flipped_to_safe':int(np.sum(flips)),
                 'rate':round(float(np.mean(flips)),3) if len(flips) else None,'rate_ci':ci(flips)}
log(f"[patch] of {len(idx)} unsafe, flipped {int(np.sum(flips))} ({np.mean(flips) if len(flips) else 0:.0%}) CI={out['patching']['rate_ci']}  ({time.time()-t0:.0f}s)")

# figures
plt.figure(figsize=(6,4))
g=steer['grid']
plt.errorbar(g,[100*x for x in steer['harmful_unsafe']],yerr=[[100*(steer['harmful_unsafe'][i]-steer['harmful_unsafe_ci'][i][0]) for i in range(len(g))],[100*(steer['harmful_unsafe_ci'][i][1]-steer['harmful_unsafe'][i]) for i in range(len(g))]],fmt='o-',color='#d7301f',label='harmful: unsafe %',capsize=3)
plt.errorbar(g,[100*x for x in steer['benign_refuse']],yerr=[[100*(steer['benign_refuse'][i]-steer['benign_refuse_ci'][i][0]) for i in range(len(g))],[100*(steer['benign_refuse_ci'][i][1]-steer['benign_refuse'][i]) for i in range(len(g))]],fmt='s-',color='#2c7fb8',label='benign: over-refusal %',capsize=3)
plt.xlabel('steering coefficient c'); plt.ylabel('%'); plt.title('Steering dose-response (n=100/60, 95% CI)'); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(ART/'fig_steering_scaled.png',dpi=110); plt.close()
if len(us) and len(sa):
    plt.figure(figsize=(6,4))
    plt.hist(sa,bins=25,alpha=.6,label=f'safe (n={len(sa)})',color='#2c7fb8',density=True)
    plt.hist(us,bins=20,alpha=.6,label=f'unsafe (n={len(us)})',color='#d7301f',density=True)
    plt.xlabel(f'projection onto refusal dir (L{LAYER})'); plt.ylabel('density'); plt.title(f'Projection vs unsafe (AUC={a}, n=300)')
    plt.legend(); plt.tight_layout(); plt.savefig(ART/'fig_auc_scaled.png',dpi=110); plt.close()
(ART/'interp5_summary.json').write_text(json.dumps(out,indent=2))
log('[done] wrote interp5_summary.json'); log('[done] SUMMARY:',json.dumps(out))
