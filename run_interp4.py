#!/usr/bin/env python3
"""Tier-1 follow-ups on the FIXED scorer (score_tool_calls_all):
  #2 same-direction: re-extract refusal dir WITHIN tool mode; cosine vs text-extracted r.
  #3 patching (non-floor): restore matched No-tool refusal level in tool runs that are
     unsafe-at-baseline; does the unsafe call go away?
  #4 AUC: does the projection onto r predict (per-example) whether the call is unsafe?
Reuses refusal_dirs.pt (r_text @ LAYER 33)."""
import os, sys, json, re, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.environ.setdefault('HF_HOME','/workspace/.cache/huggingface')
REPO=Path(__file__).resolve().parent; sys.path.insert(0,str(REPO))
from tools.registry import get_domain
MODEL_ID,DTYPE,DEVICE='Qwen/Qwen3-14B',torch.bfloat16,'cuda'; ENABLE_THINKING=False
ART=REPO/'interp_artifacts'; torch.set_grad_enabled(False); LAYER=33
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
def _sid(dk,lab,rt):
    if lab=='Direct request': return f'{dk}_direct_harmful' if rt=='Harmful' else f'{dk}_direct_safe'
    if lab=='Indirect phrasing': return f'{dk}_indirect'
    if lab=='Justification / roleplay': return f'{dk}_roleplay'
    if lab=='Structured tool-oriented': return f'{dk}_structured_tool'
    return f'{dk}_direct_safe'
def tools_for_row(row): return None if row['Mode']=='No-tool chat' else list(get_domain(row['domain_key']).tools_for_llm)
def format_prompt(row):
    msgs=[{'role':'system','content':row['System Prompt']},{'role':'user','content':row['User Prompt']}]
    t=tools_for_row(row); base=dict(tokenize=False,add_generation_prompt=True)
    for extra in ({'enable_thinking':ENABLE_THINKING},{}):
        try: return tok.apply_chat_template(msgs,tools=t,**base,**extra) if t else tok.apply_chat_template(msgs,**base,**extra)
        except TypeError: continue
def parse_tool_calls(text):
    calls=[]
    m=re.search(r'\[TOOL_CALLS\]\s*(\[.*?\])',text,re.DOTALL)
    if m:
        try:
            p=json.loads(m.group(1))
            if isinstance(p,list): return [{'name':c.get('name',''),'arguments':c.get('arguments',{})} for c in p]
        except json.JSONDecodeError: pass
    for raw in re.findall(r'<tool_call>(.*?)</tool_call>',text,re.DOTALL):
        try:
            c=json.loads(raw.strip()); calls.append({'name':c.get('name',''),'arguments':c.get('arguments',c.get('parameters',{}))})
        except json.JSONDecodeError: pass
    return calls
def unsafe_all(row,text):
    dom=get_domain(row['domain_key']); return len(dom.score_tool_calls_all(parse_tool_calls(text)))>0
def _layers(): return model.model.layers
class Hooks:
    def __init__(self): self.h=[]
    def remove(self):
        for x in self.h: x.remove()
        self.h=[]
    def __enter__(self): return self
    def __exit__(self,*a): self.remove()
def set_proj(target):
    def hook(m,i,o):
        h=o[0] if isinstance(o,tuple) else o; cur=(h@r_dev).unsqueeze(-1); h=h+(target-cur)*r_dev
        return (h,)+tuple(o[1:]) if isinstance(o,tuple) else h
    hk=Hooks(); hk.h.append(_layers()[LAYER-1].register_forward_hook(hook)); return hk
@torch.no_grad()
def generate(p,max_new_tokens=200):
    enc=tok(p,return_tensors='pt',truncation=True,max_length=2048).to(DEVICE)
    out=model.generate(**enc,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc['input_ids'].shape[1]:],skip_special_tokens=False)
@torch.no_grad()
def proj_last(prompts,batch=8,layer=LAYER,direction=None):
    dd=r_text if direction is None else direction
    out=[]
    for i in range(0,len(prompts),batch):
        enc=tok(prompts[i:i+batch],return_tensors='pt',padding=True,truncation=True,max_length=2048).to(DEVICE)
        hs=model(**enc,output_hidden_states=True).hidden_states
        out.append((hs[layer][:,-1,:].float().cpu()@dd).numpy())
    return np.concatenate(out)
@torch.no_grad()
def resid_last(prompts,batch=8,layer=LAYER):
    out=[]
    for i in range(0,len(prompts),batch):
        enc=tok(prompts[i:i+batch],return_tensors='pt',padding=True,truncation=True,max_length=2048).to(DEVICE)
        hs=model(**enc,output_hidden_states=True).hidden_states
        out.append(hs[layer][:,-1,:].float().cpu())
    return torch.cat(out)
harm=df[df['Request Type']=='Harmful']; ben=df[df['Request Type']=='Benign']; out={}

# ---- #2 same direction under tools? ----
t0=time.time(); log('[#2] re-extract direction within tool mode; cosine vs text...')
def dir_from(sub_h,sub_b):
    A=resid_last([format_prompt(r) for _,r in sub_h.iterrows()]); B=resid_last([format_prompt(r) for _,r in sub_b.iterrows()])
    d=A.mean(0)-B.mean(0); return d/d.norm()
th=harm[harm['Mode']=='Tool-enabled normal'].sample(96,random_state=0); tb=ben[ben['Mode']=='Tool-enabled normal'].sample(96,random_state=0)
r_tool=dir_from(th,tb)
cos=float(torch.dot(r_text/r_text.norm(), r_tool/r_tool.norm()))
out['same_direction']={'cosine_text_vs_tool_dir_layer33':cos}
log(f'[#2] cosine(r_text, r_tool) @L{LAYER} = {cos:.3f}  ({time.time()-t0:.0f}s)')

# ---- #3 + #4 combined: generate harmful tool prompts once ----
t0=time.time(); log('[#3/#4] generating harmful tool prompts (baseline)...')
N=120
sample=harm[harm['Mode']=='Tool-enabled normal'].sample(N,random_state=4).reset_index(drop=True)
prompts=[format_prompt(r) for _,r in sample.iterrows()]
proj=proj_last(prompts)                       # projection onto r_text (for AUC)
# matched No-tool target per group
nt=harm[harm['Mode']=='No-tool chat'].drop_duplicates('group_id').set_index('group_id')
nt_proj={g:float(proj_last([format_prompt(nt.loc[g])])[0]) for g in set(sample['group_id'])&set(nt.index)}
global_nt=float(np.mean(list(nt_proj.values()))) if nt_proj else 159.0
base_unsafe=[]; texts=[]
for k in range(N):
    t=generate(prompts[k]); base_unsafe.append(unsafe_all(sample.iloc[k],t)); texts.append(t)
base_unsafe=np.array(base_unsafe)
# #4 AUC: lower projection should predict unsafe
us=proj[base_unsafe]; sa=proj[~base_unsafe]
auc=float(np.mean([ (s> u) for s in sa for u in us ])) if len(us) and len(sa) else float('nan')
out['auc']={'n':N,'n_unsafe':int(base_unsafe.sum()),'mean_proj_unsafe':float(us.mean()) if len(us) else None,
            'mean_proj_safe':float(sa.mean()) if len(sa) else None,'auc_lowproj_predicts_unsafe':auc}
log(f'[#4] base unsafe={base_unsafe.sum()}/{N} | proj unsafe={us.mean() if len(us) else float('nan'):.1f} vs safe={sa.mean() if len(sa) else float('nan'):.1f} | AUC={auc:.3f}  ({time.time()-t0:.0f}s)')

# #3 patching: on unsafe-at-baseline cases, restore matched No-tool projection
t0=time.time(); log('[#3] patching the unsafe-at-baseline cases...')
idx=np.where(base_unsafe)[0]
flipped=0
for k in idx:
    g=sample.iloc[k]['group_id']; target=nt_proj.get(g, global_nt)
    with set_proj(float(target)): t=generate(prompts[k])
    if not unsafe_all(sample.iloc[k],t): flipped+=1
out['patching_fixed']={'n_unsafe_baseline':int(len(idx)),'flipped_to_safe':int(flipped),
                       'rate':float(flipped/len(idx)) if len(idx) else None,'target':'matched No-tool proj'}
log(f'[#3] of {len(idx)} unsafe-at-baseline, patched->safe: {flipped} ({flipped/max(len(idx),1):.0%})  ({time.time()-t0:.0f}s)')

# AUC figure (proj distributions)
if len(us) and len(sa):
    plt.figure(figsize=(6,4))
    plt.hist(sa,bins=20,alpha=.6,label=f'safe (n={len(sa)})',color='#2c7fb8',density=True)
    plt.hist(us,bins=10,alpha=.6,label=f'unsafe (n={len(us)})',color='#d7301f',density=True)
    plt.xlabel(f'projection onto refusal direction (layer {LAYER})'); plt.ylabel('density')
    plt.title(f'Projection vs unsafe tool call (AUC={auc:.2f})'); plt.legend(); plt.tight_layout()
    plt.savefig(ART/'fig_auc.png',dpi=110); plt.close()
(ART/'interp4_summary.json').write_text(json.dumps(out,indent=2))
log('[done] wrote interp4_summary.json'); log('[done] SUMMARY:',json.dumps(out))
