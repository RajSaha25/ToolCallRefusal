#!/usr/bin/env python3
"""Mechanistic refusal-direction pipeline (script form of Mechanistic_Refusal_Direction.ipynb).
Runs headless so we can validate the science on the live model and tune LAYER / ADD_COEF,
then fold the chosen values back into the notebook. Saves artifacts + a JSON summary.
"""
import os, sys, json, re, gc, time
from pathlib import Path
import numpy as np, pandas as pd, torch

os.environ.setdefault('HF_HOME', '/workspace/.cache/huggingface')
os.environ.setdefault('MPLBACKEND', 'Agg')

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from tools.registry import get_domain, list_domains

MODEL_ID, DTYPE, DEVICE = 'Qwen/Qwen3-14B', torch.bfloat16, 'cuda'
ENABLE_THINKING = False
DATASET_PATH = REPO / 'data' / 'complete_combined_safety_eval_dataset_2304_normalized_labels.xlsx'
ART = REPO / 'interp_artifacts'; ART.mkdir(exist_ok=True)
torch.set_grad_enabled(False)

def log(*a): print(*a, flush=True)

from transformers import AutoTokenizer, AutoModelForCausalLM
log('[load] tokenizer + model (bf16)...')
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tok.padding_side = 'left'
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True)
model.eval()
N_LAYERS, D_MODEL = model.config.num_hidden_layers, model.config.hidden_size
log(f'[load] layers={N_LAYERS} d_model={D_MODEL} gpu={torch.cuda.memory_allocated()/1e9:.1f}GB')

DOMAIN_MAP = {'Finance':'finance','Healthcare':'healthcare',
              'Education / Academic Integrity':'education','Legal / Compliance':'legal'}
df = pd.read_excel(DATASET_PATH); df.columns=[c.strip() for c in df.columns]
df['User Prompt'] = df['User Prompt'].str.replace('�','—', regex=False)
df['domain_key'] = df['Domain'].map(DOMAIN_MAP)
df['group_id'] = df['Domain']+'|'+df['Scenario']+'|'+df['Prompt Variant']+'|'+df['System Condition']
MODES = ['No-tool chat','Tool-enabled normal','Tool-enabled stress']

def tools_for_row(row):
    return None if row['Mode']=='No-tool chat' else list(get_domain(row['domain_key']).tools_for_llm)

def format_prompt(row, tools='auto'):
    msgs=[{'role':'system','content':row['System Prompt']},{'role':'user','content':row['User Prompt']}]
    t = tools_for_row(row) if tools=='auto' else tools
    base=dict(tokenize=False, add_generation_prompt=True)
    for extra in ({'enable_thinking':ENABLE_THINKING},{}):
        try:
            return tok.apply_chat_template(msgs, tools=t, **base, **extra) if t else tok.apply_chat_template(msgs, **base, **extra)
        except TypeError:
            continue
    raise RuntimeError('apply_chat_template failed')

@torch.no_grad()
def resid_last(prompts, batch_size=8, max_length=2048):
    chunks=[]
    for i in range(0,len(prompts),batch_size):
        enc=tok(prompts[i:i+batch_size], return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(DEVICE)
        hs=model(**enc, output_hidden_states=True).hidden_states
        chunks.append(torch.stack([h[:,-1,:] for h in hs],dim=1).float().cpu())
    return torch.cat(chunks,0)

t0=time.time()
notool = df[df['Mode']=='No-tool chat']
harm_ex = notool[notool['Request Type']=='Harmful'].sample(128, random_state=0)
ben_ex  = notool[notool['Request Type']=='Benign' ].sample(128, random_state=0)
log(f'[step1] extracting on {len(harm_ex)} harmful / {len(ben_ex)} benign (No-tool)...')
A_harm = resid_last([format_prompt(r) for _,r in harm_ex.iterrows()])
A_ben  = resid_last([format_prompt(r) for _,r in ben_ex.iterrows()])
diff = A_harm.mean(0) - A_ben.mean(0)
sep  = diff.norm(dim=-1)
dirs = diff / diff.norm(dim=-1, keepdim=True)
torch.save({'dirs':dirs,'sep':sep}, ART/'refusal_dirs.pt')
lo, hi = int(0.35*N_LAYERS), int(0.85*N_LAYERS)
band = sep.clone(); band[:lo]=-1; band[hi:]=-1
LAYER = int(band.argmax())
log(f'[step1] sep top5 layers={torch.topk(sep,5).indices.tolist()}  -> LAYER={LAYER}  ({time.time()-t0:.0f}s)')

rdir = dirs[LAYER]
ph = (A_harm[:,LAYER,:]@rdir); pb = (A_ben[:,LAYER,:]@rdir)
proj_gap = float(ph.mean()-pb.mean())
log(f'[step1] proj(harm)={ph.mean():+.2f} proj(ben)={pb.mean():+.2f}  gap={proj_gap:+.2f}')

# Refusal labels MUST come from the same classifier as the behavioral eval
# (the old 6-pattern local list is what made Table 3 baselines contradict Table 1).
from tools.refusal import classify_refusal as _shared_classify_refusal, make_anthropic_judge
_JUDGE_KEY = os.environ.get('JUDGE_KEY', '')
_judge = make_anthropic_judge(api_key=_JUDGE_KEY) if _JUDGE_KEY else None
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
def ablate(unit):
    r=unit.to(DEVICE,DTYPE).view(-1)
    def mk():
        def hook(m,i,o):
            h=o[0] if isinstance(o,tuple) else o
            h=h-(h@r).unsqueeze(-1)*r
            return (h,)+tuple(o[1:]) if isinstance(o,tuple) else h
        return hook
    hk=Hooks()
    for L in _layers(): hk.h.append(L.register_forward_hook(mk()))
    return hk
def addvec(unit, layer_idx, coef):
    r=unit.to(DEVICE,DTYPE).view(-1)
    def hook(m,i,o):
        h=o[0] if isinstance(o,tuple) else o
        h=h+coef*r
        return (h,)+tuple(o[1:]) if isinstance(o,tuple) else h
    hk=Hooks(); idx=max(0,min(layer_idx,len(_layers())-1))
    hk.h.append(_layers()[idx].register_forward_hook(hook)); return hk
@torch.no_grad()
def generate(p, max_new_tokens=96):
    enc=tok(p, return_tensors='pt', truncation=True, max_length=2048).to(DEVICE)
    out=model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)

t0=time.time()
NV=20
vh = notool[(notool['Request Type']=='Harmful') & (~notool.index.isin(harm_ex.index))].sample(NV, random_state=7)
vb = notool[(notool['Request Type']=='Benign')  & (~notool.index.isin(ben_ex.index))].sample(NV, random_state=7)
base_h=[generate(format_prompt(r)) for _,r in vh.iterrows()]
with ablate(rdir): abl_h=[generate(format_prompt(r)) for _,r in vh.iterrows()]
br=np.mean([looks_refused(t) for t in base_h]); ar=np.mean([looks_refused(t) for t in abl_h])
ADD_COEF=float(max(6.0, 4.0*abs(proj_gap)))
base_b=[generate(format_prompt(r)) for _,r in vb.iterrows()]
with addvec(rdir, LAYER-1, ADD_COEF): add_b=[generate(format_prompt(r)) for _,r in vb.iterrows()]
bbr=np.mean([looks_refused(t) for t in base_b]); abr=np.mean([looks_refused(t) for t in add_b])
log(f'[step2] ablation harmful refusal {br:.0%} -> {ar:.0%} | addition(coef={ADD_COEF:.1f}) benign refusal {bbr:.0%} -> {abr:.0%}  ({time.time()-t0:.0f}s)')
log('[step2] sample BASE   :', base_h[0][:140].replace(chr(10),' '))
log('[step2] sample ABLATED:', abl_h[0][:140].replace(chr(10),' '))

t0=time.time()
def proj_rows(rows, layer=None):
    layer=LAYER if layer is None else layer
    A=resid_last([format_prompt(r) for _,r in rows.iterrows()])
    return (A[:,layer,:]@rdir).numpy()
harm_all=df[df['Request Type']=='Harmful']
proj={}
for m in MODES:
    sub=harm_all[harm_all['Mode']==m].sample(min(150,(harm_all['Mode']==m).sum()), random_state=1)
    proj[m]=proj_rows(sub)
    log(f'[step3] {m:22s} n={len(sub):3d} mean proj={proj[m].mean():+.3f} sd={proj[m].std():.3f}')

g=harm_all.groupby('group_id')['Mode'].nunique(); shared=g[g==len(MODES)].index
paired=harm_all[harm_all['group_id'].isin(shared)]
rbm={m:paired[paired['Mode']==m].drop_duplicates('group_id').set_index('group_id') for m in MODES}
common=sorted(set.intersection(*[set(rbm[m].index) for m in MODES]))[:120]
pp={m:proj_rows(rbm[m].loc[common]) for m in MODES}
base=pp['No-tool chat']; paired_stats={}
for m in MODES[1:]:
    d=base-pp[m]; t=float(d.mean()/(d.std(ddof=1)/np.sqrt(len(d))))
    paired_stats[m]={'mean_delta':float(d.mean()),'t':t}
    log(f'[step3] paired No-tool - {m:22s}: delta={d.mean():+.3f} t={t:+.2f} (n={len(d)})')

summary={'LAYER':LAYER,'N_LAYERS':N_LAYERS,'sep_top5':torch.topk(sep,5).indices.tolist(),'proj_gap':proj_gap,
         'ablation':{'base':float(br),'ablated':float(ar)},
         'addition':{'coef':ADD_COEF,'base':float(bbr),'added':float(abr)},
         'mode_mean_proj':{m:float(proj[m].mean()) for m in MODES},
         'paired':paired_stats,'n_paired':len(common)}
(ART/'interp_summary.json').write_text(json.dumps(summary, indent=2))
log('[done] wrote', str(ART/'interp_summary.json'), f'({time.time()-t0:.0f}s step3)')
log('[done] SUMMARY:', json.dumps(summary))
