#!/usr/bin/env python3
"""Re-score a behavioral results CSV under all-actions scoring (fixes the scenario-scoping
under-count) and quantify the gap vs the original per-scenario tc_safe. CPU only."""
import sys, json
import pandas as pd
sys.path.insert(0,'.')
from tools.registry import get_domain
DOMAIN_MAP={'Finance':'finance','Healthcare':'healthcare','Education / Academic Integrity':'education','Legal / Compliance':'legal'}
path=sys.argv[1] if len(sys.argv)>1 else 'results/results_Qwen3-14B.csv'
d=pd.read_csv(path)
def calls(x):
    try: return json.loads(x) if isinstance(x,str) else []
    except Exception: return []
def unsafe_scoped(row):
    if row['mode']=='No-tool chat': return None
    dk=DOMAIN_MAP[row['domain']]; dom=get_domain(dk)
    return len(dom.score_tool_calls(row['scenario_id'], calls(row['tool_calls'])))>0
def unsafe_all(row):
    if row['mode']=='No-tool chat': return None
    dk=DOMAIN_MAP[row['domain']]; dom=get_domain(dk)
    return len(dom.score_tool_calls_all(calls(row['tool_calls'])))>0
d['unsafe_scoped']=d.apply(unsafe_scoped,axis=1)
d['unsafe_all']=d.apply(unsafe_all,axis=1)
tool=d[d['mode']!='No-tool chat']
h=tool[tool['request_type']=='Harmful']; b=tool[tool['request_type']=='Benign']
print(f'== {path}  ({len(d)} rows; {len(tool)} tool-mode) ==')
print(f'HARMFUL tool rows: n={len(h)}')
print(f'  unsafe rate  per-scenario (old): {h["unsafe_scoped"].mean():.1%}')
print(f'  unsafe rate  all-actions  (new): {h["unsafe_all"].mean():.1%}')
newly=h[(h["unsafe_all"]==True)&(h["unsafe_scoped"]==False)]
print(f'  newly caught by fix: {len(newly)}  ({len(newly)/max(len(h),1):.1%} of harmful tool rows)')
if len(h):
    print('  by domain (old -> new unsafe rate):')
    for dom,g in h.groupby('domain'):
        print(f'    {dom:32s} {g["unsafe_scoped"].mean():.0%} -> {g["unsafe_all"].mean():.0%}  (n={len(g)})')
print(f'BENIGN tool rows: n={len(b)}  newly-flagged (possible false positives): {((b["unsafe_all"]==True)&(b["unsafe_scoped"]==False)).sum()}')
d.to_csv(path.replace('.csv','_rescored.csv'),index=False)
print('wrote', path.replace('.csv','_rescored.csv'))
