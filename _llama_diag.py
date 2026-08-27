import json, sys, collections
sys.path.insert(0, '.')
from rerun_steering_gen import parse_tool_calls
from tools.registry import get_domain

DM = {"Finance": "finance", "Healthcare": "healthcare",
      "Education / Academic Integrity": "education", "Legal / Compliance": "legal"}

d = json.load(open('relabel_analysis/steer_raw_Meta-Llama-3.1-70B-Instruct.json'))
gens = [g for g in d.get('harmful_gens', []) if g['coef'] == 0]
print(f"harmful gens at c=0: {len(gens)}")

ncalls = collections.Counter()
empty = 0
for g in gens:
    calls = parse_tool_calls(g['text'] or '')
    ncalls[len(calls)] += 1
    if not (g['text'] or '').strip():
        empty += 1
print(f"tool calls parsed per response: {dict(ncalls)}")
print(f"empty responses: {empty}")

# how many calls does the scorer flag?
flagged = 0
for g in gens:
    calls = parse_tool_calls(g['text'] or '')
    if calls:
        v = get_domain(DM[g['domain']]).score_tool_calls_all(calls)
        if v:
            flagged += 1
print(f"responses with >=1 flagged unsafe call: {flagged}")

print("\n--- first 3 raw generations ---")
for g in gens[:3]:
    print(f"  [{g['id']}] {str(g['text'])[:400]!r}\n")
