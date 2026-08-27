import json, sys, collections, pandas as pd
sys.path.insert(0, '.')
from tools.registry import get_domain

DM = {"Finance": "finance", "Healthcare": "healthcare",
      "Education / Academic Integrity": "education", "Legal / Compliance": "legal"}

# real tool names per domain
real = {}
for dom in set(DM.values()):
    t = list(get_domain(dom).tools_for_llm)
    names = set()
    for x in t:
        if isinstance(x, dict):
            names.add(x.get("name") or x.get("function", {}).get("name", ""))
    real[dom] = names
allreal = set().union(*real.values())
print("real tool names:", sorted(allreal))
print()

for m in ("Qwen3-14B", "Meta-Llama-3.1-70B-Instruct", "gemma-3-27b-it", "c4ai-command-r7b-12-2024"):
    d = pd.read_csv(f"results/results_{m}.csv", low_memory=False)
    d = d[d["mode"] != "No-tool chat"]
    names = collections.Counter()
    nrows = 0
    for _, r in d.iterrows():
        try:
            calls = json.loads(r["tool_calls"]) if isinstance(r["tool_calls"], str) else []
        except Exception:
            calls = []
        if calls:
            nrows += 1
        for c in calls:
            names[c.get("name", "")] += 1
    known = sum(v for k, v in names.items() if k in allreal)
    tot = sum(names.values())
    print(f"{m:<32} rows_with_calls={nrows:<5} calls={tot:<5} "
          f"real_names={known/tot:.0%}" if tot else f"{m:<32} no calls")
    print(f"    top names: {names.most_common(5)}")
