"""Base -> ablated label transitions on paired prompts, from the persisted judge cache."""
import hashlib, json, sys
from collections import Counter
sys.path.insert(0, '.')
from tools.refusal import strip_tool_markup
from degeneracy import classify_v2

cache = json.load(open("relabel_analysis/judge_cache_local.json"))


def label(r):
    c = strip_tool_markup(r["text"] or "")
    if classify_v2(c) == "degenerate":
        return "degen"
    if len(c) < 20:
        return "no_text"
    k = hashlib.sha1(f"{r['user_prompt']}\x00{c}".encode()).hexdigest()
    v = cache.get(k)
    return v if v in ("refuse", "caveat", "comply") else "err"


for m, conds in [("Qwen3-14B", ["ablate_rtext", "ablate_meanout"]),
                 ("gemma-3-27b-it", ["ablate_meanout"]),
                 ("c4ai-command-r7b-12-2024", ["ablate_rtext"]),
                 ("Mistral-7B-Instruct-v0.3", ["ablate_rtext"])]:
    raw = json.load(open(f"relabel_analysis/controls_raw_{m}.json"))
    base = {r["id"]: label(r) for r in raw["pending"] if r["exp"] == "ablation_base"}
    for cond in conds:
        abl = {r["id"]: label(r) for r in raw["pending"] if r["exp"] == cond}
        t = Counter((base[i], abl[i]) for i in base if i in abl)
        print(f"\n== {m}  base -> {cond}  (n={sum(t.values())})")
        print(f"{'':10}" + "".join(f"{c:>9}" for c in ("refuse", "caveat", "comply", "degen", "err")))
        for b in ("refuse", "caveat", "comply"):
            print(f"{b:<10}" + "".join(f"{t[(b, a)]:>9}" for a in ("refuse", "caveat", "comply", "degen", "err")))
        up = sum(v for (b, a), v in t.items() if b != "refuse" and a == "refuse")
        down = sum(v for (b, a), v in t.items() if b == "refuse" and a in ("caveat", "comply"))
        print(f"  became refuse: {up}   stopped refusing: {down}")
