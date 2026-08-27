#!/usr/bin/env python3
"""CPU half of the ablation / addition / steering rerun: label the completions
dumped by rerun_steering_gen.py with the shared three-way judge and report the
old (regex) vs new (judge) numbers side by side.

Runs where the Anthropic key lives, so the key never goes to the GPU host.

  JUDGE_KEY=... python judge_generations.py --model Qwen3-14B
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

API_URL = "https://api.anthropic.com/v1/messages"


class UrllibAnthropic:
    """Minimal stand-in for anthropic.Anthropic exposing just .messages.create.

    The installed SDK rides on httpx, which cannot open a connection from this
    sandbox (urllib to the same endpoint works), so the judge gets this instead.
    HTTP error bodies are folded into the exception text because
    make_anthropic_judge inspects the message for 'temperature'.
    """

    class _Block:
        def __init__(self, d):
            self.type, self.text = d.get("type"), d.get("text", "")

    class _Resp:
        def __init__(self, d):
            self.content = [UrllibAnthropic._Block(b) for b in d.get("content", [])]

    class _Messages:
        def __init__(self, key, timeout):
            self.key, self.timeout = key, timeout

        def create(self, model, max_tokens, messages, **opts):
            body = {"model": model, "max_tokens": max_tokens, "messages": messages, **opts}
            req = urllib.request.Request(
                API_URL, data=json.dumps(body).encode(),
                headers={"x-api-key": self.key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return UrllibAnthropic._Resp(json.loads(r.read()))
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:500].decode('utf-8', 'replace')}") from e

    def __init__(self, api_key, timeout=60):
        self.messages = self._Messages(api_key, timeout)

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from tools.refusal import (MIN_TEXT_CHARS, make_anthropic_judge,  # noqa: E402
                           regex_refused, strip_tool_markup)
from rerun_against_relabels import OUT  # noqa: E402


def boot_ci(vals, n=2000, seed=0):
    v = np.asarray(vals, float)
    if not len(v):
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    bs = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(n)]
    return (round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    key = os.environ.get("JUDGE_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:  # fall back to the repo .env, which is where the key normally lives
        for base in [REPO, *REPO.parents[:4]]:
            f = base / ".env"
            if f.exists():
                for line in f.read_text().splitlines():
                    if line.startswith("ANTHROPIC_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip("\"'")
                if key:
                    print(f"[key] read from {f}")
                    break
    if not key:
        sys.exit("set JUDGE_KEY (or ANTHROPIC_API_KEY) -- refusing to fall back to regex, "
                 "which is the classifier this rerun exists to replace")
    judge = make_anthropic_judge(client=UrllibAnthropic(key))

    raw = json.loads((OUT / f"steer_raw_{a.model}.json").read_text())
    pending, meta = raw["pending"], raw["meta"]
    print(f"[load] {len(pending)} completions from steer_raw_{a.model}.json")

    # judge each unique (prompt, cleaned response) once
    cache, order = {}, []
    for p in pending:
        p["clean"] = strip_tool_markup(p["text"] or "")
        p["key"] = (p["user_prompt"], p["clean"])
        if p["key"] not in cache:
            cache[p["key"]] = None
            order.append(p["key"])
    todo = [k for k in order if len(k[1]) >= MIN_TEXT_CHARS]
    print(f"[judge] {len(todo)} unique pairs (of {len(order)}) need the judge")

    done = [0]

    def run(k):
        up, txt = k
        try:
            v = judge(txt, up)
        except Exception as e:                      # noqa: BLE001
            v = f"error:{type(e).__name__}"
        done[0] += 1
        if done[0] % 50 == 0:
            print(f"[judge] {done[0]}/{len(todo)}", flush=True)
        return k, v

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for k, v in ex.map(run, todo):
            cache[k] = v

    errs = sum(1 for v in cache.values() if isinstance(v, str) and v.startswith("error:"))
    if errs:
        print(f"[judge] WARNING {errs} judge errors; those rows are dropped")

    # old label = the regex classifier the published numbers used
    for p in pending:
        v = cache.get(p["key"])
        p["new_refused"] = (v == "refuse")
        p["new_label"] = v
        p["old_refused"] = regex_refused(p["clean"]) if len(p["clean"]) >= MIN_TEXT_CHARS else False
        p["usable"] = not (isinstance(v, str) and v.startswith("error:")) and v is not None

    groups = defaultdict(list)
    for p in pending:
        if p["usable"]:
            groups[(p["exp"], p["coef"])].append(p)

    res = {"model": a.model, "layer": meta["layer"], "add_coef": meta["add_coef"],
           "n_judged": len(todo), "judge_errors": errs}

    def rate(exp, coef=None):
        g = groups.get((exp, coef), [])
        if not g:
            return None
        old = [x["old_refused"] for x in g]
        new = [x["new_refused"] for x in g]
        return {"n": len(g),
                "old": round(float(np.mean(old)), 3), "old_ci": boot_ci(old),
                "new": round(float(np.mean(new)), 3), "new_ci": boot_ci(new),
                "labels": {k: sum(1 for x in g if x["new_label"] == k)
                           for k in ("refuse", "caveat", "comply")}}

    for name in ("ablation_base", "ablation_ablated", "addition_base"):
        res[name] = rate(name)
    res["addition_added"] = rate("addition_added", meta["add_coef"])

    grid = meta["steering"]["grid"]
    res["steering"] = {"grid": grid,
                       "harmful_unsafe": meta["steering"]["harmful_unsafe"],
                       "benign_refuse_old": [], "benign_refuse_new": [],
                       "benign_refuse_new_ci": []}
    for c in grid:
        r = rate("steer_benign", c)
        res["steering"]["benign_refuse_old"].append(r["old"] if r else None)
        res["steering"]["benign_refuse_new"].append(r["new"] if r else None)
        res["steering"]["benign_refuse_new_ci"].append(r["new_ci"] if r else None)

    (OUT / f"steer_{a.model}.json").write_text(json.dumps(res, indent=2))

    print(f"\n== {a.model} (layer {meta['layer']}) ==")
    print(f"{'experiment':<20}{'old (regex)':>14}{'new (judge)':>14}   three-way")
    for name in ("ablation_base", "ablation_ablated", "addition_base", "addition_added"):
        r = res[name]
        if r:
            print(f"{name:<20}{r['old']:>14.3f}{r['new']:>14.3f}   {r['labels']}")
    print("\nsteering dose-response:")
    print(f"{'coef':>6}{'harmful unsafe':>16}{'benign refuse old':>20}{'benign refuse new':>20}")
    s = res["steering"]
    for i, c in enumerate(grid):
        print(f"{c:>6}{s['harmful_unsafe'][i]:>16.3f}"
              f"{s['benign_refuse_old'][i]:>20.3f}{s['benign_refuse_new'][i]:>20.3f}")
    print(f"\nwrote {OUT / f'steer_{a.model}.json'}")


if __name__ == "__main__":
    main()
