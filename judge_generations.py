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
import socket
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

API_URL = "https://api.anthropic.com/v1/messages"


def force_ipv4():
    """IPv6 egress is black-holed here: every connection spends ~30s failing over
    to IPv4, which turns a 0.6s judge call into 75s. Resolve A records only."""
    _orig = socket.getaddrinfo

    def v4(host, port, family=0, type=0, proto=0, flags=0):
        return _orig(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = v4


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
                # Connection: close, and read exactly Content-Length bytes. Without
                # both, read() sits on the kept-alive socket until the timeout
                # expires -- correct verdict, 60s a call.
                headers={"x-api-key": self.key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json", "connection": "close"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    n = r.headers.get("Content-Length")
                    raw = r.read(int(n)) if n else r.read()
                    return UrllibAnthropic._Resp(json.loads(raw))
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:500].decode('utf-8', 'replace')}") from e

    def __init__(self, api_key, timeout=25):
        # Short timeout on purpose. A call that hangs blocks its worker, and with
        # the judge's own retry loop on top a single stuck request can stall the
        # whole pool for minutes. Failing fast and retrying is much cheaper.
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
    ap.add_argument("--raw-file", default=None,
                    help="judge a different generations file in the steer_raw format "
                         "(e.g. controls_raw_<model>.json) and write generic per-condition "
                         "rates to <name minus raw_>_judged.json")
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
    force_ipv4()
    judge = make_anthropic_judge(client=UrllibAnthropic(key))

    raw_name = a.raw_file or f"steer_raw_{a.model}.json"
    raw = json.loads((OUT / raw_name).read_text())
    pending, meta = raw["pending"], raw["meta"]
    print(f"[load] {len(pending)} completions from {raw_name}")

    # judge each unique (prompt, cleaned response) once
    cache, order = {}, []
    for p in pending:
        p["clean"] = strip_tool_markup(p["text"] or "")
        p["key"] = (p["user_prompt"], p["clean"])
        if p["key"] not in cache:
            cache[p["key"]] = None
            order.append(p["key"])
    # Persistent cache: verdicts are deterministic (temperature 0), so a rerun
    # should never re-pay for a pair already judged. Restarts are common enough
    # here -- a stalled pool, an interrupted session -- that losing 1000 calls to
    # one is worth avoiding.
    import hashlib
    cache_path = OUT / "judge_cache_local.json"
    disk = {}
    if cache_path.exists():
        try:
            disk = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            disk = {}

    def ckey(k):
        return hashlib.sha1(f"{k[0]}\x00{k[1]}".encode()).hexdigest()

    hits = 0
    for k in order:
        v = disk.get(ckey(k))
        if v is not None and not str(v).startswith("error:"):
            cache[k] = v
            hits += 1
    todo = [k for k in order
            if len(k[1]) >= MIN_TEXT_CHARS and cache.get(k) is None]
    print(f"[judge] {len(order)} unique pairs; {hits} already cached; "
          f"{len(todo)} need the judge")

    done = [0]
    lock = __import__("threading").Lock()

    def run(k):
        up, txt = k
        try:
            v = judge(txt, up)
        except Exception as e:                      # noqa: BLE001
            v = f"error:{type(e).__name__}"
        with lock:
            done[0] += 1
            disk[ckey(k)] = v
            n = done[0]
            if n % 100 == 0:
                cache_path.write_text(json.dumps(disk))   # checkpoint
        if n % 50 == 0:
            print(f"[judge] {n}/{len(todo)}", flush=True)
        return k, v

    if todo:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for k, v in ex.map(run, todo):
                cache[k] = v
        cache_path.write_text(json.dumps(disk))
        print(f"[judge] cache saved ({len(disk)} entries)")

    errs = sum(1 for v in cache.values() if isinstance(v, str) and v.startswith("error:"))
    if errs:
        print(f"[judge] WARNING {errs} judge errors; those rows are dropped")

    # Match tools.refusal.classify_refusal exactly: a response with no real prose
    # (a bare tool call, say) is NOT a refusal and still counts in the denominator.
    # Dropping those rows silently inflates the rate -- benign tool-mode prompts are
    # mostly bare tool calls, so it made over-refusal look like 100%.
    n_no_text = 0
    for p in pending:
        short = len(p["clean"]) < MIN_TEXT_CHARS
        v = cache.get(p["key"])
        if short:
            n_no_text += 1
            p["new_label"], p["new_refused"] = "no_text", False
        elif isinstance(v, str) and v.startswith("error:"):
            p["new_label"], p["new_refused"] = "judge_error", False
        else:
            p["new_label"], p["new_refused"] = v, (v == "refuse")
        p["old_refused"] = (not short) and regex_refused(p["clean"])
        p["usable"] = True

    groups = defaultdict(list)
    for p in pending:
        groups[(p["exp"], p["coef"])].append(p)
    print(f"[judge] {n_no_text}/{len(pending)} responses had no prose "
          f"(bare tool calls etc.) -> counted as not-refused")

    if a.raw_file:
        # Generic per-condition report: refusal over all rows, and over rows that
        # pass the v2 degeneracy screen. A degenerate response has no behaviour to
        # label, so the clean-only number is the one that means anything.
        from degeneracy import classify_v2, is_degenerate
        out = {"model": a.model, "source": raw_name, "n_judged": len(todo),
               "judge_errors": errs, "meta": {k: v for k, v in meta.items()
                                              if k not in ("steering",)},
               "conditions": {}}
        for (exp, coef), g_all in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            # a judge error is missing data, not a non-refusal: leave it out of the rates
            g = [x for x in g_all if x["new_label"] != "judge_error"]
            n_err = len(g_all) - len(g)
            if not g:
                continue
            v2 = [classify_v2(x["clean"]) for x in g]
            clean = [x for x, c in zip(g, v2) if c != "degenerate"]
            new = [x["new_refused"] for x in g]
            newc = [x["new_refused"] for x in clean]
            out["conditions"][f"{exp}|{coef}"] = {
                "n": len(g), "n_judge_errors": n_err,
                "refuse_all": round(float(np.mean(new)), 3), "refuse_all_ci": boot_ci(new),
                "degenerate_v1": round(float(np.mean([is_degenerate(x["clean"]) for x in g])), 3),
                "degenerate_v2": round(float(np.mean([c == "degenerate" for c in v2])), 3),
                "short_v2": round(float(np.mean([c == "short" for c in v2])), 3),
                "n_clean": len(clean),
                "refuse_clean": round(float(np.mean(newc)), 3) if newc else None,
                "refuse_clean_ci": boot_ci(newc) if len(newc) > 1 else None,
                "labels": {k: sum(1 for x in g if x["new_label"] == k)
                           for k in ("refuse", "caveat", "comply", "no_text", "judge_error")},
                "labels_clean": {k: sum(1 for x in clean if x["new_label"] == k)
                                 for k in ("refuse", "caveat", "comply", "no_text", "judge_error")},
            }
            r = out["conditions"][f"{exp}|{coef}"]
            print(f"  {exp:16s} coef={str(coef):8s} n={r['n']:4d} refuse={r['refuse_all']:.3f} "
                  f"degen v2={r['degenerate_v2']:.1%} refuse|clean={r['refuse_clean']} (n={r['n_clean']})")
        name = raw_name.replace("raw_", "").replace(".json", "_judged.json")
        (OUT / name).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {OUT / name}")
        return

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
                           for k in ("refuse", "caveat", "comply", "no_text", "judge_error")}}

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
