#!/usr/bin/env python3
"""Cross-check the primary (Haiku) three-way judge against a stronger judge (Sonnet).

Reads relabel_out/relabel_*.csv (produced by relabel_refusals.py), draws a
stratified sample (per model x request type), re-judges it with the stronger
model using the IDENTICAL prompt from tools/refusal.py, and reports three-way
and binary (refuse vs not) agreement with Cohen's kappa. Disagreements are
dumped for manual inspection and for seeding the human-annotation set.

Env: JUDGE_KEY (required), JUDGE2_MODEL (default claude-sonnet-5),
     PER_CELL (default 45), WORKERS (default 6), RELABEL_OUT (default ./relabel_out)
"""
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from tools.refusal import make_anthropic_judge, strip_tool_markup

JUDGE_KEY = os.environ["JUDGE_KEY"]
MODEL2 = os.environ.get("JUDGE2_MODEL", "claude-sonnet-5")
PER_CELL = int(os.environ.get("PER_CELL", "45"))
WORKERS = int(os.environ.get("WORKERS", "6"))
OUT_DIR = Path(os.environ.get("RELABEL_OUT", REPO / "relabel_out"))
CACHE_PATH = OUT_DIR / f"judge_cache_v2_{MODEL2.replace('/', '_')}.json"

_cache_lock = threading.Lock()
cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}


def _key(user_prompt, cleaned):
    return hashlib.sha256((user_prompt + "\x00" + cleaned).encode()).hexdigest()[:24]


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def main():
    frames = []
    for f in sorted(glob(str(OUT_DIR / "relabel_*.csv"))):
        df = pd.read_csv(f)
        df["source_model"] = Path(f).stem.replace("relabel_", "")
        frames.append(df)
    allrows = pd.concat(frames, ignore_index=True)
    pool = allrows[allrows["new_label"].isin(["refused", "caveat", "complied"])].copy()
    # shuffle + groupby.head keeps all columns (groupby.apply drops the
    # grouping columns on pandas 3, which broke the stratification fields)
    sample = (
        pool.sample(frac=1, random_state=11)
        .groupby(["source_model", "request_type"])
        .head(PER_CELL)
        .reset_index(drop=True)
    )
    sample["cleaned"] = sample["response_text"].fillna("").map(strip_tool_markup)
    sample["user_prompt"] = sample["user_prompt"].fillna("")
    sample["k2"] = [_key(up, c) for up, c in zip(sample["user_prompt"], sample["cleaned"])]
    print(f"[xcheck] sample={len(sample)} rows | judge2={MODEL2}", flush=True)

    judge2 = make_anthropic_judge(api_key=JUDGE_KEY, model=MODEL2, max_retries=5, retry_wait=5.0)
    pending = {k: (c, up) for k, c, up in zip(sample["k2"], sample["cleaned"], sample["user_prompt"])
               if cache.get(k) is None}
    print(f"[xcheck] {len(pending)} not cached", flush=True)
    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futures = {ex.submit(judge2, c, up): k for k, (c, up) in pending.items()}
        for fut in as_completed(futures):
            k = futures[fut]
            try:
                verdict = fut.result()
            except Exception as e:
                print(f"[xcheck] error after retries: {type(e).__name__}: {e}", flush=True)
                verdict = None
            with _cache_lock:
                cache[k] = verdict
            done += 1
            if done % 100 == 0:
                CACHE_PATH.write_text(json.dumps(cache))
                print(f"[xcheck] {done}/{len(pending)}", flush=True)
    CACHE_PATH.write_text(json.dumps(cache))

    v2l = {"refuse": "refused", "caveat": "caveat", "comply": "complied"}
    sample["label2"] = sample["k2"].map(lambda k: v2l.get(cache.get(k), "judge_error"))
    ok = sample[sample["label2"] != "judge_error"].copy()
    a, b = ok["new_label"], ok["label2"]
    bin_a, bin_b = a == "refused", b == "refused"

    report = {
        "n": len(ok),
        "judge_errors": int((sample["label2"] == "judge_error").sum()),
        "three_way_agreement": round(float((a == b).mean()), 3),
        "three_way_kappa": round(kappa(a, b), 3),
        "binary_agreement": round(float((bin_a == bin_b).mean()), 3),
        "binary_kappa": round(kappa(bin_a, bin_b), 3),
        "confusion": ok.groupby(["new_label", "label2"]).size().to_dict(),
    }
    for rt in ("Harmful", "Benign"):
        s = ok[ok["request_type"] == rt]
        report[f"binary_agreement_{rt.lower()}"] = round(float(((s["new_label"] == "refused") == (s["label2"] == "refused")).mean()), 3)

    dis = ok[a != b][["source_model", "id", "request_type", "user_prompt", "response_text", "new_label", "label2"]]
    dis.to_csv(OUT_DIR / "crosscheck_disagreements.csv", index=False)
    (OUT_DIR / "crosscheck_report.json").write_text(json.dumps({k: str(v) if isinstance(v, dict) else v for k, v in report.items()}, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "confusion"}, indent=2))
    print("[confusion] haiku -> sonnet:", report["confusion"], flush=True)
    print(f"[done] {len(dis)} disagreements -> {OUT_DIR}/crosscheck_disagreements.csv", flush=True)


if __name__ == "__main__":
    main()
