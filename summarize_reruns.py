#!/usr/bin/env python3
"""Collect the per-model rerun JSON into the tables that go in the write-up."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "relabel_analysis"

# published values, for the like-for-like column
PUB_AUC = {"Qwen3-14B": 0.724, "Mistral-7B-Instruct-v0.3": 0.689,
           "c4ai-command-r7b-12-2024": 0.751, "gemma-3-27b-it": 0.791,
           "Meta-Llama-3.1-70B-Instruct": 0.881}
ORDER = ["Qwen3-14B", "Mistral-7B-Instruct-v0.3", "c4ai-command-r7b-12-2024",
         "gemma-3-27b-it", "Meta-Llama-3.1-70B-Instruct"]
SHORT = {"Qwen3-14B": "Qwen3-14B", "Mistral-7B-Instruct-v0.3": "Mistral-7B",
         "c4ai-command-r7b-12-2024": "Command-R-7B", "gemma-3-27b-it": "Gemma-3-27B",
         "Meta-Llama-3.1-70B-Instruct": "Llama-3.1-70B"}


def load(m):
    p = OUT / f"gpu_{m}.json"
    return json.loads(p.read_text()) if p.exists() else None


rows = [(m, load(m)) for m in ORDER]
rows = [(m, d) for m, d in rows if d]

print("=" * 104)
print("DIRECTION COSINES  (r_behav = refused vs complied harmful, new labels)")
print("=" * 104)
print(f"{'model':<16}{'layer':>7}{'proj_gap':>11}{'r_text·r_behav':>16}{'(caveats in)':>14}"
      f"{'r_text·r_tool':>15}{'behav old·new':>15}{'n/class':>9}")
for m, d in rows:
    # proj_gap was added after the first three models ran; they predate it
    gap = f"{d['proj_gap']:.1f}" if "proj_gap" in d else "-"
    print(f"{SHORT[m]:<16}{d['layer']:>7}{gap:>11}"
          f"{d.get('cos_rtext_rbehav_strict', float('nan')):>16.3f}"
          f"{d.get('cos_rtext_rbehav_loose', float('nan')):>14.3f}"
          f"{d.get('cos_rtext_rtool', float('nan')):>15.3f}"
          f"{d.get('cos_rbehav_old_new', float('nan')):>15.3f}"
          f"{d.get('n_behav_strict', 0):>9}")

print()
print("=" * 104)
print("AUC OF THE PROJECTION")
print("=" * 104)
print(f"{'model':<16}{'vs unsafe(old)':>16}{'vs unsafe(fixed)':>18}{'published':>11}"
      f"{'reproduces?':>13}{'vs refusal old':>16}{'vs refusal new':>16}")
for m, d in rows:
    a_old = d["auc_unsafe_old"]["auc"]
    ci = d["auc_unsafe_old"]["ci"]
    pub = PUB_AUC.get(m)
    ok = "yes" if pub is not None and ci[0] <= pub <= ci[1] else "NO"
    print(f"{SHORT[m]:<16}{a_old:>16.3f}{d['auc_unsafe_fixed']['auc']:>18.3f}{pub:>11.3f}"
          f"{ok:>13}{d['auc_refusal_old']['auc']:>16.3f}{d['auc_refusal_new']['auc']:>16.3f}")

print()
print("=" * 104)
print("STEERING / ABLATION / ADDITION  (old regex vs new judge)")
print("=" * 104)
for m in ORDER:
    p = OUT / f"steer_{m}.json"
    if not p.exists():
        continue
    s = json.loads(p.read_text())
    print(f"\n-- {SHORT[m]} (layer {s['layer']}, add_coef {s['add_coef']})")
    for name in ("ablation_base", "ablation_ablated", "addition_base", "addition_added"):
        r = s.get(name)
        if r:
            print(f"   {name:<18} regex={r['old']:.3f}  judge={r['new']:.3f}   {r['labels']}")
    st = s.get("steering")
    if st and len(st["grid"]) > 1:
        print(f"   {'coef':>8}{'harmful unsafe':>16}{'benign refuse old':>20}{'benign refuse new':>20}")
        for i, c in enumerate(st["grid"]):
            print(f"   {c:>8}{st['harmful_unsafe'][i]:>16.3f}"
                  f"{st['benign_refuse_old'][i]:>20.3f}{st['benign_refuse_new'][i]:>20.3f}")
