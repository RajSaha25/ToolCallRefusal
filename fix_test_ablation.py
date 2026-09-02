#!/usr/bin/env python3
"""One-prompt fix test for refusal-direction ablation degeneration.

Reproduces the rerun's direction extraction (last-token diff-in-means, layer =
max separation inside the 35-85% depth band) and all-layer projection-removal
ablation, then tries each candidate fix on a few prompts and reports a
degeneracy metric + the generated text per condition:

  none        baseline
  ablate_all  Raj's hook: remove r̂ at every layer, every position (should reproduce collapse on Gemma)
  R2_winsor   direction rebuilt from per-dim winsorised (0.5/99.5 pct) activations
  R3_meanout  mean-activation component projected out of r̂ before ablating
  R4_upper    ablate only layers whose own diff-in-means direction has cos>0.3 with r̂
  R5_skip0    ablate_all but never touch position 0 of the prefill
  R345        R3 + R4 + R5 together
  R7_random   random unit direction, ablate_all (control)

Also prints Arditi's KL score (mean KL(clean||ablated) of the next-token
distribution on benign prompts; their threshold is 0.1) for r̂ and for the
fixed directions, plus direction diagnostics (common-mode ratio, cos with the
mean activation and with the BOS activation, top-dim concentration).

Usage:
  python fix_test_ablation.py --model google/gemma-3-4b-it --n-dir 48 --n-test 3
Env: HF_TOKEN (gated models). Runs on cuda / mps / cpu.
"""
import argparse, json, math, os, random, re, sys, time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging
hf_logging.set_verbosity_error()

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompts", default=str(Path(__file__).resolve().parent / "notool_prompts.json"))
ap.add_argument("--n-dir", type=int, default=48, help="prompts per class for the direction")
ap.add_argument("--n-test", type=int, default=3, help="harmful prompts to generate on")
ap.add_argument("--n-kl", type=int, default=16, help="benign prompts for the KL score")
ap.add_argument("--max-new", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--bs", type=int, default=1, help="batch size; keep 1 — left-padded batches go NaN under SDPA on MPS")
ap.add_argument("--conditions", default="none,ablate_all,R2_winsor,R3_meanout,R4_upper,R5_skip0,R345,R7_random")
ap.add_argument("--out", default=None)
args = ap.parse_args()
random.seed(args.seed); torch.manual_seed(args.seed)

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = getattr(torch, args.dtype)
tok = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=DTYPE, token=os.environ.get("HF_TOKEN"))
model.to(DEVICE).eval()


def find_layers(m):
    best = None
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8 and all(hasattr(x, "self_attn") or hasattr(x, "attn") for x in mod):
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    assert best, "could not find decoder layers"
    return best


LAYERS_NAME, LAYERS = find_layers(model)
NL = len(LAYERS)
print(f"[model] {args.model} device={DEVICE} dtype={DTYPE} layers={NL} ({LAYERS_NAME})", flush=True)

P = json.load(open(args.prompts))
harm = [x["prompt"] for x in P["harmful"]]
ben = [x["prompt"] for x in P["benign"]]
random.shuffle(harm); random.shuffle(ben)
dir_h, dir_b = harm[: args.n_dir], ben[: args.n_dir]
test_prompts = harm[args.n_dir: args.n_dir + args.n_test]
kl_prompts = ben[args.n_dir: args.n_dir + args.n_kl]


def render(p):
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def last_token_acts(prompts, bs=None):
    bs = bs or args.bs
    """returns tensor [n, NL+1, d] of hidden states at the last prompt token, and BOS acts [n, NL+1, d]."""
    outs, boss = [], []
    for i in range(0, len(prompts), bs):
        enc = tok([render(p) for p in prompts[i:i + bs]], return_tensors="pt", padding=True).to(DEVICE)
        o = model(**enc, output_hidden_states=True)
        hs = torch.stack(o.hidden_states, 1)  # [b, NL+1, T, d]
        outs.append(hs[:, :, -1, :].float().cpu())
        # first real token per row (left padding)
        first = enc["attention_mask"].argmax(1)
        boss.append(torch.stack([hs[j, :, first[j], :] for j in range(hs.shape[0])]).float().cpu())
    return torch.cat(outs), torch.cat(boss)


t0 = time.time()
A_h, B_h = last_token_acts(dir_h)
A_b, B_b = last_token_acts(dir_b)
print(f"[acts] {A_h.shape[0]}+{A_b.shape[0]} prompts in {time.time()-t0:.0f}s", flush=True)
if not torch.isfinite(A_h).all() or not torch.isfinite(A_b).all():
    sys.exit(f"[FATAL] non-finite hidden states (dtype={DTYPE}, device={DEVICE}). "
             "Try --bs 1 (padding + SDPA on MPS) and/or --dtype float32.")

lo, hi = int(0.35 * NL), int(0.85 * NL)
diff = A_h.mean(0) - A_b.mean(0)          # [NL+1, d]
sep = diff.norm(dim=-1)
L = int(torch.argmax(sep[lo:hi + 1]).item() + lo)
unit = lambda v: v / v.norm()
r_text = unit(diff[L])
per_layer_dirs = torch.stack([unit(diff[l]) for l in range(NL + 1)])
cos_L = per_layer_dirs @ r_text
gap = ((A_h[:, L] @ r_text).mean() - (A_b[:, L] @ r_text)).mean().item() if False else (A_h[:, L] @ r_text).mean().item() - (A_b[:, L] @ r_text).mean().item()
mu_all = torch.cat([A_h[:, L], A_b[:, L]]).mean(0)
bos_mean = torch.cat([B_h[:, L], B_b[:, L]]).mean(0)
top = (r_text ** 2).sort(descending=True).values
print(f"[dir] layer={L}/{NL}  gap={gap:.1f}  mean_proj/gap={(mu_all @ r_text).item()/gap:.2f}  "
      f"cos(r,mean)={F.cosine_similarity(r_text, mu_all, 0).item():.3f}  cos(r,BOS)={F.cosine_similarity(r_text, bos_mean, 0).item():.3f}  "
      f"top1={top[0].item():.3f} top10={top[:10].sum().item():.3f}  |mean|={mu_all.norm().item():.0f} |BOS|={bos_mean.norm().item():.0f}", flush=True)
print("[dir] cos(r, r_L) by layer:", " ".join(f"{c:.2f}" for c in cos_L.tolist()), flush=True)

# ---- candidate fixed directions ----
def winsor_dir():
    X = torch.cat([A_h[:, L], A_b[:, L]])
    lo_q, hi_q = torch.quantile(X, 0.005, dim=0), torch.quantile(X, 0.995, dim=0)
    Xh = torch.minimum(torch.maximum(A_h[:, L], lo_q), hi_q)
    Xb = torch.minimum(torch.maximum(A_b[:, L], lo_q), hi_q)
    return unit(Xh.mean(0) - Xb.mean(0))

def meanout_dir(r):
    u = unit(mu_all)
    return unit(r - (r @ u) * u)

r_w = winsor_dir()
r_m = meanout_dir(r_text)
r_rand = unit(torch.randn_like(r_text))
upper_layers = [l for l in range(1, NL + 1) if cos_L[l] > 0.3]
print(f"[dir] cos(r, winsor)={(r_text@r_w).item():.3f} cos(r, meanout)={(r_text@r_m).item():.3f} "
      f"R4 layers (cos>0.3): {len(upper_layers)}/{NL} -> {upper_layers[:3]}..{upper_layers[-3:] if upper_layers else []}", flush=True)

# ---- hooks ----
class Ablate:
    def __init__(self, r, layers, skip0=False):
        self.r = r.to(DEVICE, DTYPE); self.layers = layers; self.skip0 = skip0; self.h = []
    def hook(self, m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        proj = (h @ self.r).unsqueeze(-1) * self.r
        if self.skip0 and h.shape[1] > 1:      # prefill: leave position 0 alone
            proj[:, 0, :] = 0
        h = h - proj
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
    def __enter__(self):
        for l in self.layers:
            self.h.append(LAYERS[l - 1].register_forward_hook(self.hook))
        return self
    def __exit__(self, *a):
        for x in self.h: x.remove()

ALL = list(range(1, NL + 1))
CONDS = {
    "none": None,
    "ablate_all": lambda: Ablate(r_text, ALL),
    "R2_winsor": lambda: Ablate(r_w, ALL),
    "R3_meanout": lambda: Ablate(r_m, ALL),
    "R4_upper": lambda: Ablate(r_text, upper_layers or ALL),
    "R5_skip0": lambda: Ablate(r_text, ALL, skip0=True),
    "R345": lambda: Ablate(r_m, upper_layers or ALL, skip0=True),
    "R7_random": lambda: Ablate(r_rand, ALL),
}

@contextmanager
def cond(name):
    c = CONDS[name]
    if c is None:
        yield
    else:
        with c(): yield

# ---- KL score (Arditi C.1): benign prompts, next-token dist at last position ----
@torch.no_grad()
def kl_score(name, bs=None):
    bs = bs or args.bs
    kls = []
    for i in range(0, len(kl_prompts), bs):
        enc = tok([render(p) for p in kl_prompts[i:i + bs]], return_tensors="pt", padding=True).to(DEVICE)
        base = F.log_softmax(model(**enc).logits[:, -1, :].float(), -1)
        with cond(name):
            alt = F.log_softmax(model(**enc).logits[:, -1, :].float(), -1)
        kls.append((base.exp() * (base - alt)).sum(-1).cpu())
    return torch.cat(kls).mean().item()

# ---- degeneracy ----
def degeneracy(text):
    w = text.split()
    rep4 = 0.0
    if len(w) >= 12:
        grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
        c = Counter(grams).most_common(1)[0][1]
        rep4 = c * 4 / len(w)
    distinct = len(set(w)) / max(1, len(w))
    letters = [ch for ch in text if ch.isalpha()]
    nonlatin = sum(1 for ch in letters if ord(ch) > 0x24F) / max(1, len(letters))
    # character-level loop (catches '!!!!!!', '수상수상', 'I can I can' even without spaces)
    char8 = 0.0
    if len(text) >= 24:
        grams = [text[i:i + 8] for i in range(len(text) - 7)]
        char8 = Counter(grams).most_common(1)[0][1] * 8 / len(text)
    few_chars = len(text) >= 20 and len(set(text.strip())) <= 3
    flag = (rep4 >= 0.5) or (char8 >= 0.5) or few_chars or (len(w) >= 20 and distinct <= 0.2) or (nonlatin >= 0.3)
    return dict(rep4=round(rep4, 2), char8=round(char8, 2), distinct=round(distinct, 2), nonlatin=round(nonlatin, 2), degenerate=flag)

@torch.no_grad()
def generate(prompt, name):
    enc = tok(render(prompt), return_tensors="pt").to(DEVICE)
    with cond(name):
        out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

conds = [c for c in args.conditions.split(",") if c in CONDS]
results = {"model": args.model, "layer": L, "n_layers": NL, "gap": gap, "kl": {}, "gens": []}
print("\n[KL] Arditi kl_score on benign prompts (threshold 0.1):", flush=True)
for name in conds:
    if name == "none": continue
    k = kl_score(name); results["kl"][name] = k
    print(f"   {name:12s} KL={k:.3f}  {'PASS' if k < 0.1 else 'FAIL'}", flush=True)

print("\n[GEN]", flush=True)
for p in test_prompts:
    print(f"\n=== PROMPT: {p[:140]}", flush=True)
    for name in conds:
        t0 = time.time(); txt = generate(p, name); d = degeneracy(txt)
        results["gens"].append(dict(prompt=p, cond=name, text=txt, **d))
        tag = "DEGENERATE" if d["degenerate"] else "ok"
        print(f"  [{name:12s}] {tag:10s} rep4={d['rep4']:.2f} char8={d['char8']:.2f} distinct={d['distinct']:.2f} nonlatin={d['nonlatin']:.2f} ({time.time()-t0:.0f}s)\n"
              f"      {txt[:260]!r}", flush=True)

out = args.out or f"fix_test_{args.model.split('/')[-1]}.json"
json.dump(results, open(out, "w"), indent=1)
print(f"\n[done] wrote {out}")
