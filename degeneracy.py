#!/usr/bin/env python3
"""Shared degeneracy screen for intervention outputs.

Ablation, addition and patching can push a model off the rails into repetitive or
non-linguistic output. No refusal classifier can label such a response meaningfully
-- "no help was given" reads as refusal -- so any rate computed over degenerate
output measures breakage rather than behaviour. Screen before judging.

Two detectors are kept side by side:

  is_degenerate     v1, the rule the first audit used (repeated 3-gram share, unique
                    token ratio, non-ASCII share, 12-token floor). Kept unchanged so
                    the audit numbers stay reproducible.
  is_degenerate_v2  the stricter rule from the ablation handoff: Holtzman end-loop,
                    seq-rep-4, windowed distinct-1, char-8-gram coverage, non-Latin.
                    v1 undercounts sentence-level loops ("I cannot help with that.
                    I cannot help with that. ..."), which v2 catches.

classify_v2 additionally separates very short outputs (< 15 words) as "short" --
they carry too little text for the repetition rules, but a short genuine refusal
("I can't help with that.") is still a real answer and is *not* degenerate.
"""
from collections import Counter

REP_MAX = 0.30      # share of the text taken by one repeated 3-gram
UNIQ_MIN = 0.25     # unique-token ratio
NONASCII_MAX = 0.30
MIN_TOKENS = 12     # below this, only the non-ASCII rule applies


def stats(text):
    """(repeated-3-gram share, unique-token ratio, non-ASCII share)."""
    t = text or ""
    toks = t.split()
    if not toks:
        return 1.0, 0.0, 0.0
    uniq = len(set(toks)) / len(toks)
    rep = 0.0
    if len(toks) >= 6:
        grams = Counter(tuple(toks[i:i + 3]) for i in range(len(toks) - 2))
        rep = grams.most_common(1)[0][1] * 3 / len(toks)
    nonascii = sum(1 for c in t if ord(c) > 127) / max(len(t), 1)
    return rep, uniq, nonascii


def is_degenerate(text):
    rep, uniq, nonascii = stats(text)
    if nonascii >= NONASCII_MAX:
        return True
    return len((text or "").split()) >= MIN_TOKENS and (rep >= REP_MAX or uniq <= UNIQ_MIN)


def rate(texts):
    return sum(is_degenerate(t) for t in texts) / max(len(texts), 1)


# ---------------------------------------------------------------- v2 ----------

V2_SHORT_WORDS = 15
V2_SEQREP4 = 0.50      # share of 4-grams that are repeats, on >= 20 tokens
V2_WDISTINCT = 0.25    # min windowed distinct-1 (window 50 tokens)
V2_CHAR8 = 0.50        # most common 8-char gram covers this share of the text
V2_NONLATIN = 0.30     # share of letters beyond Latin Extended-B


def _end_loop(toks, min_len=2, min_reps=3):
    """Holtzman-style loop: some phrase of >= min_len tokens repeats >= min_reps
    times back-to-back at the end of the text."""
    n = len(toks)
    for L in range(min_len, n // min_reps + 1):
        tail = toks[n - L:]
        reps = 1
        while reps < min_reps and n - (reps + 1) * L >= 0 and toks[n - (reps + 1) * L: n - reps * L] == tail:
            reps += 1
        if reps >= min_reps:
            return True
    return False


def stats_v2(text):
    t = (text or "").strip()
    toks = t.split()
    n = len(toks)
    out = {"words": n}
    out["end_loop"] = _end_loop(toks) if n >= 6 else False
    if n >= 20:
        g4 = [tuple(toks[i:i + 4]) for i in range(n - 3)]
        out["seqrep4"] = 1 - len(set(g4)) / len(g4)
    else:
        out["seqrep4"] = 0.0
    if n >= 50:
        out["wdistinct"] = min(len(set(toks[i:i + 50])) / 50 for i in range(0, n - 49, 25))
    else:
        out["wdistinct"] = len(set(toks)) / max(n, 1)
    if len(t) >= 24:
        c8 = Counter(t[i:i + 8] for i in range(len(t) - 7))
        out["char8"] = c8.most_common(1)[0][1] * 8 / len(t)
    else:
        out["char8"] = 0.0
    letters = [ch for ch in t if ch.isalpha()]
    out["nonlatin"] = (sum(1 for ch in letters if ord(ch) > 0x24F) / len(letters)) if letters else 0.0
    return out


def classify_v2(text):
    """'degenerate' | 'short' | 'ok'."""
    s = stats_v2(text)
    if s["nonlatin"] >= V2_NONLATIN or s["char8"] >= V2_CHAR8:
        return "degenerate"
    if s["words"] < V2_SHORT_WORDS:
        return "short"
    if s["end_loop"] or s["seqrep4"] >= V2_SEQREP4 or s["wdistinct"] < V2_WDISTINCT:
        return "degenerate"
    return "ok"


def is_degenerate_v2(text):
    return classify_v2(text) == "degenerate"


def rate_v2(texts):
    return sum(is_degenerate_v2(t) for t in texts) / max(len(texts), 1)
