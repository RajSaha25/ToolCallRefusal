#!/usr/bin/env python3
"""Batched, bf16 (non-quantized) behavioral runner.

Same prompts / tool-call parsing / forbidden-action scoring as run_qwen_eval.py, but:
  - HF_HOME forced onto /workspace (the container /root disk is ~1.5 GB and OOMs downloads)
  - models loaded in bf16 (non-quantized), matching the mech-interp configuration
  - BATCHED generation (left-padded) instead of one prompt at a time
  - robust chat template: enable_thinking=False when the model supports it, else fall back
  - Anthropic judge optional (regex-only refusal labels when JUDGE_KEY is unset)
  - per-model resume (skips already-saved ids)

Env: MODELS (comma-sep ids), OUT (dir under repo, default results_nothink),
     BS (batch size, default 16), MAX_NEW (default 512), SAMPLE_N (0=full),
     HF_TOKEN (gated models), JUDGE_KEY (optional Anthropic key).
"""
import os, sys
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")  # MUST precede transformers import
from pathlib import Path

REPO_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_PATH))

OUTPUT_DIR = REPO_PATH / os.environ.get("OUT", "results_nothink")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [m.strip() for m in os.environ.get(
    "MODELS",
    "mistralai/Mistral-7B-Instruct-v0.3,google/gemma-3-27b-it,CohereLabs/c4ai-command-r7b-12-2024",
).split(",") if m.strip()]

HF_TOKEN  = os.environ.get("HF_TOKEN", "") or None
JUDGE_PROVIDER = "anthropic"
JUDGE_KEY = os.environ.get("JUDGE_KEY", "")
DATASET_PATH = str(REPO_PATH / "data" / "complete_combined_safety_eval_dataset_2304_normalized_labels.xlsx")
SAMPLE_N = int(os.environ.get("SAMPLE_N", "0")) or None
BS       = int(os.environ.get("BS", "16"))
MAX_NEW  = int(os.environ.get("MAX_NEW", "256"))  # refusal/tool-call decision lands early; long tails waste batch time
CHECKPOINT_EVERY_BATCHES = 2

print(f"[runner] OUT={OUTPUT_DIR} BS={BS} MAX_NEW={MAX_NEW} HF_HOME={os.environ['HF_HOME']}", flush=True)
print(f"[runner] MODELS={MODELS}", flush=True)

import re, json, gc
import pandas as pd
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from tools.registry import get_domain, list_domains

# ── Dataset (prep identical to run_qwen_eval.py) ───────────────────────
df = pd.read_excel(DATASET_PATH)
df.columns = [c.strip() for c in df.columns]
df['User Prompt'] = df['User Prompt'].str.replace('�', '—', regex=False)

def normalize_id(x): return str(x).strip()
df["ID_STR"] = df["ID"].apply(normalize_id)

DOMAIN_MAP = {'Finance':'finance','Healthcare':'healthcare',
              'Education / Academic Integrity':'education','Legal / Compliance':'legal'}

def _scenario_id(domain_key, scenario_label, request_type):
    if scenario_label == 'Direct request':
        return f'{domain_key}_direct_harmful' if request_type == 'Harmful' else f'{domain_key}_direct_safe'
    elif scenario_label == 'Indirect phrasing':       return f'{domain_key}_indirect'
    elif scenario_label == 'Justification / roleplay': return f'{domain_key}_roleplay'
    elif scenario_label == 'Structured tool-oriented': return f'{domain_key}_structured_tool'
    return f'{domain_key}_direct_safe'

df['domain_key']  = df['Domain'].map(DOMAIN_MAP)
df['scenario_id'] = df.apply(lambda r: _scenario_id(r['domain_key'], r['Scenario'], r['Request Type']), axis=1)
df['group_id']    = df['Domain'] + '|' + df['Scenario'] + '|' + df['Prompt Variant'] + '|' + df['System Condition']
if SAMPLE_N:
    df = df.sample(SAMPLE_N, random_state=42).reset_index(drop=True)
print(f'Dataset: {len(df)} rows | domains: {list_domains()}', flush=True)

# ── Refusal classification: shared module (see tools/refusal.py) ───────
from tools.refusal import classify_refusal as _shared_classify_refusal, make_anthropic_judge, strip_tool_markup

_judge = make_anthropic_judge(api_key=JUDGE_KEY) if (JUDGE_PROVIDER == 'anthropic' and JUDGE_KEY) else None
if _judge is None:
    print('[runner] WARNING: JUDGE_KEY unset — regex-only refusal labels, not comparable to judged runs', flush=True)

def classify_refusal(text, user_prompt=None):
    return _shared_classify_refusal(text, judge=_judge, user_prompt=user_prompt)

# ── Tool-call parser + markup strip (verbatim from run_qwen_eval.py) ───
def parse_tool_calls(text):
    calls = []
    m = re.search(r'\[TOOL_CALLS\]\s*(\[.*?\])', text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, list):
                return [{'name': c.get('name',''), 'arguments': c.get('arguments',{})} for c in parsed]
        except json.JSONDecodeError: pass
    for raw in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try:
            c = json.loads(raw.strip())
            calls.append({'name': c.get('name',''), 'arguments': c.get('arguments', c.get('parameters',{}))})
        except json.JSONDecodeError: pass
    if calls: return calls
    # Command-R / Cohere: <|START_ACTION|>[{"tool_name":..., "parameters":...}]<|END_ACTION|>
    m = re.search(r'<\|START_ACTION\|>(.*?)<\|END_ACTION\|>', text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1).strip())
            if isinstance(parsed, list):
                out = [{'name': c.get('tool_name', c.get('name','')), 'arguments': c.get('parameters', c.get('arguments',{}))} for c in parsed if isinstance(c, dict)]
                if out: return out
        except json.JSONDecodeError: pass
    m = re.search(r'<\|python_tag\|>(.*?)(?:<\|eom_id\||<\|eot_id\||$)', text, re.DOTALL)
    if m:
        try:
            c = json.loads(m.group(1).strip())
            if isinstance(c, dict) and 'name' in c:
                return [{'name': c['name'], 'arguments': c.get('parameters', c.get('arguments',{}))}]
        except json.JSONDecodeError: pass
    m = re.search(r'(\[\s*\{.*?\}\s*\])', text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, list) and parsed and all(isinstance(c, dict) and ('name' in c or 'tool_name' in c) for c in parsed):
                return [{'name': c.get('name', c.get('tool_name','')), 'arguments': c.get('arguments', c.get('parameters',{}))} for c in parsed]
        except json.JSONDecodeError: pass
    return []

# strip_tool_markup comes from tools.refusal (shared with classification).

# ── model globals + batched generation ────────────────────────────────
model = None
tokenizer = None

def render_prompt(system_prompt, user_prompt, tools):
    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user',   'content': user_prompt}]
    attempts = ([dict(tools=tools, enable_thinking=False), dict(tools=tools)] if tools else []) \
               + [dict(enable_thinking=False), dict()]
    for kw in attempts:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
        except Exception:
            continue
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def generate_batch(prompts):
    enc = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=2048).to(model.device)
    out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                         temperature=None, top_p=None, pad_token_id=tokenizer.eos_token_id)
    gen = out[:, enc['input_ids'].shape[1]:]
    parsed = []
    for i in range(len(prompts)):
        raw = tokenizer.decode(gen[i], skip_special_tokens=False)
        tcs = parse_tool_calls(raw)
        clean = strip_tool_markup(raw)
        clean = tokenizer.decode(tokenizer.encode(clean, add_special_tokens=False), skip_special_tokens=True)
        parsed.append((clean, tcs))
    return parsed

# ── main loop over models ──────────────────────────────────────────────
for MODEL_ID in MODELS:
    MODEL_SHORT = MODEL_ID.split('/')[-1]
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f'results_{MODEL_SHORT}.csv')

    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        if len(existing) >= len(df):
            print(f'[SKIP] {MODEL_SHORT} already complete ({len(existing)} rows)', flush=True)
            continue

    start_time = datetime.now()
    print(f'\n===== START {MODEL_SHORT} @ {start_time:%H:%M:%S} =====', flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto',
            trust_remote_code=True, token=HF_TOKEN,
        )
        model.eval()
        print(f'  loaded {MODEL_SHORT} on {model.device} | gpu {torch.cuda.memory_allocated()/1e9:.1f} GB', flush=True)
    except Exception as e:
        print(f'  [ERROR] load failed {MODEL_SHORT}: {e}\n  skipping.', flush=True)
        continue

    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        existing["id"] = existing["id"].apply(normalize_id)
        done = set(existing["id"])
        df_run = df[~df["ID_STR"].isin(done)].reset_index(drop=True)
        results = existing.to_dict("records")
        print(f'  resume: {len(done)} done, {len(df_run)} remaining', flush=True)
    else:
        df_run = df
        results = []

    # Pre-render every prompt, then sort by length so each batch is length-homogeneous
    # (short no-tool prompts batch together; long tool-schema prompts batch together) —
    # this minimizes padding waste and keeps a short generation from waiting on a long one.
    rendered = []
    for _, row in df_run.iterrows():
        domain = get_domain(row['domain_key'])
        mode = row['Mode']
        tools = list(domain.tools_for_llm) if mode != 'No-tool chat' else None
        p = render_prompt(row['System Prompt'], row['User Prompt'], tools)
        rendered.append((row, domain, mode, p))
    rendered.sort(key=lambda x: len(x[3]))
    nb = 0
    for start in range(0, len(rendered), BS):
        chunk = rendered[start:start + BS]
        prompts = [c[3] for c in chunk]
        try:
            gen = generate_batch(prompts)
        except Exception as e:
            print(f'  [batch ERROR @ {start}] {e}', flush=True)
            gen = [('', []) for _ in prompts]
        for (row, domain, mode, _p), (resp, tcs) in zip(chunk, gen):
            req_type = row['Request Type']
            scenario_id = row['scenario_id']
            refusal = classify_refusal(resp, row['User Prompt'])
            t_ref = refusal['refused']
            if mode != 'No-tool chat':
                viol = domain.score_tool_calls(scenario_id, tcs)
                tc_safe = len(viol) == 0
                vj = json.dumps([v.as_dict() for v in viol])
            else:
                tc_safe, vj = True, '[]'
            results.append({
                'id': row['ID'], 'model': MODEL_SHORT, 'domain': row['Domain'],
                'scenario': row['Scenario'], 'scenario_id': scenario_id, 'mode': mode,
                'system_condition': row['System Condition'], 'prompt_variant': row['Prompt Variant'],
                'request_type': req_type, 'group_id': row['group_id'], 'response_text': resp,
                'tool_calls': json.dumps(tcs), 'refusal_method': refusal['method'],
                'refusal_label': refusal['label'], 'refused': t_ref,
                'tc_safe': tc_safe,
                'divergence': (req_type == 'Harmful') and t_ref and not tc_safe,
                'over_refusal': (req_type == 'Benign') and t_ref, 'violations': vj, 'error': '',
            })
        nb += 1
        if nb % CHECKPOINT_EVERY_BATCHES == 0:
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
            mins = (datetime.now() - start_time).seconds // 60
            print(f'  {len(results)}/{len(df)} rows ({mins}m)', flush=True)

    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    mins = (datetime.now() - start_time).seconds // 60
    print(f'  DONE {MODEL_SHORT}: {len(results)} rows in {mins} min -> {OUTPUT_PATH}', flush=True)
    del model, tokenizer
    model = tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()

print('ALL DONE', flush=True)
