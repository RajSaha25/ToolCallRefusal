#!/usr/bin/env python3
"""Local headless runner for Behavioral_eval.ipynb — Qwen3-14B on the A100.
Faithfully reuses the notebook's Cell 5 (helpers) and Cell 6 (eval loop) verbatim.
Colab-specific cells (Drive mount / userdata) are replaced by this local config.
Config via env: SAMPLE_N (int, 0=full), OUT (output dir name under repo).
"""
import os, sys
from pathlib import Path

REPO_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_PATH))

OUTPUT_DIR = REPO_PATH / os.environ.get("OUT", "results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MODELS ──────────────────────────────────────────────────────────────
MODELS = ["Qwen/Qwen3-14B"]            # not gated — no HF token required

HF_TOKEN = os.environ.get("HF_TOKEN", "")          # empty -> from_pretrained uses None

# ── JUDGE (anthropic / gpt) ─────────────────────────────────────────────
JUDGE_PROVIDER = "anthropic"
JUDGE_KEY      = os.environ["JUDGE_KEY"]           # passed in via env (not hardcoded)

# ── DATASET (local copy pulled from Drive) ──────────────────────────────
DATASET_PATH = str(REPO_PATH / "data" / "complete_combined_safety_eval_dataset_2304_normalized_labels.xlsx")

# ── SAMPLING / CHECKPOINT ───────────────────────────────────────────────
SAMPLE_N         = int(os.environ.get("SAMPLE_N", "0")) or None
CHECKPOINT_EVERY = 25

print(f"[runner] OUTPUT_DIR={OUTPUT_DIR}  SAMPLE_N={SAMPLE_N}  model={MODELS[0]}", flush=True)

# ====================== Cell 5 · helpers (verbatim) =====================
import re, json, gc
import pandas as pd
import torch
from datetime import datetime
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tools.registry import get_domain, list_domains

# ── Dataset ────────────────────────────────────────────────────────────
df = pd.read_excel(DATASET_PATH)
df.columns = [c.strip() for c in df.columns]
df['User Prompt'] = df['User Prompt'].str.replace('\ufffd', '\u2014', regex=False)

def normalize_id(x):
    return str(x).strip()

df["ID_STR"] = df["ID"].apply(normalize_id)

DOMAIN_MAP = {
    'Finance':                         'finance',
    'Healthcare':                      'healthcare',
    'Education / Academic Integrity':  'education',
    'Legal / Compliance':              'legal',
}

def _scenario_id(domain_key, scenario_label, request_type):
    if scenario_label == 'Direct request':
        return f'{domain_key}_direct_harmful' if request_type == 'Harmful' else f'{domain_key}_direct_safe'
    elif scenario_label == 'Indirect phrasing':      return f'{domain_key}_indirect'
    elif scenario_label == 'Justification / roleplay': return f'{domain_key}_roleplay'
    elif scenario_label == 'Structured tool-oriented': return f'{domain_key}_structured_tool'
    return f'{domain_key}_direct_safe'

df['domain_key']  = df['Domain'].map(DOMAIN_MAP)
df['scenario_id'] = df.apply(lambda r: _scenario_id(r['domain_key'], r['Scenario'], r['Request Type']), axis=1)
df['group_id']    = df['Domain'] + '|' + df['Scenario'] + '|' + df['Prompt Variant'] + '|' + df['System Condition']

if SAMPLE_N:
    df = df.sample(SAMPLE_N, random_state=42).reset_index(drop=True)

print(f'Dataset: {len(df)} rows loaded')
print('Domains:', list_domains())

# ── Judge: shared classifier module (see tools/refusal.py) ─────────────
from tools.refusal import classify_refusal as _shared_classify_refusal, make_anthropic_judge, strip_tool_markup

_judge = make_anthropic_judge(api_key=JUDGE_KEY)

def classify_refusal(text):
    return _shared_classify_refusal(text, judge=_judge)

# ── Tool-call parser ───────────────────────────────────────────────────
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
            if isinstance(parsed, list) and all('name' in c for c in parsed):
                return [{'name': c['name'], 'arguments': c.get('arguments', c.get('parameters',{}))} for c in parsed]
        except json.JSONDecodeError: pass
    return []

# strip_tool_markup comes from tools.refusal (shared with classification).

# ── Inference (uses global model / tokenizer set inside the loop) ──────
@torch.no_grad()
def run_inference(system_prompt, user_prompt, tools=None, max_new_tokens=512):
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user',   'content': user_prompt},
    ]
    if tools:
        try:
            prompt_str = tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    else:
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs     = tokenizer(prompt_str, return_tensors='pt', truncation=True, max_length=2048).to(model.device)
    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, temperature=None, top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids    = output_ids[0][inputs['input_ids'].shape[1]:]
    raw_text   = tokenizer.decode(new_ids, skip_special_tokens=False)
    tool_calls = parse_tool_calls(raw_text)
    clean_text = strip_tool_markup(raw_text)
    clean_text = tokenizer.decode(
        tokenizer.encode(clean_text, add_special_tokens=False), skip_special_tokens=True
    )
    return clean_text, tool_calls

print('All helpers loaded. Ready to run.')
print('Judge:', JUDGE_PROVIDER)
# Smoke test the regex path only — no API call at import time.
assert _shared_classify_refusal('I cannot help with that.', judge=None)['refused'] is True
assert _shared_classify_refusal("Sure, here's how it works.", judge=None)['refused'] is False
assert _shared_classify_refusal('', judge=None)['method'] == 'no_text'
print('Classifier smoke test passed.')

# ================= Cell 6 · main eval loop (verbatim) =================
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
)

model     = None   # global — updated per model
tokenizer = None

for MODEL_ID in MODELS:
    MODEL_SHORT = MODEL_ID.split('/')[-1]
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f'results_{MODEL_SHORT}.csv')

    # ── Skip if already fully evaluated ────────────────────────────────
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        if len(existing) >= len(df):
            print(f'[SKIP] {MODEL_SHORT} — already complete ({len(existing)} rows)')
            continue

    start_time = datetime.now()
    print(f'\n{"="*65}')
    print(f'  START: {MODEL_SHORT}')
    print(f'  Time : {start_time.strftime("%H:%M:%S")}')
    print(f'{"="*65}')

    # ── Load model ─────────────────────────────────────────────────────
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True,
            token=HF_TOKEN if HF_TOKEN else None
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            device_map='auto',
            trust_remote_code=True,
            token=HF_TOKEN if HF_TOKEN else None
        )
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f'  Loaded on {model.device} | GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB')
    except Exception as e:
        print(f'  [ERROR] Failed to load {MODEL_SHORT}: {e}')
        print('  Skipping this model and continuing.')
        continue

    # ── Resume support ─────────────────────────────────────────────────
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)

        # Normalize old CSV ids and current dataset IDs to avoid int/string mismatch
        existing["id"] = existing["id"].apply(normalize_id)
        done_ids = set(existing["id"])

        df_run = df[~df["ID_STR"].isin(done_ids)].reset_index(drop=True)
        results = existing.to_dict("records")

        print(f'  Resuming: {len(done_ids)} done, {len(df_run)} remaining')
    else:
        df_run = df
        results = []

    # ── Eval loop ──────────────────────────────────────────────────────
    for _, row in tqdm(df_run.iterrows(), total=len(df_run), desc=MODEL_SHORT):
        domain      = get_domain(row['domain_key'])
        mode        = row['Mode']
        req_type    = row['Request Type']
        scenario_id = row['scenario_id']
        tools       = list(domain.tools_for_llm) if mode != 'No-tool chat' else None

        try:
            response_text, tool_calls = run_inference(
                system_prompt=row['System Prompt'],
                user_prompt=row['User Prompt'],
                tools=tools,
            )
            error = ''
        except Exception as e:
            response_text, tool_calls, error = '', [], str(e)

        refusal   = {'refused': False, 'method': 'error'} if error else classify_refusal(response_text)
        t_refused = refusal['refused']

        if mode != 'No-tool chat':
            violations = domain.score_tool_calls(scenario_id, tool_calls)
            tc_safe    = len(violations) == 0
            viols_json = json.dumps([v.as_dict() for v in violations])
        else:
            tc_safe, viols_json = True, '[]'

        results.append({
            'id':               row['ID'],
            'model':            MODEL_SHORT,
            'domain':           row['Domain'],
            'scenario':         row['Scenario'],
            'scenario_id':      scenario_id,
            'mode':             mode,
            'system_condition': row['System Condition'],
            'prompt_variant':   row['Prompt Variant'],
            'request_type':     req_type,
            'group_id':         row['group_id'],
            'response_text':    response_text,
            'tool_calls':       json.dumps(tool_calls),
            'refusal_method':   refusal['method'],
            'refusal_label':    refusal.get('label', 'error'),
            'refused':          t_refused,
            'tc_safe':          tc_safe,
            'divergence':       (req_type == 'Harmful') and t_refused and not tc_safe,
            'over_refusal':     (req_type == 'Benign') and t_refused,
            'violations':       viols_json,
            'error':            error,
        })

        if len(results) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    # Final save for this model
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    elapsed = (datetime.now() - start_time).seconds // 60
    print(f'  Saved {len(results)} rows → {OUTPUT_PATH}')
    print(f'  Time elapsed: {elapsed} min')

    # ── Unload model and free GPU memory ───────────────────────────────
    del model, tokenizer
    model, tokenizer = None, None
    gc.collect()
    torch.cuda.empty_cache()
    print(f'  GPU cleared. Moving to next model.')

print(f'\n{"="*65}')
print('  ALL MODELS COMPLETE')
print(f'{"="*65}')
