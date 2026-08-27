"""Guard: does the tool definition actually land in the rendered prompt?

apply_chat_template silently ignores tools= when a model's template has no
tool-handling block, which turns a tool-enabled prompt into a plain chat prompt
and makes every response score as "no unsafe tool call". Run this before any
tool-mode sweep.
"""
import sys
sys.path.insert(0, '.')
from transformers import AutoTokenizer
from tools.registry import get_domain

MODELS = [
    ("Qwen3-14B", "Qwen/Qwen3-14B"),
    ("Mistral-7B", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("Command-R-7B", "CohereLabs/c4ai-command-r7b-12-2024"),
    ("gemma-3-27b-it", "google/gemma-3-27b-it"),
    ("Llama-3.1-70B (nous)", "NousResearch/Meta-Llama-3.1-70B-Instruct"),
    ("Llama-3.1-70B (unsloth)", "unsloth/Meta-Llama-3.1-70B-Instruct"),
]
tools = list(get_domain("finance").tools_for_llm)
probe = tools[0]["function"]["name"]
print(f"probe tool name: {probe!r}  (n_tools={len(tools)})\n")

msgs = [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Submit a trade order for client 12345."}]

bad = []
for name, hf in MODELS:
    try:
        tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    except Exception as e:
        print(f"{name:<26} tokenizer FAILED: {type(e).__name__}")
        continue
    kw = dict(tokenize=False, add_generation_prompt=True)
    got = None
    for extra in ({"enable_thinking": False}, {}):
        try:
            got = tok.apply_chat_template(msgs, tools=tools, **kw, **extra)
            break
        except Exception:
            continue
    if got is None:
        print(f"{name:<26} could not render")
        bad.append(name)
        continue
    ok = probe in got
    tmpl = tok.chat_template
    tl = len(tmpl) if isinstance(tmpl, str) else 0
    print(f"{name:<26} tools_in_prompt={str(ok):<6} prompt_len={len(got):<6} template_len={tl}")
    if not ok:
        bad.append(name)

print()
print("OK - every template renders tools" if not bad
      else f"BROKEN (tools silently dropped): {', '.join(bad)}")
sys.exit(1 if bad else 0)
