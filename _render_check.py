"""Does the tool definition actually land in the rendered prompt for each family?"""
import sys
sys.path.insert(0, '.')
from transformers import AutoTokenizer
from tools.registry import get_domain

MODELS = [
    ("Qwen3-14B", "Qwen/Qwen3-14B"),
    ("Mistral-7B", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("gemma-3-27b-it", "google/gemma-3-27b-it"),
    ("Llama-3.1-70B", "NousResearch/Meta-Llama-3.1-70B-Instruct"),
]
tools = list(get_domain("finance").tools_for_llm)
probe = tools[0]["function"]["name"] if isinstance(tools[0], dict) and "function" in tools[0] \
    else (tools[0].get("name") if isinstance(tools[0], dict) else str(tools[0])[:40])
print(f"probing for tool name: {probe!r}  (n_tools={len(tools)})\n")

msgs = [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Submit a trade order for client 12345."}]

for name, hf in MODELS:
    try:
        tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    except Exception as e:
        print(f"{name:<16} tokenizer FAILED: {str(e)[:60]}")
        continue
    kw = dict(tokenize=False, add_generation_prompt=True)
    got = None
    for extra in ({"enable_thinking": False}, {}):
        try:
            got = tok.apply_chat_template(msgs, tools=tools, **kw, **extra)
            used = extra
            break
        except TypeError:
            continue
        except Exception as e:
            print(f"{name:<16} render raised {type(e).__name__}: {str(e)[:70]}")
            break
    if got is None:
        print(f"{name:<16} could not render with tools")
        continue
    has = probe in got
    print(f"{name:<16} tools_in_prompt={has!s:<6} len={len(got):<6} kwargs={used}")
    if not has:
        print(f"    !! rendered prompt lacks the tool name; first 200 chars: {got[:200]!r}")
