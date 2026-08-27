import sys
from transformers import AutoTokenizer

for name, hf in (("gemma-3-27b-it", "google/gemma-3-27b-it"),
                 ("Llama-3.1-70B", "NousResearch/Meta-Llama-3.1-70B-Instruct"),
                 ("Qwen3-14B", "Qwen/Qwen3-14B")):
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    t = tok.chat_template
    if isinstance(t, dict):
        print(f"{name}: chat_template is a DICT with keys {list(t)}")
        t = t.get("default") or next(iter(t.values()))
    if not t:
        print(f"{name}: NO chat_template")
        continue
    print(f"{name:<16} len={len(t):<6} mentions_tools={'tools' in t}  "
          f"has_tool_loop={'for tool in tools' in t or 'tools is not none' in t.lower()}")
