#!/usr/bin/env python3
"""Prompt rendering that actually gets the tools into the prompt.

`apply_chat_template(..., tools=...)` silently ignores the tools when a model's
template has no tool-handling block — the variable is simply unused and nothing is
raised. A tool-enabled prompt then renders as a plain chat prompt, the model has
nothing to call, and every response scores as "made no unsafe tool call", i.e.
perfectly safe. Gemma-3 has no tool block in any mirror; the NousResearch Llama-3.1
reupload ships a stripped template with none either.

`make_renderer` probes the template once, and falls back to writing the schemas into
the prompt text when the template will not carry them. The fallback is checked
against the committed behavioural data: on Gemma it produces a 12.5% unsafe rate on
harmful tool-normal prompts, against 14.6% stored under the same global-scope scorer,
so it reconstructs the original tool-mode condition closely.
"""
import json

DM = {"Finance": "finance", "Healthcare": "healthcare",
      "Education / Academic Integrity": "education", "Legal / Compliance": "legal"}


def tool_block(tools):
    """The schemas as prompt text, for templates that cannot carry them."""
    lines = ["You have access to the following tools. To call one, emit a JSON object",
             'with "name" and "parameters" keys inside <tool_call></tool_call> tags.', ""]
    for t in tools:
        lines.append(json.dumps(t["function"] if "function" in t else t))
    return "\n".join(lines)


def _apply(tok, msgs, tools=None):
    kw = dict(tokenize=False, add_generation_prompt=True)
    for extra in ({"enable_thinking": False}, {}):
        try:
            return tok.apply_chat_template(msgs, tools=tools, **kw, **extra) if tools \
                else tok.apply_chat_template(msgs, **kw, **extra)
        except TypeError:
            continue
        except Exception:
            break
    # some templates reject a system role outright; fold it into the user turn
    if len(msgs) == 2 and msgs[0]["role"] == "system":
        merged = [{"role": "user", "content": f"{msgs[0]['content']}\n\n{msgs[1]['content']}"}]
        return tok.apply_chat_template(merged, **kw)
    return tok.apply_chat_template(msgs, **kw)


def make_renderer(tok, get_domain, log=print):
    """Return render(row) -> prompt string, choosing the strategy once up front."""
    probe_tools = list(get_domain("finance").tools_for_llm)
    probe_name = probe_tools[0]["function"]["name"]
    probe = _apply(tok, [{"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": "hi"}], tools=probe_tools)
    native = probe_name in (probe or "")
    log(f"[render] template carries tools natively: {native}"
        + ("" if native else "  -> injecting schemas into the prompt text"))

    def render(row):
        no_tools = row["mode"] == "No-tool chat"
        tools = None if no_tools else list(get_domain(DM[row["domain"]]).tools_for_llm)
        sysmsg = row["system_prompt"]
        if tools and not native:
            sysmsg = f"{sysmsg}\n\n{tool_block(tools)}"
            tools = None          # template cannot use them; they are in the text now
        msgs = [{"role": "system", "content": sysmsg},
                {"role": "user", "content": row["user_prompt"]}]
        return _apply(tok, msgs, tools=tools)

    render.native_tools = native
    return render
