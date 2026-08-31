#!/usr/bin/env python3
"""Render a markdown handoff to PDF via LaTeX, so the PDF regenerates from the
markdown rather than being a second copy that drifts out of sync.

Handles the subset of markdown these documents use: ATX headings, pipe tables,
bold/italic/code spans, blockquotes, bullets, numbered items and rules. Wrapped
lines are joined into blocks first, so inline markup that straddles a newline
still converts, and table columns are sized to the text width so wide tables do
not run off the page.

  python3 make_handoff_pdf.py [source.md] [output-basename]
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "HANDOFF.md"
BASE = sys.argv[2] if len(sys.argv) > 2 else SRC.stem

ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
       "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
       "^": r"\textasciicircum{}"}
# applied AFTER escaping, so the LaTeX these insert is not itself escaped
UNI = [("→", r"$\rightarrow$"), ("←", r"$\leftarrow$"), ("—", "---"), ("–", "--"),
       ("≈", r"$\approx$"), ("×", r"$\times$"), ("±", r"$\pm$"), ("Δ", r"$\Delta$"),
       ("κ", r"$\kappa$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("…", r"\ldots{}"),
       ("§", r"\S{}"), ("“", "``"), ("”", "''"), ("‘", "`"), ("’", "'"),
       ("•", r"$\bullet$")]
TEXTW = 6.8   # inches available inside the margins


def inline(s):
    spans = []

    def stash(m):
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    s = re.sub(r"`([^`]+)`", stash, s)
    s = "".join(ESC.get(c, c) for c in s)
    for u, r in UNI:
        s = s.replace(u, r)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s, flags=re.S)
    s = re.sub(r"(?<![\w*])\*([^*]+?)\*(?![\w*])", r"\\emph{\1}", s, flags=re.S)

    def pop(m):
        raw = spans[int(m.group(1))]
        out = "".join(ESC.get(c, c) for c in raw)
        for u, r in UNI:
            out = out.replace(u, r)
        return r"\texttt{" + out + "}"

    return re.sub(r"\x00(\d+)\x00", pop, s)


def blockify(md):
    """Group lines into blocks, joining wrapped continuation lines."""
    lines = md.split("\n")
    blocks, i = [], 0
    while i < len(lines):
        ln = lines[i]
        st = ln.strip()
        if not st:
            i += 1
            continue
        if re.match(r"^\s*\|", ln) and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|?\s*$", lines[i + 1]):
            rows = [ln]
            i += 2
            while i < len(lines) and re.match(r"^\s*\|", lines[i]):
                rows.append(lines[i])
                i += 1
            blocks.append(("table", rows))
            continue
        if re.match(r"^#{1,4}\s", st):
            blocks.append(("head", st))
            i += 1
            continue
        if re.match(r"^---+$", st):
            blocks.append(("rule", ""))
            i += 1
            continue
        kind = None
        m = re.match(r"^\s*[-*]\s+(.*)$", ln)
        if m:
            kind, text = "item", m.group(1)
        else:
            m = re.match(r"^\s*(\d+)\.\s+(.*)$", ln)
            if m:
                kind, text = "enum", (m.group(1), m.group(2))
            elif st.startswith(">"):
                kind, text = "quote", st.lstrip("> ").strip()
            else:
                kind, text = "para", st
        i += 1
        # absorb wrapped continuation lines
        cont = []
        while i < len(lines):
            nxt = lines[i]
            s2 = nxt.strip()
            if not s2 or re.match(r"^#{1,4}\s", s2) or re.match(r"^---+$", s2) \
               or re.match(r"^\s*\|", nxt) or re.match(r"^\s*[-*]\s+", nxt) \
               or re.match(r"^\s*\d+\.\s+", nxt) or s2.startswith(">"):
                break
            cont.append(s2)
            i += 1
        joined = " ".join(cont)
        if kind == "enum":
            blocks.append(("enum", (text[0], (text[1] + " " + joined).strip())))
        else:
            blocks.append((kind, (text + " " + joined).strip()))
    return blocks


def render_table(rows):
    header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    body = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows[1:]]
    n = len(header)
    body = [(r + [""] * n)[:n] for r in body]
    widths = []
    for c in range(n):
        widths.append(max([len(header[c])] + [len(r[c]) for r in body]) or 1)
    total = sum(widths)
    # proportional p{} columns so long cells wrap instead of overflowing
    spec = "".join("p{%.2fin}" % max(0.55, TEXTW * w / total - 0.06) for w in widths)
    out = [r"\begin{center}\footnotesize", r"\begin{tabular}{" + spec + "}", r"\toprule",
           " & ".join(r"\textbf{" + inline(h) + "}" for h in header) + r" \\", r"\midrule"]
    for r in body:
        out.append(" & ".join(inline(c) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return out


def convert(md):
    out, listmode = [], None

    def close():
        nonlocal listmode
        if listmode:
            out.append(r"\end{itemize}")
            listmode = None

    for kind, payload in blockify(md):
        if kind in ("item", "enum"):
            if not listmode:
                out.append(r"\begin{itemize}\setlength{\itemsep}{1pt}\setlength{\parskip}{0pt}")
                listmode = True
            if kind == "enum":
                out.append(r"\item[" + payload[0] + ".] " + inline(payload[1]))
            else:
                out.append(r"\item " + inline(payload))
            continue
        close()
        if kind == "table":
            out += render_table(payload)
        elif kind == "head":
            m = re.match(r"^(#{1,4})\s+(.*)$", payload)
            lvl, title = len(m.group(1)), inline(m.group(2))
            cmd = {1: "section*", 2: "section*", 3: "subsection*", 4: "subsubsection*"}[lvl]
            out.append(f"\\{cmd}{{{title}}}")
        elif kind == "rule":
            out.append(r"\vspace{0.3em}\hrule\vspace{0.7em}")
        elif kind == "quote":
            out.append(r"\begin{quote}\itshape " + inline(payload) + r"\end{quote}")
        else:
            out.append(inline(payload))
            out.append("")
    close()
    return "\n".join(out)


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=0.85in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em}
\setlength{\emergencystretch}{3em}
\begin{document}
"""

tex = PREAMBLE + convert(SRC.read_text()) + "\n\\end{document}\n"
tex_path = HERE / f"_{BASE}.tex"
tex_path.write_text(tex)
r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory",
                    str(HERE), str(tex_path)], capture_output=True, text=True)
pdf = HERE / f"_{BASE}.pdf"
if pdf.exists():
    final = HERE / f"{BASE}.pdf"
    pdf.replace(final)
    for ext in (".aux", ".log", ".out", ".tex"):
        p = HERE / f"_{BASE}{ext}"
        if p.exists():
            p.unlink()
    over = [l for l in r.stdout.split("\n") if "Overfull" in l]
    print(f"wrote {final}" + (f"  ({len(over)} overfull boxes)" if over else ""))
else:
    print("pdflatex failed:\n" + "\n".join(
        [l for l in r.stdout.split("\n") if l.startswith("!")][:8]))
    sys.exit(1)
