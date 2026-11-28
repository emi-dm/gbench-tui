"""Comparative Markdown & HTML report generator (`gbench-tui report`) for GitHub, Reddit or HF model cards."""
from __future__ import annotations

import html
import time
from pathlib import Path

from . import compare as C, config, leaderboard, results as R, theme as T


def _fmt_num(v: float | None, sfx: str = "", prec: int = 1) -> str:
    return f"{v:.{prec}f}{sfx}" if v is not None else "–"


def build_markdown(runs: list[R.Run], title: str = "gbench-tui Evaluation Report") -> str:
    lb = leaderboard.build(runs)
    lines = [
        f"# {title}",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · {len(runs)} run(s) · {len(lb['models'])} variant(s)",
        "",
        "## Leaderboard Summary",
        "",
        leaderboard.to_markdown(lb).split("\n", 3)[-1].strip(),
        "",
    ]

    # Quantization degradation tables per base model family
    by_fam: dict[str, list[dict]] = {}
    for m in lb["models"]:
        if m.get("quant") and not m.get("harness"):
            by_fam.setdefault(m["base_model"], []).append(m)
    deg_fams = {k: v for k, v in by_fam.items() if len(v) >= 2}
    if deg_fams:
        lines += ["## Quantization Degradation", ""]
        for fam, members in deg_fams.items():
            members.sort(key=lambda m: (-(m.get("quant_bits") or 0), m["quant"]))
            ref = members[0]
            lines += [f"### {fam}", "", "| Quant | bpw | Adapter | gScore | Δ vs top | Decode tok/s | TTFT |",
                      "|---|---|---|---|---|---|---|"]
            for m in members:
                bits = _fmt_num(m.get("quant_bits"), "", 2)
                g = _fmt_num(m.get("gscore"), "%")
                dg = (f"{m['gscore'] - ref['gscore']:+.1f}"
                      if (m is not ref and m.get("gscore") is not None and ref.get("gscore") is not None) else "ref")
                spd = next((s.value for b, s in m["scores"].items() if "decode tok/s" in b.lower()), None)
                ttft = next((s.value for b, s in m["scores"].items() if "ttft" in b.lower()), None)
                lines.append(f"| {m['quant']} | {bits} | {m.get('adapter') or 'base'} | {g} | {dg} | "
                             f"{_fmt_num(spd, ' tok/s')} | {_fmt_num(ttft, ' s', 2)} |")
            lines.append("")

    # Failure taxonomy & Token Economy across recent runs
    diags = [(r, C.diagnose_run(r)) for r in runs[:12]]
    diags = [(r, d) for r, d in diags if d["total"] > 0]
    if diags:
        lines += [
            "## Failure Taxonomy & Token Economy", "",
            "| Run | Variant | Pass / Fail | Truncated | Tool/Fmt | Err | Wrong | Avg Tok (Pass / Fail) | gScore / 1k tok |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r, d in diags:
            cs = d["causes"]
            tok_pf = f"{_fmt_num(d['avg_tokens_pass'], '', 0)} / {_fmt_num(d['avg_tokens_fail'], '', 0)}"
            lines.append(
                f"| `{r.id[:28]}` | {r.model_label} | {d['passed']} / {d['failed']} | "
                f"{cs['truncated']} | {cs['tool_format']} | {cs['timeout_error']} | {cs['wrong_answer']} | "
                f"{tok_pf} | {_fmt_num(d['gscore_per_1k_tokens'], '', 2)} |"
            )
        lines.append("")

    # Top sample regressions between top-2 variants (if comparable runs exist)
    ok_runs = [r for r in runs if r.status == "ok" and R.items(r)]
    if len(ok_runs) >= 2:
        cmp = C.compare_runs(ok_runs[1], ok_runs[0])
        regs = [s for s in cmp.samples if s.verdict == "regression"]
        imps = [s for s in cmp.samples if s.verdict == "improvement"]
        if regs or imps:
            lines += [
                f"## Diff: `{cmp.run_a.model_label}` → `{cmp.run_b.model_label}`", "",
                f"- **Regressions (Pass → Fail):** {len(regs)} · **Improvements (Fail → Pass):** {len(imps)} · "
                f"**Both Pass:** {cmp.both_pass} · **Both Fail:** {cmp.both_fail}", "",
            ]
            for s in regs[:5]:
                detail = ((s.b or {}).get("detail") or "")[:140]
                lines.append(f"- ❌ `{s.suite}/{s.sample_id}` ({s.category}): {detail or 'failed in B'}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_html(runs: list[R.Run], title: str = "gbench-tui Evaluation Report", theme_name: str | None = None) -> str:
    md = build_markdown(runs, title)
    hx = T.theme_info(theme_name)["HEX"]
    e = html.escape
    # Simple, clean, offline HTML wrapper around the structured Markdown tables & headings
    body_parts: list[str] = []
    in_table = False
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= {"-", ":"} for c in cells if c):
                continue
            if not in_table:
                body_parts.append("<div class=tbl><table><thead><tr>" + "".join(f"<th>{e(c)}</th>" for c in cells) + "</tr></thead><tbody>")
                in_table = True
            else:
                body_parts.append("<tr>" + "".join(f"<td>{e(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            body_parts.append("</tbody></table></div>")
            in_table = False
        if line.startswith("### "):
            body_parts.append(f"<h3>{e(line[4:])}</h3>")
        elif line.startswith("## "):
            body_parts.append(f"<h2>{e(line[3:])}</h2>")
        elif line.startswith("# "):
            body_parts.append(f"<h1>{e(line[2:])}</h1>")
        elif line.startswith("- "):
            body_parts.append(f"<p class=li>• {e(line[2:])}</p>")
        elif line.strip():
            body_parts.append(f"<p>{e(line)}</p>")
    if in_table:
        body_parts.append("</tbody></table></div>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{e(title)}</title>
<style>
body{{margin:0;padding:32px;background:{hx['bg']};color:{hx['ink']};font:14px/1.6 system-ui,-apple-system,sans-serif;max-width:1180px;margin-inline:auto}}
h1{{background:linear-gradient(90deg,{hx['accent1']},{hx['accent2']},{hx['accent3']});-webkit-background-clip:text;color:transparent;margin-top:0}}
h2{{color:{hx['accent1']};border-bottom:1px solid {hx['rule']};padding-bottom:6px;margin-top:28px}}
h3{{color:{hx['accent2']};margin-top:18px}}
.tbl{{overflow-x:auto;border:1px solid {hx['rule']};border-radius:10px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;background:{hx['card']}}}
th,td{{padding:9px 12px;border-bottom:1px solid {hx['rule']};text-align:left;white-space:nowrap}}
th{{background:{hx['card2']};color:{hx['accent1']};font-size:11px;text-transform:uppercase;letter-spacing:.05em}}
p.li{{margin:4px 0 4px 12px;color:{hx['muted']}}}
</style></head><body>{''.join(body_parts)}</body></html>"""


def write_report(settings: config.Settings, out_dir: Path | None = None, runs: list[R.Run] | None = None,
                 title: str = "gbench-tui Evaluation Report") -> tuple[Path, Path]:
    target = (out_dir or settings.results_path.parent).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    run_list = runs if runs is not None else R.list_runs(settings.results_path)
    md_p = target / "report.md"
    html_p = target / "report.html"
    md_p.write_text(build_markdown(run_list, title), encoding="utf-8")
    html_p.write_text(build_html(run_list, title, getattr(settings, "theme", None)), encoding="utf-8")
    return md_p, html_p
