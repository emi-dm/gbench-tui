"""Leaderboard: the latest result of every (model, benchmark) across all runs, ranked.

A model's **gScore** is the mean of its percentage scores (Golden Set and every eval suite it has a score for). Two
models are only strictly comparable on the suites they share with the same --eval-limit, so the table shows the number
of samples next to each score, and `comparable_only` keeps only the results gbench itself marks leaderboard-comparable.
Written after every run as leaderboard.md and leaderboard.html (DeepMind palette) next to the results directory.
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path

from . import results as R
from .config import Settings


def build(runs: list, comparable_only: bool = False) -> dict:
    """{'models': [{'model', 'gscore', 'scores': {bench: Score}, 'runs', 'when', 'perf': {...}}], 'benches': [...]}"""
    table: dict = {}
    for run in sorted(runs, key=lambda r: r.started):          # later runs overwrite earlier ones
        if run.status not in ("ok", "unknown"):
            continue
        row = table.setdefault(run.model, {"model": run.model, "scores": {}, "runs": 0, "when": "", "endpoint": run.endpoint})
        row["runs"] += 1
        row["when"] = run.when
        for s in run.scores:
            if s.value is None:
                continue
            if comparable_only and s.kind == "eval" and s.comparable is False:
                continue
            row["scores"][s.name] = s
    benches_pct = sorted({b for r in table.values() for b, s in r["scores"].items() if s.unit == "%"},
                         key=lambda b: (b != "golden_set", b))
    benches_perf = sorted({b for r in table.values() for b, s in r["scores"].items() if s.unit != "%"})
    rows = []
    for r in table.values():
        pct = [s.value for s in r["scores"].values() if s.unit == "%"]
        r["gscore"] = sum(pct) / len(pct) if pct else None
        r["n_bench"] = len(pct)
        rows.append(r)
    rows.sort(key=lambda r: (r["gscore"] is None, -(r["gscore"] or 0)))
    return {"models": rows, "benches": benches_pct, "perf": benches_perf, "updated": time.strftime("%Y-%m-%d %H:%M")}


def _cell(s) -> str:
    if s is None:
        return "–"
    if s.unit == "%":
        return f"{s.value:.1f}" + (f" ({s.n})" if s.n else "")
    return f"{s.value:.2f} {s.unit}"


def to_markdown(lb: dict) -> str:
    benches, perf = lb["benches"], lb["perf"]
    head = ["#", "Model", "gScore", *benches, *perf]
    lines = ["# gbench-tui leaderboard", "",
             f"Updated {lb['updated']} · scores from [gbench](https://github.com/google-deepmind/gbench) runs, each in a "
             "Docker Sandbox · gScore = mean of the % scores · (n) = samples scored", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, r in enumerate(lb["models"], 1):
        g = f"**{r['gscore']:.1f}**" if r["gscore"] is not None else "–"
        lines.append("| " + " | ".join([str(i), r["model"], g, *[_cell(r["scores"].get(b)) for b in benches],
                                        *[_cell(r["scores"].get(b)) for b in perf]]) + " |")
    return "\n".join(lines) + "\n"


def to_html(lb: dict) -> str:
    benches, perf = lb["benches"], lb["perf"]
    e = html.escape
    rows = []
    for i, r in enumerate(lb["models"], 1):
        cells = []
        for b in benches:
            s = r["scores"].get(b)
            if s is None:
                cells.append("<td class=na>–</td>")
            else:
                cells.append(f"<td><div class=bar style='--v:{max(0, min(100, s.value)):.1f}%'></div>"
                             f"<span>{s.value:.1f}</span><small>{s.n or ''}</small></td>")
        for b in perf:
            s = r["scores"].get(b)
            cells.append(f"<td class=num>{_cell(s)}</td>" if s else "<td class=na>–</td>")
        g = f"{r['gscore']:.1f}" if r["gscore"] is not None else "–"
        rows.append(f"<tr><td class=rank>{i}</td><td class=model>{e(r['model'])}<small>{e(r['when'])}</small></td>"
                    f"<td class=g>{g}</td>{''.join(cells)}</tr>")
    head = "".join(f"<th>{e(b)}</th>" for b in [*benches, *perf])
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>gbench-tui leaderboard</title><style>
:root{{--bg:#0E1117;--card:#161B26;--ink:#E8EAED;--muted:#9AA0A6;--rule:#2A3040;--blue:#4285F4;--violet:#9177C7;--pink:#CA6673;--cyan:#24C1E0}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Google Sans",Inter,system-ui,sans-serif}}
header{{padding:28px 24px 8px}}h1{{margin:0;font-size:26px;font-weight:600;background:linear-gradient(90deg,var(--blue),var(--violet),var(--pink));-webkit-background-clip:text;background-clip:text;color:transparent}}
p{{color:var(--muted);margin:6px 0 0}}.wrap{{overflow-x:auto;padding:16px 24px 32px}}
table{{border-collapse:collapse;min-width:100%;background:var(--card);border-radius:12px;overflow:hidden}}
th,td{{padding:9px 12px;border-bottom:1px solid var(--rule);text-align:left;white-space:nowrap}}
th{{font-weight:500;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
td{{position:relative}}.bar{{position:absolute;left:0;top:0;bottom:0;width:var(--v);background:linear-gradient(90deg,#4285F433,#9177C744)}}
td span{{position:relative;font-variant-numeric:tabular-nums}}td small,.model small{{position:relative;color:var(--muted);margin-left:6px;font-size:11px}}
.model small{{display:block;margin:0}}.g{{font-weight:700;color:var(--cyan)}}.rank{{color:var(--muted)}}.na{{color:#5f6368}}.num{{font-variant-numeric:tabular-nums}}
</style></head><body><header><h1>gbench-tui leaderboard</h1>
<p>Updated {e(lb['updated'])} · results of <a style="color:var(--blue)" href="https://github.com/google-deepmind/gbench">gbench</a> runs, each inside a Docker Sandbox · gScore = mean of the % scores · small numbers = samples scored</p></header>
<div class=wrap><table><thead><tr><th>#</th><th>Model</th><th>gScore</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></body></html>"""


def write(settings: Settings, comparable_only: bool = False) -> dict:
    runs = R.list_runs(settings.results_path)
    lb = build(runs, comparable_only)
    base = settings.results_path.parent
    (base / "leaderboard.md").write_text(to_markdown(lb))
    (base / "leaderboard.html").write_text(to_html(lb))
    slim = {"updated": lb["updated"], "benches": lb["benches"], "perf": lb["perf"],
            "models": [{"model": r["model"], "gscore": r["gscore"], "when": r["when"],
                        "scores": {b: {"value": s.value, "unit": s.unit, "n": s.n} for b, s in r["scores"].items()}}
                       for r in lb["models"]]}
    (base / "leaderboard.json").write_text(json.dumps(slim, indent=1))
    return lb


def paths(settings: Settings) -> dict:
    base: Path = settings.results_path.parent
    return {k: base / f"leaderboard.{k}" for k in ("md", "html", "json")}
