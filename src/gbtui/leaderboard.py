"""Leaderboard: the latest result of every (model, quant, benchmark) across all runs, ranked.

A model's **gScore** is the mean of its percentage scores (Golden Set and every eval suite it has a score for). Two
models are only strictly comparable on the suites they share with the same --eval-limit, so the table shows the number
of samples next to each score, and `comparable_only` keeps only the results gbench itself marks leaderboard-comparable.
Different quantizations of the same base model (e.g. LFM3-1.7B Q4_K_M vs Q8_0) are tracked as distinct variants.
Written after every run as leaderboard.md, leaderboard.html (interactive charts + radar + polar + quant bars + table)
and leaderboard.json.
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path

from . import config, results as R, theme as T
from .config import Settings


def _quant_sort_key(q: str) -> tuple[float, str]:
    bits = config.quant_bits(q)
    return (bits if bits is not None else 999.0, q)


def build(runs: list, comparable_only: bool = False) -> dict:
    """{'models': [{'model', 'base_model', 'quant', 'quant_bits', 'gscore', 'scores': {...}, ...}],
        'benches': [...], 'perf': [...], 'quants': [...], 'families': [...], 'updated': ...}"""
    table: dict = {}
    for run in sorted(runs, key=lambda r: r.started):          # later runs overwrite earlier ones
        if run.status not in ("ok", "unknown"):
            continue
        kept = []
        for s in run.scores:
            if s.value is None:
                continue
            if comparable_only and s.kind == "eval" and s.comparable is not True:
                continue
            kept.append(s)
        if not kept:
            continue
        base = run.base_model
        quant = run.quant
        label = f"{base} ({quant})" if quant else run.model
        key = (base.lower(), quant)
        row = table.setdefault(
            key,
            {
                "model": label,
                "base_model": base,
                "quant": quant,
                "quant_bits": config.quant_bits(quant),
                "scores": {},
                "runs": 0,
                "when": "",
                "endpoint": run.endpoint,
            },
        )
        row["runs"] += 1
        row["when"] = run.when
        for s in kept:
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
    quants = sorted({r["quant"] for r in rows if r["quant"]}, key=_quant_sort_key)
    families = list(dict.fromkeys(r["base_model"] for r in rows))
    return {
        "models": rows,
        "benches": benches_pct,
        "perf": benches_perf,
        "quants": quants,
        "families": families,
        "updated": time.strftime("%Y-%m-%d %H:%M"),
    }


def _cell(s) -> str:
    if s is None:
        return "–"
    if s.unit == "%":
        return f"{s.value:.1f}" + (f" ({s.n})" if s.n else "")
    return f"{s.value:.2f} {s.unit}"


def _md_esc(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def to_markdown(lb: dict) -> str:
    benches, perf = lb["benches"], lb["perf"]
    has_q = bool(lb.get("quants"))
    head = ["#", "Model", *(["Quant"] if has_q else []), "gScore", *[_md_esc(b) for b in benches], *[_md_esc(b) for b in perf]]
    lines = ["# gbench-tui leaderboard", "",
             f"Updated {lb['updated']} · scores from [gbench](https://github.com/google-deepmind/gbench) runs, each in a "
             "Docker Sandbox · gScore = mean of the % scores · (n) = samples scored", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, r in enumerate(lb["models"], 1):
        g = f"**{r['gscore']:.1f}**" if r["gscore"] is not None else "–"
        name_col = r.get("base_model") or r["model"] if has_q else r["model"]
        q_col = [_md_esc(r.get("quant") or "–")] if has_q else []
        lines.append("| " + " | ".join([str(i), _md_esc(name_col), *q_col, g,
                                        *[_md_esc(_cell(r["scores"].get(b))) for b in benches],
                                        *[_md_esc(_cell(r["scores"].get(b))) for b in perf]]) + " |")
    return "\n".join(lines) + "\n"


def _payload(lb: dict) -> dict:
    return {
        "updated": lb["updated"],
        "benches": lb["benches"],
        "perf": lb["perf"],
        "quants": lb.get("quants", []),
        "families": lb.get("families", []),
        "models": [
            {
                "model": r["model"],
                "base_model": r.get("base_model") or r["model"],
                "quant": r.get("quant", ""),
                "quant_bits": r.get("quant_bits"),
                "gscore": round(r["gscore"], 2) if r["gscore"] is not None else None,
                "when": r["when"],
                "runs": r.get("runs", 0),
                "n_bench": r.get("n_bench", 0),
                "scores": {
                    b: {"value": s.value, "unit": s.unit, "n": s.n}
                    for b, s in r["scores"].items()
                },
            }
            for r in lb["models"]
        ],
    }


def _safe_json_for_script(obj: object) -> str:
    """Serialize JSON safely for inline <script> blocks (escapes </script> sequences)."""
    return json.dumps(obj, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _quant_tone(bits: float | None) -> str:
    if bits is None:
        return "q-unk"
    if bits <= 3.95:
        return "q-low"
    if bits <= 5.2:
        return "q-q4"
    if bits <= 7.0:
        return "q-q6"
    if bits <= 9.0:
        return "q-q8"
    return "q-fp"


def to_html(lb: dict, theme_name: str | None = None) -> str:
    benches, perf = lb["benches"], lb["perf"]
    quants = lb.get("quants") or []
    families = lb.get("families") or []
    e = html.escape
    active_theme = theme_name if theme_name in T.THEMES else T.current_theme()
    themes_hex = {k: T.THEMES[k]["HEX"] for k in T.THEME_ORDER}
    hx = themes_hex[active_theme]

    rows = []
    for i, r in enumerate(lb["models"], 1):
        cells = []
        for b in benches:
            s = r["scores"].get(b)
            if s is None:
                cells.append("<td class=na data-v=-1>–</td>")
            else:
                v = max(0.0, min(100.0, s.value))
                cells.append(
                    f"<td data-v='{s.value:.4f}'><div class=bar style='--v:{v:.1f}%'></div>"
                    f"<span>{s.value:.1f}</span><small>{s.n or ''}</small></td>"
                )
        for b in perf:
            s = r["scores"].get(b)
            if s is None:
                cells.append("<td class=na data-v=-1>–</td>")
            else:
                cells.append(f"<td class=num data-v='{s.value:.4f}'>{e(_cell(s))}</td>")
        g = f"{r['gscore']:.1f}" if r["gscore"] is not None else "–"
        gv = f"{r['gscore']:.4f}" if r["gscore"] is not None else "-1"
        badge_cls = f"rank r{i}" if i <= 3 else "rank"
        base = r.get("base_model") or r["model"]
        q = r.get("quant") or ""
        q_bits = r.get("quant_bits")
        qv = f"{q_bits:.2f}" if q_bits is not None else "-1"
        q_cell = (f"<td data-v='{qv}'><span class='qbadge {_quant_tone(q_bits)}'>{e(q)}</span></td>"
                  if q else "<td class=na data-v=-1>–</td>")
        rows.append(
            f"<tr data-model='{e(r['model'])}' data-base='{e(base)}' data-quant='{e(q)}'>"
            f"<td class='{badge_cls}' data-v='{i}'><span>{i}</span></td>"
            f"<td class=model data-v='{e(base)}'>{e(base)}<small>{e(r['when'])}</small></td>"
            f"{q_cell}"
            f"<td class=g data-v='{gv}'>{g}</td>{''.join(cells)}</tr>"
        )
    head = "".join(f"<th data-col='{idx}'>{e(b)}</th>" for idx, b in enumerate([*benches, *perf], start=4))
    top_m = lb["models"][0] if lb["models"] else None
    top_g = f"{top_m['gscore']:.1f}" if (top_m and top_m["gscore"] is not None) else "–"
    top_name = e(top_m["model"]) if top_m else "—"
    data_js = _safe_json_for_script(_payload(lb))
    themes_js = _safe_json_for_script(themes_hex)
    theme_btns = "".join(
        f"<button class='tbtn{' active' if k == active_theme else ''}' data-theme='{k}'>{e(T.THEMES[k]['label'].split()[-1])}</button>"
        for k in T.THEME_ORDER
    )
    models_hint = (f"{len(families)} base · {len(quants)} quant(s)" if quants else "click chips to filter charts")
    dims_hint = f"{len(perf)} perf · {len(quants)} quant(s)" if quants else f"{len(perf)} performance metric(s)"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>gbench-tui leaderboard</title>
<style>
:root{{
  --bg:{hx['bg']};--card:{hx['card']};--card2:{hx['card2']};--ink:{hx['ink']};--muted:{hx['muted']};
  --rule:{hx['rule']};--a1:{hx['accent1']};--a2:{hx['accent2']};--a3:{hx['accent3']};
  --good:{hx['good']};--mid:{hx['mid']};--warn:{hx['warn']};--bad:{hx['bad']};
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 "Google Sans",Inter,system-ui,-apple-system,sans-serif;min-height:100vh}}
header{{padding:26px 28px 18px;border-bottom:1px solid var(--rule);
  background:radial-gradient(900px 260px at 8% -20%,color-mix(in srgb,var(--a1) 20%,transparent),transparent),
             radial-gradient(700px 240px at 88% -10%,color-mix(in srgb,var(--a2) 18%,transparent),transparent),var(--bg)}}
.head-top{{display:flex;flex-wrap:wrap;gap:14px;align-items:center;justify-content:space-between}}
h1{{margin:0;font-size:28px;font-weight:700;letter-spacing:-.025em;
  background:linear-gradient(90deg,var(--a1),var(--a2),var(--a3));-webkit-background-clip:text;background-clip:text;color:transparent}}
.sub{{color:var(--muted);margin:6px 0 0;font-size:13px}}
.sub a{{color:var(--a1);text-decoration:none}}.sub a:hover{{text-decoration:underline}}
.themes{{display:flex;gap:6px;flex-wrap:wrap;background:var(--card);padding:4px;border-radius:10px;border:1px solid var(--rule)}}
.tbtn{{border:0;background:transparent;color:var(--muted);padding:5px 11px;border-radius:7px;font:600 12px/1 inherit;cursor:pointer;transition:.15s}}
.tbtn:hover{{color:var(--ink)}}.tbtn.active{{background:linear-gradient(90deg,var(--a1),var(--a2));color:#fff}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:18px}}
.kpi{{background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:12px 15px;position:relative;overflow:hidden}}
.kpi::before{{content:"";position:absolute;left:0;top:0;right:0;height:2px;background:linear-gradient(90deg,var(--a1),var(--a2),var(--a3))}}
.kpi .lbl{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600}}
.kpi .val{{font-size:22px;font-weight:700;margin-top:3px;font-variant-numeric:tabular-nums}}
.kpi .hint{{font-size:12px;color:var(--muted);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
main{{padding:20px 28px 48px;max-width:1440px;margin:0 auto}}
.filters{{display:none;flex-wrap:wrap;gap:14px;align-items:center;justify-content:space-between;background:var(--card);
  border:1px solid var(--rule);border-radius:14px;padding:10px 16px;margin-bottom:18px}}
.filters.show{{display:flex}}
.fgroup{{display:flex;flex-wrap:wrap;align-items:center;gap:6px}}
.flbl{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-right:4px}}
.fpill{{border:1px solid var(--rule);background:var(--card2);color:var(--muted);padding:4px 10px;border-radius:999px;
  font:600 12px/1.1 inherit;cursor:pointer;transition:.15s}}
.fpill:hover{{color:var(--ink);border-color:var(--a2)}}
.fpill.active{{background:color-mix(in srgb,var(--a1) 24%,var(--card2));color:var(--ink);border-color:var(--a1)}}
.charts{{display:grid;grid-template-columns:minmax(340px,1fr) minmax(420px,1.25fr);gap:18px;margin-bottom:22px}}
@media(max-width:940px){{.charts{{grid-template-columns:1fr}}}}
.panel{{background:var(--card);border:1px solid var(--rule);border-radius:16px;padding:18px 20px;box-shadow:0 8px 30px rgba(0,0,0,.18);display:flex;flex-direction:column}}
.panel-head{{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px}}
.panel-title{{font-size:15px;font-weight:700;display:flex;align-items:center;gap:8px}}
.panel-title .dot{{width:8px;height:8px;border-radius:50%;background:linear-gradient(135deg,var(--a1),var(--a3))}}
.tabs{{display:flex;gap:4px;background:var(--card2);padding:3px;border-radius:9px;border:1px solid var(--rule)}}
.tab{{border:0;background:transparent;color:var(--muted);padding:5px 10px;border-radius:6px;font:600 12px/1 inherit;cursor:pointer;transition:.15s}}
.tab:hover{{color:var(--ink)}}.tab.active{{background:var(--a1);color:#fff}}
.toggle{{display:inline-flex;align-items:center;gap:5px;font-size:12px;color:var(--muted);cursor:pointer;user-select:none}}
.toggle input{{accent-color:var(--a2);cursor:pointer}}
.svg-wrap{{flex:1;display:flex;align-items:center;justify-content:center;min-height:310px;position:relative}}
svg{{width:100%;height:auto;max-height:390px;overflow:visible}}
.legend{{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;padding-top:10px;border-top:1px solid var(--rule)}}
.chip{{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;background:var(--card2);border:1px solid var(--rule);
  font-size:12px;font-weight:600;color:var(--ink);cursor:pointer;transition:.15s;user-select:none}}
.chip .sw{{width:9px;height:9px;border-radius:50%;flex:none}}
.chip.off{{opacity:.38;text-decoration:line-through}}
.chip.sel{{border-color:var(--a1);background:color-mix(in srgb,var(--a1) 18%,var(--card2));box-shadow:0 0 0 1px var(--a1)}}
.chip:hover{{border-color:var(--a2);transform:translateY(-1px)}}
.qbadge{{display:inline-block;padding:2px 7px;border-radius:6px;font-size:11px;font-weight:700;letter-spacing:.02em;font-variant-numeric:tabular-nums}}
.chip .qbadge{{padding:1px 6px;font-size:10px;margin-left:2px}}
.q-low{{background:rgba(251,113,133,.18);color:#FB7185;border:1px solid rgba(251,113,133,.35)}}
.q-q4{{background:rgba(251,191,36,.16);color:#FBBF24;border:1px solid rgba(251,191,36,.35)}}
.q-q6{{background:rgba(56,189,248,.16);color:#38BDF8;border:1px solid rgba(56,189,248,.35)}}
.q-q8{{background:rgba(192,132,252,.18);color:#C084FC;border:1px solid rgba(192,132,252,.35)}}
.q-fp{{background:rgba(52,211,153,.18);color:#34D399;border:1px solid rgba(52,211,153,.35)}}
.q-unk{{background:var(--card2);color:var(--muted);border:1px solid var(--rule)}}
.bar-rect{{transition:opacity .18s,filter .18s;cursor:pointer}}
.bar-rect:hover{{filter:brightness(1.18)}}
.radar-poly{{transition:fill-opacity .18s,stroke-width .18s,opacity .18s;cursor:pointer}}
.radar-dot{{transition:r .15s;cursor:pointer}}
.table-head{{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px}}
.search{{background:var(--card2);color:var(--ink);border:1px solid var(--rule);border-radius:9px;padding:7px 12px;font:13px inherit;min-width:220px;outline:none}}
.search:focus{{border-color:var(--a1)}}
.tbl-wrap{{overflow-x:auto;border-radius:12px;border:1px solid var(--rule)}}
table{{border-collapse:collapse;min-width:100%;background:var(--card)}}
th,td{{padding:11px 14px;border-bottom:1px solid var(--rule);text-align:left;white-space:nowrap}}
tbody tr{{transition:background .15s ease;cursor:pointer}}
tbody tr:hover,tbody tr.hl{{background:var(--card2)}}
th{{font-weight:600;color:var(--a1);font-size:11px;text-transform:uppercase;letter-spacing:.06em;background:var(--card2);cursor:pointer;user-select:none}}
th:hover{{color:var(--ink)}}
td{{position:relative}}.bar{{position:absolute;left:0;top:0;bottom:0;width:var(--v);
  background:linear-gradient(90deg,color-mix(in srgb,var(--a1) 24%,transparent),color-mix(in srgb,var(--a2) 22%,transparent),color-mix(in srgb,var(--a3) 18%,transparent))}}
td span{{position:relative;font-variant-numeric:tabular-nums}}
td small,.model small{{position:relative;color:var(--muted);margin-left:6px;font-size:11px}}
.model{{font-weight:600}}.model small{{display:block;margin:0;font-weight:400}}
.g{{font-weight:700;color:var(--a2);font-size:15px}}
.rank span{{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:7px;font-weight:700;font-size:12px;color:var(--muted)}}
.r1 span{{background:#FBBF24;color:#111}}.r2 span{{background:#CBD5E1;color:#111}}.r3 span{{background:#FDBA74;color:#111}}
.na{{color:var(--muted);opacity:.45}}.num{{font-variant-numeric:tabular-nums;color:var(--a1)}}
#tip{{position:fixed;pointer-events:none;z-index:99;opacity:0;transition:opacity .12s;
  background:color-mix(in srgb,var(--card2) 94%,black);color:var(--ink);border:1px solid var(--a2);
  border-radius:10px;padding:8px 11px;font-size:12px;box-shadow:0 10px 28px rgba(0,0,0,.35);backdrop-filter:blur(8px)}}
#tip strong{{display:block;margin-bottom:2px}}#tip span{{color:var(--muted)}}
.empty{{color:var(--muted);font-size:13px;padding:24px;text-align:center}}
</style>
</head>
<body>
<header>
  <div class="head-top">
    <div>
      <h1>gbench-tui leaderboard</h1>
      <p class="sub">Updated {e(lb['updated'])} · results of <a href="https://github.com/google-deepmind/gbench">gbench</a> runs inside Docker Sandboxes · gScore = mean of % scores</p>
    </div>
    <div class="themes" id="themeBar">{theme_btns}</div>
  </div>
  <div class="kpis">
    <div class="kpi"><div class="lbl">Variants Ranked</div><div class="val">{len(lb['models'])}</div><div class="hint">{e(models_hint)}</div></div>
    <div class="kpi"><div class="lbl">Leader gScore</div><div class="val" style="color:var(--good)">{top_g}</div><div class="hint">{top_name}</div></div>
    <div class="kpi"><div class="lbl">Eval Dimensions</div><div class="val">{len(benches)}</div><div class="hint">{e(dims_hint)}</div></div>
    <div class="kpi"><div class="lbl">Best Throughput</div><div class="val" id="kpiSpeed" style="color:var(--a1)">–</div><div class="hint" id="kpiSpeedSub">decode tok/s</div></div>
  </div>
</header>

<main>
  <section class="filters" id="filterBar">
    <div class="fgroup" id="familyFilters"></div>
    <div class="fgroup" id="quantFilters"></div>
  </section>

  <section class="charts">
    <div class="panel">
      <div class="panel-head">
        <div class="panel-title"><span class="dot"></span>Area Chart · Dimensions</div>
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
          <div class="tabs" id="areaTabs">
            <button class="tab active" data-mode="radar">Radar</button>
            <button class="tab" data-mode="polar">Polar Area</button>
          </div>
          <label class="toggle" id="perfToggleWrap" style="display:none"><input type="checkbox" id="incPerf"> perf</label>
          <label class="toggle" id="quantToggleWrap" style="display:none"><input type="checkbox" id="incQuant"> quant</label>
        </div>
      </div>
      <div class="svg-wrap" id="radarWrap"></div>
      <div class="legend" id="radarLegend"></div>
    </div>
    <div class="panel">
      <div class="panel-head">
        <div class="panel-title"><span class="dot"></span>Bar Charts</div>
        <div class="tabs" id="barTabs">
          <button class="tab active" data-mode="grouped">Benchmarks</button>
          <button class="tab" data-mode="quant" id="quantTabBtn" style="display:none">By Quant</button>
          <button class="tab" data-mode="gscore">gScore</button>
          <button class="tab" data-mode="perf" id="perfTabBtn" style="display:none">Speed / TTFT</button>
        </div>
      </div>
      <div class="svg-wrap" id="barWrap"></div>
      <div class="legend" id="barLegend"></div>
    </div>
  </section>

  <section class="panel">
    <div class="table-head">
      <div class="panel-title"><span class="dot"></span>Full Score Matrix</div>
      <input class="search" id="searchBox" type="search" placeholder="Filter model or quant…">
    </div>
    <div class="tbl-wrap" id="lbWrap">
      <table>
        <thead><tr><th data-col="0">#</th><th data-col="1">Model</th><th data-col="2">Quant</th><th data-col="3">gScore</th>{head}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
  </section>
</main>
<div id="tip"></div>

<script>
const DATA = {data_js};
const THEMES = {themes_js};
const SERIES_PALETTE = ["#4F7CFF","#F472B6","#34D399","#FBBF24","#38BDF8","#C084FC","#FB7185","#2DD4BF","#FB923C","#A3E635"];
const state = {{
  hidden: new Set(), hover: null, focus: null,
  family: "all", quant: "all", search: "",
  areaMode: "radar", barMode: "grouped",
  includePerf: DATA.benches.length < 3 && DATA.perf.length > 0,
  includeQuant: DATA.quants.length >= 2 && DATA.benches.length <= 4,
}};

const $ = s => document.querySelector(s);
const tip = $("#tip");
function showTip(ev, title, body) {{
  tip.innerHTML = `<strong>${{esc(title)}}</strong><span>${{esc(body)}}</span>`;
  tip.style.opacity = "1";
  tip.style.left = Math.min(window.innerWidth - 280, ev.clientX + 14) + "px";
  tip.style.top = Math.min(window.innerHeight - 80, ev.clientY + 14) + "px";
}}
function hideTip() {{ tip.style.opacity = "0"; }}
function esc(s) {{ return String(s).replace(/[&<>"']/g, m => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[m])); }}
function mColor(i) {{ return SERIES_PALETTE[i % SERIES_PALETTE.length]; }}
function qTone(bits) {{
  if (bits == null) return "q-unk";
  if (bits <= 3.95) return "q-low";
  if (bits <= 5.2) return "q-q4";
  if (bits <= 7.0) return "q-q6";
  if (bits <= 9.0) return "q-q8";
  return "q-fp";
}}
function qColor(bits, idx) {{
  if (bits == null) return SERIES_PALETTE[idx % SERIES_PALETTE.length];
  if (bits <= 3.95) return "#FB7185";
  if (bits <= 5.2) return "#FBBF24";
  if (bits <= 7.0) return "#38BDF8";
  if (bits <= 9.0) return "#C084FC";
  return "#34D399";
}}
function compactPct(bits) {{
  if (bits == null || bits <= 0) return 0;
  return Math.max(10, Math.min(100, 100 - ((Math.log2(bits) - 1) / 4) * 90));
}}
function shortLbl(s) {{
  const m = s.match(/^(.+?) \\(in(\\d+)\\)$/);
  if (m) return ({{"decode tok/s":"tok/s","output tok/s":"out t/s"}}[m[1]] || m[1]) + "·" + m[2];
  return s.length > 15 ? s.slice(0, 14) + "…" : s;
}}
function matchesFilters(m) {{
  if (state.family !== "all" && m.base_model !== state.family) return false;
  if (state.quant !== "all" && (m.quant || "") !== state.quant) return false;
  return true;
}}
function filteredModels() {{
  return DATA.models.map((m, idx) => ({{ ...m, idx }})).filter(matchesFilters);
}}
function visibleModels() {{
  return filteredModels().filter(m => !state.hidden.has(m.model));
}}

// Initialize KPI + optional controls
(function() {{
  const speedKey = DATA.perf.find(k => k.toLowerCase().includes("tok/s"));
  if (speedKey) {{
    let best = null;
    for (const m of DATA.models) {{
      const s = m.scores[speedKey];
      if (s && (!best || s.value > best.v)) best = {{ v: s.value, m: m.model, u: s.unit }};
    }}
    if (best) {{
      $("#kpiSpeed").textContent = best.v.toFixed(1) + " " + best.u;
      $("#kpiSpeedSub").textContent = best.m + " · " + speedKey;
    }}
  }}
  if (DATA.perf.length > 0) {{
    $("#perfTabBtn").style.display = "";
    $("#perfToggleWrap").style.display = "inline-flex";
    $("#incPerf").checked = state.includePerf;
  }}
  if (DATA.quants.length > 0) {{
    $("#quantTabBtn").style.display = "";
    $("#quantToggleWrap").style.display = "inline-flex";
    $("#incQuant").checked = state.includeQuant;
  }}
  if (DATA.quants.length > 0 || DATA.families.length > 1) {{
    $("#filterBar").classList.add("show");
  }}
}})();

function renderFilters() {{
  const fb = $("#filterBar");
  if (!fb.classList.contains("show")) return;
  const fams = DATA.families;
  $("#familyFilters").innerHTML = fams.length > 1
    ? `<span class="flbl">Model</span>`
      + `<button class="fpill${{state.family === 'all' ? ' active' : ''}}" data-fam="all">All (${{DATA.models.length}})</button>`
      + fams.map(f => `<button class="fpill${{state.family === f ? ' active' : ''}}" data-fam="${{esc(f)}}">${{esc(f)}}</button>`).join("")
    : "";
  const qs = DATA.quants;
  $("#quantFilters").innerHTML = qs.length > 0
    ? `<span class="flbl">Quant</span>`
      + `<button class="fpill${{state.quant === 'all' ? ' active' : ''}}" data-q="all">All</button>`
      + qs.map(q => `<button class="fpill${{state.quant === q ? ' active' : ''}}" data-q="${{esc(q)}}">${{esc(q)}}</button>`).join("")
    : "";
}}

function buildDimensions() {{
  const dims = DATA.benches.map(b => ({{ key: b, label: shortLbl(b), kind: "pct", raw: m => m.scores[b]?.value ?? null,
    norm: m => m.scores[b]?.value ?? 0, fmt: m => m.scores[b] ? `${{m.scores[b].value.toFixed(1)}}% (n=${{m.scores[b].n || "?"}})` : "–" }}));
  if (state.includePerf) {{
    for (const p of DATA.perf) {{
      const vals = DATA.models.map(m => m.scores[p]?.value).filter(v => v != null && v > 0);
      if (!vals.length) continue;
      const isLat = p.toLowerCase().includes("ttft");
      const mx = Math.max(...vals), mn = Math.min(...vals);
      dims.push({{
        key: p, label: shortLbl(p), kind: "perf",
        raw: m => m.scores[p]?.value ?? null,
        norm: m => {{
          const v = m.scores[p]?.value;
          if (v == null || v <= 0) return 0;
          return isLat ? Math.min(100, (mn / v) * 100) : Math.min(100, (v / mx) * 100);
        }},
        fmt: m => m.scores[p] ? `${{m.scores[p].value.toFixed(2)}} ${{m.scores[p].unit}}` : "–",
      }});
    }}
  }}
  if (state.includeQuant && DATA.quants.length > 0) {{
    dims.push({{
      key: "quant_efficiency", label: "compact (bits)", kind: "quant",
      raw: m => m.quant_bits,
      norm: m => compactPct(m.quant_bits),
      fmt: m => m.quant
        ? `${{m.quant}}${{m.quant_bits != null ? ' (~' + m.quant_bits.toFixed(1) + ' bpw · ' + compactPct(m.quant_bits).toFixed(0) + '% compact)' : ''}}`
        : "unspecified",
    }});
  }}
  return dims;
}}

function renderRadar() {{
  const wrap = $("#radarWrap");
  const visible = visibleModels();
  const dims = buildDimensions();
  if (!DATA.models.length || !dims.length || !visible.length) {{
    wrap.innerHTML = `<div class="empty">No matching models or dimensions to plot.</div>`;
    return;
  }}
  const W = 440, H = 360, cx = W / 2, cy = H / 2 + 4, R = 124;
  const isPolar = state.areaMode === "polar" && visible.length > 0;
  const innerR = isPolar ? 35 : 0;
  const rOf = pct => innerR + ((R - innerR) * pct) / 100;
  const parts = [`<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="Model radar area chart">`];
  parts.push(`<defs><radialGradient id="rg"><stop offset="0%" stop-color="var(--a2)" stop-opacity="0.16"/><stop offset="100%" stop-color="var(--a1)" stop-opacity="0"/></radialGradient></defs>`);
  parts.push(`<circle cx="${{cx}}" cy="${{cy}}" r="${{R + 6}}" fill="url(#rg)"/>`);

  const lblAng = -Math.PI / 2 + 0.34;
  const ringLabels = [];
  for (const pct of [25, 50, 75, 100]) {{
    const r = rOf(pct);
    parts.push(`<circle cx="${{cx}}" cy="${{cy}}" r="${{r.toFixed(1)}}" fill="none" stroke="var(--rule)" stroke-dasharray="${{pct === 100 ? '' : '3 3'}}" stroke-width="${{pct === 100 ? 1.4 : 1}}"/>`);
    const tx = cx + r * Math.cos(lblAng), ty = cy + r * Math.sin(lblAng);
    ringLabels.push(`<text x="${{tx.toFixed(1)}}" y="${{(ty + 3).toFixed(1)}}" fill="var(--muted)" font-size="9.5" paint-order="stroke" stroke="var(--card)" stroke-width="3" stroke-linejoin="round">${{pct}}</text>`);
  }}

  if (isPolar) {{
    const focus = visible.find(m => m.model === (state.hover || state.focus)) || visible[0];
    const n = dims.length;
    const step = (2 * Math.PI) / n;
    dims.forEach((d, i) => {{
      const a0 = -Math.PI / 2 + i * step + 0.03, a1 = -Math.PI / 2 + (i + 1) * step - 0.03;
      const v = Math.max(2, d.norm(focus)), r = rOf(v);
      const x1 = cx + r * Math.cos(a0), y1 = cy + r * Math.sin(a0);
      const x2 = cx + r * Math.cos(a1), y2 = cy + r * Math.sin(a1);
      const large = (a1 - a0) > Math.PI ? 1 : 0;
      const col = SERIES_PALETTE[i % SERIES_PALETTE.length];
      const path = `M ${{cx}} ${{cy}} L ${{x1.toFixed(1)}} ${{y1.toFixed(1)}} A ${{r.toFixed(1)}} ${{r.toFixed(1)}} 0 ${{large}} 1 ${{x2.toFixed(1)}} ${{y2.toFixed(1)}} Z`;
      parts.push(`<path class="radar-poly" data-m="${{esc(focus.model)}}" d="${{path}}" fill="${{col}}" fill-opacity="0.46" stroke="${{col}}" stroke-width="2" data-tip-t="${{esc(focus.model + ' · ' + d.key)}}" data-tip-b="${{esc(d.fmt(focus))}}"/>`);
      const mid = (a0 + a1) / 2, lx = cx + (R + 22) * Math.cos(mid), ly = cy + (R + 18) * Math.sin(mid);
      const anchor = Math.cos(mid) > 0.25 ? "start" : Math.cos(mid) < -0.25 ? "end" : "middle";
      parts.push(`<text x="${{lx.toFixed(1)}}" y="${{(ly + 4).toFixed(1)}}" text-anchor="${{anchor}}" fill="var(--ink)" font-size="11.5" font-weight="600">${{esc(d.label)}}</text>`);
    }});
    parts.push(...ringLabels);
    const shortM = focus.base_model.length > 11 ? focus.base_model.slice(0, 10) + "…" : focus.base_model;
    parts.push(`<circle cx="${{cx}}" cy="${{cy}}" r="${{innerR}}" fill="var(--card)" stroke="var(--rule)" stroke-width="1.4"/>`);
    parts.push(`<text x="${{cx}}" y="${{focus.quant ? cy - 7 : cy - 1}}" text-anchor="middle" fill="var(--ink)" font-size="13" font-weight="700">${{focus.gscore != null ? focus.gscore.toFixed(1) : '–'}}</text>`);
    parts.push(`<text x="${{cx}}" y="${{focus.quant ? cy + 6 : cy + 12}}" text-anchor="middle" fill="var(--muted)" font-size="8.5">${{esc(shortM)}}</text>`);
    if (focus.quant) {{
      parts.push(`<text x="${{cx}}" y="${{cy + 18}}" text-anchor="middle" fill="${{qColor(focus.quant_bits, focus.idx)}}" font-size="8.5" font-weight="700">${{esc(focus.quant)}}</text>`);
    }}
  }} else if (dims.length < 3) {{
    const d0 = dims[0];
    const gap = Math.min(22, (R - 26) / Math.max(1, visible.length));
    visible.forEach((m, k) => {{
      const v = d0.norm(m), r = R - 10 - k * gap, circ = 2 * Math.PI * r;
      const col = mColor(m.idx);
      const dim = state.hover && state.hover !== m.model ? 0.25 : 1;
      parts.push(`<circle cx="${{cx}}" cy="${{cy}}" r="${{r.toFixed(1)}}" fill="none" stroke="var(--rule)" stroke-width="10" opacity="0.35"/>`);
      parts.push(`<circle class="radar-poly" data-m="${{esc(m.model)}}" cx="${{cx}}" cy="${{cy}}" r="${{r.toFixed(1)}}" fill="none" stroke="${{col}}" stroke-width="10" stroke-linecap="round" stroke-dasharray="${{(circ * v / 100).toFixed(1)}} ${{circ.toFixed(1)}}" transform="rotate(-90 ${{cx}} ${{cy}})" opacity="${{dim}}" data-tip-t="${{esc(m.model)}}" data-tip-b="${{esc(d0.key + ': ' + d0.fmt(m))}}"/>`);
    }});
    parts.push(...ringLabels);
    parts.push(`<text x="${{cx}}" y="${{cy + 4}}" text-anchor="middle" fill="var(--muted)" font-size="12" font-weight="600">${{esc(d0.label)}}</text>`);
  }} else {{
    const n = dims.length;
    const angle = i => -Math.PI / 2 + (2 * Math.PI * i) / n;
    dims.forEach((d, i) => {{
      const a = angle(i), x2 = cx + R * Math.cos(a), y2 = cy + R * Math.sin(a);
      parts.push(`<line x1="${{cx}}" y1="${{cy}}" x2="${{x2.toFixed(1)}}" y2="${{y2.toFixed(1)}}" stroke="var(--rule)" stroke-width="1"/>`);
      const lx = cx + (R + 22) * Math.cos(a), ly = cy + (R + 18) * Math.sin(a);
      const anchor = Math.cos(a) > 0.25 ? "start" : Math.cos(a) < -0.25 ? "end" : "middle";
      parts.push(`<text x="${{lx.toFixed(1)}}" y="${{(ly + 4).toFixed(1)}}" text-anchor="${{anchor}}" fill="var(--ink)" font-size="11.5" font-weight="600">${{esc(d.label)}}</text>`);
    }});
    parts.push(...ringLabels);
    visible.forEach(m => {{
      const col = mColor(m.idx);
      const isHov = state.hover === m.model;
      const opacity = state.hover && !isHov ? 0.18 : 1;
      const pts = dims.map((d, i) => {{
        const r = rOf(Math.max(4, d.norm(m))), a = angle(i);
        return [cx + r * Math.cos(a), cy + r * Math.sin(a), d];
      }});
      const polyPts = pts.map(p => `${{p[0].toFixed(1)}},${{p[1].toFixed(1)}}`).join(" ");
      parts.push(`<polygon class="radar-poly" data-m="${{esc(m.model)}}" points="${{polyPts}}" fill="${{col}}" fill-opacity="${{isHov ? 0.34 : 0.20}}" stroke="${{col}}" stroke-width="${{isHov ? 2.8 : 2}}" stroke-linejoin="round" opacity="${{opacity}}" data-tip-t="${{esc(m.model)}}" data-tip-b="${{esc('gScore: ' + (m.gscore ?? '–') + (m.quant ? ' · quant: ' + m.quant : ''))}}"/>`);
      pts.forEach(([px, py, d]) => {{
        parts.push(`<circle class="radar-dot" data-m="${{esc(m.model)}}" cx="${{px.toFixed(1)}}" cy="${{py.toFixed(1)}}" r="${{isHov ? 5 : 3.8}}" fill="${{col}}" stroke="var(--card)" stroke-width="1.6" opacity="${{opacity}}" data-tip-t="${{esc(m.model + ' · ' + d.key)}}" data-tip-b="${{esc(d.fmt(m))}}"/>`);
      }});
    }});
  }}
  parts.push(`</svg>`);
  wrap.innerHTML = parts.join("");
}}

function renderBars() {{
  const wrap = $("#barWrap");
  const visible = visibleModels();
  if (!visible.length) {{
    wrap.innerHTML = `<div class="empty">Enable at least one model below to inspect bar charts.</div>`;
    return;
  }}
  const W = 540;
  const H = state.barMode === "perf" && DATA.perf.length
    ? Math.max(340, 34 + DATA.perf.length * Math.max(76, 26 + visible.length * 22))
    : 340;
  const parts = [`<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="Benchmark bar chart">`];

  if (state.barMode === "quant") {{
    // Grouped by base model, bars ordered from lowest -> highest quantization precision
    const fams = [...new Set(visible.map(m => m.base_model))];
    const padL = 40, padR = 16, padT = 22, padB = 48;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    for (const pct of [0, 25, 50, 75, 100]) {{
      const gy = padT + plotH - (plotH * pct) / 100;
      parts.push(`<line x1="${{padL}}" y1="${{gy}}" x2="${{W - padR}}" y2="${{gy}}" stroke="var(--rule)" stroke-dasharray="${{pct ? '3 3' : ''}}"/>`);
      parts.push(`<text x="${{padL - 7}}" y="${{gy + 4}}" text-anchor="end" fill="var(--muted)" font-size="10.5">${{pct}}</text>`);
    }}
    const grpW = plotW / Math.max(1, fams.length);
    fams.forEach((fam, fi) => {{
      const members = visible
        .filter(m => m.base_model === fam)
        .sort((a, b) => ((a.quant_bits ?? 999) - (b.quant_bits ?? 999)) || (a.quant || "").localeCompare(b.quant || ""));
      const ref = members[members.length - 1];
      const innerW = Math.min(grpW * 0.78, members.length * 48);
      const barW = Math.max(12, innerW / Math.max(1, members.length) - 6);
      const gx = padL + fi * grpW + (grpW - innerW) / 2;
      members.forEach((m, mi) => {{
        const g = m.gscore ?? 0, bh = (plotH * g) / 100;
        const bx = gx + mi * (barW + 6), by = padT + plotH - bh;
        const col = qColor(m.quant_bits, mi);
        const op = state.hover && state.hover !== m.model ? 0.25 : 1;
        const delta = (ref && ref.gscore != null && m.gscore != null && m !== ref)
          ? ` (Δ ${{(m.gscore - ref.gscore) >= 0 ? '+' : ''}}${{(m.gscore - ref.gscore).toFixed(1)}} vs ${{ref.quant || 'ref'}})`
          : "";
        parts.push(`<rect class="bar-rect" data-m="${{esc(m.model)}}" x="${{bx.toFixed(1)}}" y="${{by.toFixed(1)}}" width="${{barW.toFixed(1)}}" height="${{Math.max(2, bh).toFixed(1)}}" rx="5" fill="${{col}}" opacity="${{op}}" data-tip-t="${{esc(m.model)}}" data-tip-b="${{esc('gScore: ' + (m.gscore != null ? g.toFixed(1) : '–') + delta)}}"/>`);
        parts.push(`<text x="${{(bx + barW / 2).toFixed(1)}}" y="${{(by - 5).toFixed(1)}}" text-anchor="middle" fill="var(--ink)" font-size="10.5" font-weight="700" opacity="${{op}}">${{m.gscore != null ? g.toFixed(1) : '–'}}</text>`);
        if (barW >= 22) {{
          parts.push(`<text x="${{(bx + barW / 2).toFixed(1)}}" y="${{(padT + plotH + 14).toFixed(1)}}" text-anchor="middle" fill="${{col}}" font-size="9.5" font-weight="700" opacity="${{op}}">${{esc((m.quant || '—').slice(0, 8))}}</text>`);
        }}
      }});
      parts.push(`<text x="${{(padL + (fi + 0.5) * grpW).toFixed(1)}}" y="${{H - 8}}" text-anchor="middle" fill="var(--ink)" font-size="12" font-weight="700">${{esc(shortLbl(fam))}}</text>`);
    }});
  }} else if (state.barMode === "gscore") {{
    const padL = 166, padR = 54, padT = 18, padB = 26;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const rowH = Math.min(44, plotH / Math.max(1, visible.length));
    for (const pct of [0, 25, 50, 75, 100]) {{
      const gx = padL + (plotW * pct) / 100;
      parts.push(`<line x1="${{gx}}" y1="${{padT}}" x2="${{gx}}" y2="${{padT + rowH * visible.length}}" stroke="var(--rule)" stroke-dasharray="${{pct ? '3 3' : ''}}"/>`);
      parts.push(`<text x="${{gx}}" y="${{padT + rowH * visible.length + 16}}" text-anchor="middle" fill="var(--muted)" font-size="10.5">${{pct}}%</text>`);
    }}
    visible.forEach((m, i) => {{
      const g = m.gscore ?? 0, y = padT + i * rowH + 6, bh = Math.max(14, rowH - 12);
      const bw = Math.max(2, (plotW * g) / 100), col = mColor(m.idx);
      const op = state.hover && state.hover !== m.model ? 0.3 : 1;
      const lbl = m.quant ? `${{m.base_model.slice(0, 13)}} · ${{m.quant}}` : m.model.slice(0, 22);
      parts.push(`<text x="${{padL - 10}}" y="${{(y + bh / 2 + 4).toFixed(1)}}" text-anchor="end" fill="var(--ink)" font-size="11.5" font-weight="600">${{esc(lbl)}}</text>`);
      parts.push(`<rect x="${{padL}}" y="${{y}}" width="${{plotW}}" height="${{bh}}" rx="6" fill="var(--card2)"/>`);
      parts.push(`<rect class="bar-rect" data-m="${{esc(m.model)}}" x="${{padL}}" y="${{y}}" width="${{bw.toFixed(1)}}" height="${{bh}}" rx="6" fill="${{col}}" opacity="${{op}}" data-tip-t="${{esc(m.model)}}" data-tip-b="${{esc('gScore: ' + g.toFixed(1) + ' (' + m.n_bench + ' benchmarks)')}}"/>`);
      parts.push(`<text x="${{(padL + bw + 8).toFixed(1)}}" y="${{(y + bh / 2 + 4).toFixed(1)}}" fill="var(--ink)" font-size="12" font-weight="700">${{m.gscore != null ? g.toFixed(1) : '–'}}</text>`);
    }});
  }} else if (state.barMode === "perf" && DATA.perf.length) {{
    const padL = 166, padR = 72, padT = 16, padB = 18;
    const plotW = W - padL - padR;
    const perMetric = Math.max(76, (H - padT - padB) / DATA.perf.length);
    DATA.perf.forEach((p, mi) => {{
      const y0 = padT + mi * perMetric;
      const vals = visible.map(m => m.scores[p]?.value || 0);
      const mx = Math.max(1e-6, ...vals);
      parts.push(`<text x="${{padL}}" y="${{y0 + 12}}" fill="var(--a1)" font-size="11.5" font-weight="700">${{esc(p)}}</text>`);
      const rh = Math.min(24, (perMetric - 22) / Math.max(1, visible.length));
      visible.forEach((m, i) => {{
        const s = m.scores[p], v = s ? s.value : 0, y = y0 + 18 + i * rh, bh = Math.max(10, rh - 4);
        const bw = Math.max(s ? 4 : 0, (plotW * v) / mx), col = mColor(m.idx);
        const op = state.hover && state.hover !== m.model ? 0.3 : 1;
        const lbl = m.quant ? `${{m.base_model.slice(0, 12)}} · ${{m.quant}}` : m.model.slice(0, 20);
        parts.push(`<text x="${{padL - 8}}" y="${{(y + bh / 2 + 4).toFixed(1)}}" text-anchor="end" fill="var(--muted)" font-size="11">${{esc(lbl)}}</text>`);
        parts.push(`<rect x="${{padL}}" y="${{y}}" width="${{plotW}}" height="${{bh}}" rx="4" fill="var(--card2)"/>`);
        if (s) {{
          parts.push(`<rect class="bar-rect" data-m="${{esc(m.model)}}" x="${{padL}}" y="${{y}}" width="${{bw.toFixed(1)}}" height="${{bh}}" rx="4" fill="${{col}}" opacity="${{op}}" data-tip-t="${{esc(m.model + ' · ' + p)}}" data-tip-b="${{esc(v.toFixed(2) + ' ' + s.unit)}}"/>`);
          parts.push(`<text x="${{(padL + bw + 6).toFixed(1)}}" y="${{(y + bh / 2 + 4).toFixed(1)}}" fill="var(--ink)" font-size="11" font-weight="600">${{v.toFixed(2)}} ${{esc(s.unit)}}</text>`);
        }}
      }});
    }});
  }} else {{
    const benches = DATA.benches;
    if (!benches.length) {{
      wrap.innerHTML = `<div class="empty">No % benchmarks yet — switch to Speed / TTFT.</div>`;
      return;
    }}
    const padL = 40, padR = 16, padT = 18, padB = 42;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    for (const pct of [0, 25, 50, 75, 100]) {{
      const gy = padT + plotH - (plotH * pct) / 100;
      parts.push(`<line x1="${{padL}}" y1="${{gy}}" x2="${{W - padR}}" y2="${{gy}}" stroke="var(--rule)" stroke-dasharray="${{pct ? '3 3' : ''}}"/>`);
      parts.push(`<text x="${{padL - 7}}" y="${{gy + 4}}" text-anchor="end" fill="var(--muted)" font-size="10.5">${{pct}}</text>`);
    }}
    const grpW = plotW / benches.length;
    const innerW = Math.min(grpW * 0.80, visible.length * 34);
    const barW = Math.max(7, innerW / visible.length - 4);
    benches.forEach((b, bi) => {{
      const gx = padL + bi * grpW + (grpW - innerW) / 2;
      visible.forEach((m, mi) => {{
        const s = m.scores[b], v = s ? Math.max(0, Math.min(100, s.value)) : 0;
        const bh = (plotH * v) / 100, bx = gx + mi * (barW + 4), by = padT + plotH - bh;
        const col = mColor(m.idx), op = state.hover && state.hover !== m.model ? 0.25 : 1;
        if (s) {{
          parts.push(`<rect class="bar-rect" data-m="${{esc(m.model)}}" x="${{bx.toFixed(1)}}" y="${{by.toFixed(1)}}" width="${{barW.toFixed(1)}}" height="${{Math.max(2, bh).toFixed(1)}}" rx="4" fill="${{col}}" opacity="${{op}}" data-tip-t="${{esc(m.model + ' · ' + b)}}" data-tip-b="${{esc(v.toFixed(1) + '% (n=' + (s.n || '?') + ')')}}"/>`);
          if (barW >= 16) {{
            parts.push(`<text x="${{(bx + barW / 2).toFixed(1)}}" y="${{(by - 5).toFixed(1)}}" text-anchor="middle" fill="var(--ink)" font-size="10" font-weight="600" opacity="${{op}}">${{v.toFixed(0)}}</text>`);
          }}
        }}
      }});
      parts.push(`<text x="${{(padL + (bi + 0.5) * grpW).toFixed(1)}}" y="${{H - 14}}" text-anchor="middle" fill="var(--ink)" font-size="11.5" font-weight="600">${{esc(shortLbl(b))}}</text>`);
    }});
  }}
  parts.push(`</svg>`);
  wrap.innerHTML = parts.join("");
}}

function chipMarkup(m, legendKind, selModel) {{
  const off = state.hidden.has(m.model) ? " off" : "";
  const sel = legendKind === "area" && state.areaMode === "polar" && m.model === selModel ? " sel" : "";
  const qb = m.quant ? `<span class="qbadge ${{qTone(m.quant_bits)}}">${{esc(m.quant)}}</span>` : "";
  const sw = state.barMode === "quant" && legendKind === "bar" ? qColor(m.quant_bits, m.idx) : mColor(m.idx);
  return `<span class="chip${{off}}${{sel}}" data-legend="${{legendKind}}" data-m="${{esc(m.model)}}"><span class="sw" style="background:${{sw}}"></span>${{esc(m.base_model)}}${{qb}}</span>`;
}}

function renderLegends() {{
  const pool = filteredModels();
  const visible = pool.filter(m => !state.hidden.has(m.model));
  const polarFocus = (visible.find(m => m.model === (state.hover || state.focus)) || visible[0])?.model;
  $("#radarLegend").innerHTML = pool.map(m => chipMarkup(m, "area", polarFocus)).join("");
  $("#barLegend").innerHTML = pool.map(m => chipMarkup(m, "bar", polarFocus)).join("");
}}

function applyTableFilters() {{
  const q = state.search;
  document.querySelectorAll("#lbWrap table tbody tr").forEach(tr => {{
    const okFam = state.family === "all" || tr.dataset.base === state.family;
    const okQ = state.quant === "all" || tr.dataset.quant === state.quant;
    const okS = !q || tr.dataset.model.toLowerCase().includes(q) || tr.dataset.quant.toLowerCase().includes(q);
    tr.style.display = (okFam && okQ && okS) ? "" : "none";
    tr.classList.toggle("hl", tr.dataset.model === (state.hover || state.focus));
  }});
}}

function redraw() {{
  renderFilters();
  renderRadar();
  renderBars();
  renderLegends();
  applyTableFilters();
}}

// Delegated interactions: tooltips, hover highlight, legend toggle, row focus, dimension filters
document.addEventListener("mousemove", ev => {{
  const el = ev.target.closest("[data-tip-t]");
  if (el) showTip(ev, el.dataset.tipT, el.dataset.tipB);
  else hideTip();
}});
document.addEventListener("mouseover", ev => {{
  const el = ev.target.closest("[data-m],[data-model]");
  const m = el ? (el.dataset.m || el.dataset.model) : null;
  if (m && m !== state.hover) {{ state.hover = m; redraw(); }}
}});
document.addEventListener("mouseout", ev => {{
  if (state.hover && !ev.relatedTarget?.closest?.("[data-m],[data-model]")) {{ state.hover = null; redraw(); }}
}});
document.addEventListener("click", ev => {{
  const fp = ev.target.closest(".fpill");
  if (fp) {{
    if (fp.dataset.fam != null) state.family = fp.dataset.fam;
    if (fp.dataset.q != null) state.quant = fp.dataset.q;
    redraw();
    return;
  }}
  const chip = ev.target.closest(".chip");
  if (chip) {{
    const m = chip.dataset.m;
    if (state.areaMode === "polar" && chip.dataset.legend === "area") {{
      state.hidden.delete(m);
      state.focus = m;
    }} else {{
      if (state.hidden.has(m)) state.hidden.delete(m); else state.hidden.add(m);
    }}
    redraw();
    return;
  }}
  const tr = ev.target.closest("#lbWrap table tbody tr");
  if (tr) {{
    state.focus = state.focus === tr.dataset.model ? null : tr.dataset.model;
    redraw();
  }}
}});

$("#areaTabs").addEventListener("click", ev => {{
  const b = ev.target.closest(".tab");
  if (!b) return;
  state.areaMode = b.dataset.mode;
  $("#areaTabs").querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x === b));
  redraw();
}});
$("#barTabs").addEventListener("click", ev => {{
  const b = ev.target.closest(".tab");
  if (!b) return;
  state.barMode = b.dataset.mode;
  $("#barTabs").querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x === b));
  redraw();
}});
$("#incPerf").addEventListener("change", ev => {{ state.includePerf = ev.target.checked; renderRadar(); }});
$("#incQuant").addEventListener("change", ev => {{ state.includeQuant = ev.target.checked; renderRadar(); }});

// Live browser theme switcher
$("#themeBar").addEventListener("click", ev => {{
  const b = ev.target.closest(".tbtn");
  if (!b || !THEMES[b.dataset.theme]) return;
  const h = THEMES[b.dataset.theme], root = document.documentElement.style;
  for (const [k, v] of [["--bg",h.bg],["--card",h.card],["--card2",h.card2],["--ink",h.ink],["--muted",h.muted],
    ["--rule",h.rule],["--a1",h.accent1],["--a2",h.accent2],["--a3",h.accent3],["--good",h.good],["--mid",h.mid],["--warn",h.warn],["--bad",h.bad]])
    root.setProperty(k, v);
  $("#themeBar").querySelectorAll(".tbtn").forEach(x => x.classList.toggle("active", x === b));
}});

// Table search + column sorting
$("#searchBox").addEventListener("input", ev => {{
  state.search = ev.target.value.trim().toLowerCase();
  applyTableFilters();
}});
let sortState = {{ col: 0, asc: true }};
document.querySelectorAll("#lbWrap table thead th").forEach(th => {{
  th.addEventListener("click", () => {{
    const col = Number(th.dataset.col);
    sortState = {{ col, asc: sortState.col === col ? !sortState.asc : col <= 1 }};
    const tbody = $("#lbWrap table tbody");
    const trs = Array.from(tbody.querySelectorAll("tr"));
    trs.sort((a, b) => {{
      const av = a.children[col].dataset.v, bv = b.children[col].dataset.v;
      const na = parseFloat(av), nb = parseFloat(bv);
      const cmp = (!Number.isNaN(na) && !Number.isNaN(nb)) ? (na - nb) : av.localeCompare(bv);
      return sortState.asc ? cmp : -cmp;
    }});
    trs.forEach(tr => tbody.appendChild(tr));
  }});
}});

redraw();
</script>
</body>
</html>
"""


def write(settings: Settings, comparable_only: bool = False) -> dict:
    runs = R.list_runs(settings.results_path)
    lb = build(runs, comparable_only)
    base = settings.results_path.parent
    (base / "leaderboard.md").write_text(to_markdown(lb), encoding="utf-8")
    (base / "leaderboard.html").write_text(to_html(lb, getattr(settings, "theme", None)), encoding="utf-8")
    (base / "leaderboard.json").write_text(json.dumps(_payload(lb), indent=1), encoding="utf-8")
    return lb


def paths(settings: Settings) -> dict:
    base: Path = settings.results_path.parent
    return {k: base / f"leaderboard.{k}" for k in ("md", "html", "json")}
