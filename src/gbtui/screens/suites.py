"""Suites & Metrics screens: browse gbench suites by pillar, Golden Set tasks, and core/benchmark scoring formulas."""
from __future__ import annotations

import curses
import json
import math

from .. import catalog, metrics
from .. import theme as T
from ..ui import ESC, SIGINT, UI
from .common import ENTER, _clamp_top, _need_color, _quant_color, _skip_heading


class SuitesMixin:
    def screen_suites(self):
        cat = catalog.load()
        all_items = [(None, f"Golden Set · {len(cat['golden'])} deterministic checks (--golden-only)", "", {"pidx": -1})]
        all_items += [("golden:" + g["id"], "  " + g["id"], f"{g.get('category', '')} · {g.get('description', '')}",
                       {"kind": "golden", "data": g}) for g in cat["golden"]]
        for pidx, p in enumerate(cat["pillars"] + ["Other"]):
            group = [s for s in cat["suites"] if s["pillar"] == p]
            if not group:
                continue
            all_items.append((None, f"{p} · {len(group)} suites", "", {"pidx": pidx}))
            for s in sorted(group, key=lambda s: (not s["light"], s["name"])):
                all_items.append((s["name"], ("★ " if s["light"] else "  ") + s["name"],
                                  s["title"] + (" · " + s["description"] if s["description"] else ""),
                                  {"kind": "suite", "data": s, "pidx": pidx}))

        ui, sel, top = self.ui, 1, 0
        flt = ""
        light_only = False
        while True:
            vis = []
            for it in all_items:
                v, lab, note, meta = it
                if v is None:
                    vis.append(it)
                    continue
                if light_only:
                    if meta["kind"] == "golden" or not meta["data"].get("light"):
                        continue
                if flt:
                    hay = (lab + " " + note + " " + " ".join((meta["data"].get("needs") or []) if meta["kind"] == "suite" else [])).lower()
                    if flt not in hay:
                        continue
                vis.append(it)
            items = []
            for idx, it in enumerate(vis):
                if it[0] is None:
                    if idx + 1 < len(vis) and vis[idx + 1][0] is not None:
                        items.append(it)
                else:
                    items.append(it)

            h, w = ui.size()
            self.scr.erase()
            queued = set(self.s.run.evals)
            sub = f"from {cat['source']} · ★ laptop-friendly"
            if light_only:
                sub += " · FILTER: ★ ONLY"
            if flt:
                sub += f" · /{flt}"
            rbadge = (f"◉ {len(queued)} queued", T.GREEN) if queued else None
            ui.header("Suites", sub, right_badge=rbadge)

            preview_h = 4 if h >= 18 else 0
            view = max(1, h - 5 - preview_h)
            sel = _skip_heading(items, sel, 1) if items else 0
            top = _clamp_top(top, sel, view, len(items), items=items)

            if not items:
                ui.put(5, 4, "No suites match the current filter. Press / or l to clear.", ui.color(T.MUTED), maxw=max(0, w - 8))
            for i in range(view):
                j = top + i
                if j >= len(items):
                    break
                v, lab, note, meta = items[j]
                y = 3 + i
                if v is None:
                    pcol = T.YELLOW if meta["pidx"] < 0 else T.pillar_color(meta["pidx"])
                    ui.section(y, 2, lab, w - 5, pcol)
                    continue
                is_sel = j == sel
                if is_sel:
                    ui.sel_bar(y, 2, w - 5, T.YELLOW if meta["kind"] == "golden" else T.pillar_color(meta.get("pidx", 0)))
                bg = T.SEL if is_sel else None
                is_q = (meta["kind"] == "suite" and v in queued)
                ui.put(y, 4, "◉" if is_q else " ", ui.color(T.GREEN, bg, True))
                if lab.startswith("★ "):
                    ui.put(y, 6, "★ ", ui.color(T.YELLOW, bg, True))
                    ui.put(y, 8, lab[2:29], ui.color(T.LIGHT if (is_sel or is_q) else T.SOFT, bg, is_sel or is_q))
                else:
                    ui.put(y, 6, lab[:29], ui.color(T.LIGHT if (is_sel or is_q) else T.SOFT, bg, is_sel or is_q))
                nx = 36
                if meta["kind"] == "suite" and meta["data"].get("needs"):
                    for need in meta["data"]["needs"]:
                        chip = f"[{need}] "
                        if nx + len(chip) >= w - 4:
                            break
                        nx += ui.put(y, nx, chip, ui.color(_need_color(need), bg))
                ui.put(y, nx, note, ui.color(T.MUTED, bg), maxw=max(0, w - nx - 4))
            ui.scrollbar(3, w - 2, view, top, view, len(items))

            if preview_h and items and items[sel][0] is not None:
                v, _lab, note, meta = items[sel]
                py = 3 + view
                if meta["kind"] == "suite":
                    sd = meta["data"]
                    bm = metrics.benchmark_metric(sd["name"])
                    title = f"{sd['name']} · {sd.get('pillar', '')}"
                    badge = (f"metric: {bm.metric_tag}" if bm else ("★ laptop-friendly" if sd.get("light") else "endpoint only"))
                    bcol = T.CYAN if bm else (T.GREEN if sd.get("light") else T.YELLOW)
                    ui.card(py, 2, preview_h, w - 4, title=title, color=T.RULE,
                            title_color=T.pillar_color(meta.get("pidx", 0)), badge=badge, badge_color=bcol)
                    desc = sd.get("title", "") + (" — " + sd["description"] if sd.get("description") else "")
                else:
                    gd = meta["data"]
                    ui.card(py, 2, preview_h, w - 4, title=f"Golden · {gd['id']}", color=T.RULE, title_color=T.YELLOW,
                            badge=gd.get("category", ""), badge_color=T.PALE)
                    desc = gd.get("description", "")
                for li, ln in enumerate(UI.wrap(desc, max(12, w - 8))[: preview_h - 2]):
                    ui.put(py + 1 + li, 4, ln, ui.color(T.SOFT), maxw=max(0, w - 8))

            ui.footer("↑↓ move · Enter guide · m metric · Space queue · / filter · l ★ only · r run · q back")
            self.scr.refresh()
            k = ui.wait_key()
            if k is None or k == "RESIZE":
                continue
            if k in ("q", ESC, SIGINT):
                return
            if k == "/":
                f = ui.edit("Filter suites", flt, hint="matches name, description or needs; empty clears")
                if f is not None:
                    flt = f.lower().strip()
                    sel = top = 0
                continue
            if k == "l":
                light_only = not light_only
                sel = top = 0
                ui.notify("showing ★ laptop-friendly suites only" if light_only else "showing all suites", T.YELLOW, 1.8)
                continue
            if k == "r":
                self.screen_run()
                continue
            if k == "m":
                v = items[sel][0] if (items and items[sel][0] is not None) else None
                target = "golden_set" if (v and v.startswith("golden:")) else v
                self.screen_metrics(initial_tab="bench", focus_name=target)
                continue
            if not items:
                continue
            if k in (curses.KEY_UP, "k"):
                sel = _skip_heading(items, sel - 1, -1)
            elif k in (curses.KEY_DOWN, "j"):
                sel = _skip_heading(items, sel + 1, 1)
            elif k == curses.KEY_NPAGE:
                sel = _skip_heading(items, min(len(items) - 1, sel + view), 1)
            elif k == curses.KEY_PPAGE:
                sel = _skip_heading(items, max(0, sel - view), -1)
            elif k in ("g", curses.KEY_HOME):
                sel = _skip_heading(items, 0, 1)
            elif k in ("G", curses.KEY_END):
                sel = _skip_heading(items, len(items) - 1, -1)
            elif k in ENTER:
                v = items[sel][0]
                if v and v.startswith("golden:"):
                    self.show_golden(v.split(":", 1)[1])
                elif v:
                    self.show_doc(v)
            elif k == " ":
                v = items[sel][0]
                if v and not v.startswith("golden:"):
                    ev = self.s.run.evals
                    if v in ev:
                        ev.remove(v)
                        ui.notify(f"removed {v} from next run ({len(ev)} queued)", T.YELLOW, 2.0)
                    else:
                        ev.append(v)
                        ui.notify(f"◉ queued {v} for next run ({len(ev)} total)", T.GREEN, 2.0)
                    self.s.run.tier = "evals"
                    self.save()

    def show_doc(self, name):
        s = catalog.suite(name) or {}
        w = self.ui.size()[1]
        c = self.ui.color
        head = [
            [("needs    ", c(T.DIM)), (", ".join(s.get("needs") or ["nothing beyond the endpoint"]),
                                       c(T.YELLOW if s.get("needs") else T.GREEN, None, True))],
            [("pillar   ", c(T.DIM)), (s.get("pillar", "—"), c(T.PALE))],
        ]
        if s.get("options"):
            head.append([("options  ", c(T.DIM)), (s.get("options", ""), c(T.CYAN))])
        head.append([("", 0)])
        self.ui.pager(f"Suite · {name}", head + self.ui.markdown(catalog.doc(name), max(20, w - 6)), sub=s.get("title", ""))

    def show_golden(self, tid):
        d = catalog.golden_task(tid) or {}
        w = self.ui.size()[1]
        md = f"# {tid}\n\n**category** {d.get('category', '')}\n\n{d.get('description', '')}\n\n## Conversation\n\n"
        for m in (d.get("conversation") or {}).get("messages", []):
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(p.get("text", f"[{p.get('type')}]") for p in c if isinstance(p, dict))
            md += f"- **{m.get('role')}**: {c}\n"
        md += "\n## Golden truth\n\n```\n" + json.dumps(d.get("golden_truth", {}), indent=2)[:3000] + "\n```\n"
        self.ui.pager(f"Golden Set · {tid}", self.ui.markdown(md, max(20, w - 6)))

    @staticmethod
    def _metric_dir_badge(direction: str) -> tuple[str, str, int]:
        return {
            "higher": ("↑", "↑ higher is better", T.GREEN),
            "lower": ("↓", "↓ lower is better", T.CYAN),
            "dimension": ("◈", "◈ grouping dimension", T.VIOLET),
        }.get(direction, ("·", direction, T.PALE))

    def _live_metric_example(self, m: metrics.MetricDef) -> str:
        if m.id == "gscore" and self._lb_cache.get("models"):
            top = self._lb_cache["models"][0]
            pcts = [(b, s.value) for b, s in top["scores"].items() if s.unit == "%" and s.value is not None]
            if pcts and top.get("gscore") is not None:
                terms = " + ".join(f"{v:.1f}" for _, v in pcts[:4])
                if len(pcts) > 4:
                    terms += " + …"
                return f"Live (#1 {top['model']}): ({terms}) / {len(pcts)} = {top['gscore']:.1f}"
        return ""

    def _show_quant_bpw_table(self):
        c = self.ui.color
        lines = [
            [("◆ ", c(T.VIOLET, None, True)), ("Quantization Bit-Width (bpw) & Compactness Scale", c(T.LIGHT, None, True))],
            [("  compact = clamp(10, 100, 100 − ((log₂(bpw) − 1) / 4) × 90)", c(T.CYAN, None, True))],
            [("", 0)],
            [(f"  {'Quant':<10}{'bpw':>6}   {'compact':>8}   bar", c(T.DIM))],
            [("  " + "─" * 54, c(T.RULE_DIM))],
        ]
        bw = 24
        for q, bpw in metrics.quant_reference_rows():
            comp = max(10.0, min(100.0, 100.0 - ((math.log2(bpw) - 1.0) / 4.0) * 90.0)) if bpw > 0 else 0.0
            filled = int(round(comp * bw / 100.0))
            col = _quant_color(q)
            segs = [(f"  {q:<10}", c(col, None, True)), (f"{bpw:6.2f}   ", c(T.SOFT)),
                    (f"{comp:6.1f}%   ", c(T.score_color(comp), None, True))]
            for j in range(bw):
                segs.append(("█" if j < filled else "·", c(col if j < filled else T.RULE_DIM, None, False, auto_sel=False)))
            lines.append(segs)
        lines += [
            [("", 0)],
            [("  Higher compactness = smaller memory footprint & faster memory-bound decode.", c(T.MUTED))],
            [("  Used when the 'quant' dimension is toggled on in the web Radar / Polar Area chart.", c(T.DIM))],
        ]
        self.ui.pager("Metrics · Quantization bpw Reference", lines, sub="bits-per-weight lookup & compactness scale")

    def _show_metric_pager(self, m: metrics.MetricDef):
        c = self.ui.color
        w = self.ui.size()[1]
        wrap_w = max(24, w - 10)
        _sym, dir_lbl, dir_col = self._metric_dir_badge(m.direction)
        lines = [
            [("category  ", c(T.DIM)), (m.category, c(T.VIOLET, None, True))],
            [("unit      ", c(T.DIM)), (m.unit, c(T.CYAN, None, True)), ("    ", 0), (dir_lbl, c(dir_col, None, True))],
            [("source    ", c(T.DIM)), (m.source, c(T.PALE))],
            [("─" * max(10, w - 6), c(T.RULE_DIM))],
            [("", 0)],
            [("╭─ ", c(T.PINK)), ("Formula", c(T.PINK, None, True))],
        ]
        for fl in m.formula.splitlines():
            for part in UI.wrap(fl, wrap_w):
                lines.append([("│ ", c(T.PINK)), (part, c(T.LIGHT, None, True))])
        lines += [[("╰─", c(T.PINK))], [("", 0)], [("◆ ", c(T.CYAN, None, True)), ("How it is calculated", c(T.LIGHT, None, True))]]
        for part in UI.wrap(m.details, wrap_w):
            lines.append([("  " + part, c(T.SOFT))])
        lines += [[("", 0)], [("◆ ", c(T.GREEN, None, True)), ("Worked example", c(T.LIGHT, None, True))]]
        for part in UI.wrap(m.example, wrap_w):
            lines.append([("  " + part, c(T.GREEN))])
        live_ex = self._live_metric_example(m)
        if live_ex:
            for part in UI.wrap(live_ex, wrap_w):
                lines.append([("  " + part, c(T.YELLOW, None, True))])
        if m.caveats:
            lines += [[("", 0)], [("⚠ ", c(T.YELLOW, None, True)), ("Notes & comparability", c(T.YELLOW, None, True))]]
            for part in UI.wrap(m.caveats, wrap_w):
                lines.append([("  " + part, c(T.MUTED))])
        self.ui.pager(f"Metric · {m.name}", lines, sub=m.summary)

    def _show_benchmark_metric_pager(self, bm: metrics.BenchmarkMetric):
        c = self.ui.color
        w = self.ui.size()[1]
        wrap_w = max(24, w - 10)
        lines = [
            [("benchmark ", c(T.DIM)), (bm.name, c(T.LIGHT, None, True)), (f"  ({bm.title})", c(T.PALE))],
            [("pillar    ", c(T.DIM)), (bm.pillar, c(T.VIOLET, None, True)),
             ("   ★ laptop-friendly" if bm.light else "", c(T.GREEN, None, True))],
            [("metric    ", c(T.DIM)), (bm.metric_tag, c(T.CYAN, None, True))],
        ]
        if bm.needs:
            lines.append([("needs     ", c(T.DIM)), (", ".join(bm.needs), c(T.YELLOW, None, True))])
        lines += [
            [("─" * max(10, w - 6), c(T.RULE_DIM))],
            [("", 0)],
            [("╭─ ", c(T.PINK)), ("How This Benchmark Is Scored", c(T.PINK, None, True))],
        ]
        for part in UI.wrap(bm.headline, wrap_w):
            lines.append([("│ ", c(T.PINK)), (part, c(T.LIGHT))])
        lines += [[("╰─", c(T.PINK))], [("", 0)]]
        if bm.sub_metrics:
            lines.append([("◆ ", c(T.GREEN, None, True)), ("Sub-metrics Reported", c(T.LIGHT, None, True))])
            lines.append([("  " + " · ".join(bm.sub_metrics), c(T.GREEN))])
            lines.append([("", 0)])
        lines.append([("◆ ", c(T.CYAN, None, True)), ("Evaluation Protocol", c(T.LIGHT, None, True))])
        for part in UI.wrap(bm.protocol, wrap_w):
            lines.append([("  " + part, c(T.SOFT))])
        if bm.has_doc:
            lines += [[("", 0)], [("─" * max(10, w - 6), c(T.RULE_DIM))],
                      [("◆ ", c(T.VIOLET, None, True)), (f"Full gbench guide (docs/evals/{bm.name}.md)", c(T.PALE, None, True))],
                      [("", 0)]]
            lines += self.ui.markdown(catalog.doc(bm.name), max(20, w - 6))
        self.ui.pager(f"Benchmark Metric · {bm.name}", lines, sub=f"{bm.metric_tag} · {bm.pillar}")

    def screen_metrics(self, initial_tab: str = "core", focus_name: str | None = None):
        ui = self.ui
        self._refresh_summary()
        tab = "bench" if initial_tab == "bench" else "core"
        core_cats = ["All", *metrics.CATEGORIES]
        all_bms = metrics.benchmark_metrics()
        bench_pillars = ["All", *list(dict.fromkeys(b.pillar for b in all_bms))]
        cat_idx = 0
        pillar_idx = 0
        light_only = False
        flt = ""
        sel, top = 0, 0
        pending_focus = focus_name

        while True:
            items: list[tuple] = []
            if tab == "core":
                cat_filter = core_cats[cat_idx]
                for c_i, (cname, mlist) in enumerate(metrics.by_category()):
                    if cat_filter != "All" and cname != cat_filter:
                        continue
                    matched = []
                    for m in mlist:
                        if flt:
                            hay = f"{m.name} {m.category} {m.unit} {m.formula} {m.summary} {m.details} {m.source} {m.caveats}".lower()
                            if flt not in hay:
                                continue
                        matched.append(m)
                    if matched:
                        items.append((None, cname, c_i))
                        for m in matched:
                            items.append((m, m.name, c_i))
            else:
                p_filter = bench_pillars[pillar_idx]
                by_p: dict[str, list[metrics.BenchmarkMetric]] = {}
                for b in all_bms:
                    if p_filter != "All" and b.pillar != p_filter:
                        continue
                    if light_only and not b.light:
                        continue
                    if flt:
                        hay = f"{b.name} {b.title} {b.pillar} {b.metric_tag} {b.headline} {b.protocol} {' '.join(b.sub_metrics)}".lower()
                        if flt not in hay:
                            continue
                    by_p.setdefault(b.pillar, []).append(b)
                for p_i, (pname, blist) in enumerate(by_p.items()):
                    items.append((None, f"{pname} · {len(blist)}", p_i))
                    for b in blist:
                        items.append((b, b.name, p_i))

            if pending_focus and items:
                target_bm = metrics.benchmark_metric(pending_focus)
                target_key = (target_bm.name if target_bm else pending_focus).lower()
                for idx, it in enumerate(items):
                    if it[0] is not None and getattr(it[0], "name", "").lower() == target_key:
                        sel = idx
                        break
                pending_focus = None

            h, w = ui.size()
            self.scr.erase()
            shown_n = sum(1 for it in items if it[0] is not None)
            if tab == "core":
                sub = "exact formulas, sources & worked examples used by gbench-tui"
                if core_cats[cat_idx] != "All":
                    sub = f"category: {core_cats[cat_idx]}"
                rbadge = (f"∑ {shown_n}/{len(metrics.METRICS)} formulas", T.CORAL)
            else:
                sub = "scoring metric & protocol for every gbench benchmark"
                if bench_pillars[pillar_idx] != "All":
                    sub = f"pillar: {bench_pillars[pillar_idx]}"
                if light_only:
                    sub += " · ★ ONLY"
                rbadge = (f"◈ {shown_n}/{len(all_bms)} benchmarks", T.CYAN)
            if flt:
                sub += f" · /{flt}"
            ui.header("Metrics & Formulas", sub, right_badge=rbadge)

            tx = 3
            if tab == "core":
                tx += ui.pill(3, tx, f"1 Core Formulas ({len(metrics.METRICS)})", fg=T.INK, bg=T.PINK, bold=True) + 2
                tx += ui.soft_pill(3, tx, f"2 By Benchmark ({len(all_bms)})", fg=T.MUTED, bg=T.BAR_ALT) + 2
            else:
                tx += ui.soft_pill(3, tx, f"1 Core Formulas ({len(metrics.METRICS)})", fg=T.MUTED, bg=T.BAR_ALT) + 2
                tx += ui.pill(3, tx, f"2 By Benchmark ({len(all_bms)})", fg=T.INK, bg=T.CYAN, bold=True) + 2
            if tx + 16 < w:
                ui.put(3, tx, "Tab / ←→ switch view", ui.color(T.DIM), maxw=max(0, w - tx - 2))

            sel = _skip_heading(items, sel, 1) if items else 0
            cur_obj = items[sel][0] if (items and items[sel][0] is not None) else None
            cur_ci = items[sel][2] if items else 0
            split = w >= 88
            body_y = 5
            body_h = max(6, h - body_y - 2)

            if split:
                left_w = min(44, max(32, int(w * 0.39)))
                list_h = body_h
            else:
                left_w = max(12, w - 4)
                detail_h = min(10, max(5, body_h // 2)) if cur_obj else 0
                list_h = max(4, body_h - detail_h)

            view = max(1, list_h - 2)
            top = _clamp_top(top, sel, view, len(items), items=items)
            l_title = "Core Metrics" if tab == "core" else "Benchmarks"
            l_badge = (core_cats[cat_idx][:14] if tab == "core" else bench_pillars[pillar_idx][:16])
            ui.card(body_y, 2, list_h, left_w, title=l_title, color=T.RULE,
                    title_color=T.CORAL if tab == "core" else T.CYAN,
                    badge=l_badge if l_badge != "All" else "all", badge_color=T.VIOLET,
                    icon="∑" if tab == "core" else "◈")
            if not items:
                ui.put(body_y + 2, 4, "No items match filter (/ or c to reset).", ui.color(T.MUTED), maxw=max(0, left_w - 6))
            for i in range(view):
                j = top + i
                if j >= len(items):
                    break
                obj, lab, ci = items[j]
                ry = body_y + 1 + i
                pcol = T.pillar_color(ci)
                if obj is None:
                    ui.section(ry, 3, lab, left_w - 6, pcol)
                    continue
                is_sel = j == sel
                if is_sel:
                    ui.sel_bar(ry, 3, left_w - 2, pcol)
                bg = T.SEL if is_sel else None
                if tab == "core":
                    sym, _dlbl, dcol = self._metric_dir_badge(obj.direction)
                    ui.put(ry, 4, sym, ui.color(dcol, bg, True))
                    u_short = {"seconds": "s", "tag": "bpw", "bytes": "RAM"}.get(obj.unit.split(" ")[0], obj.unit.split(" ")[0])
                    u_room = len(u_short) + 2 if left_w >= 32 else 0
                    name_max = max(4, left_w - 8 - u_room)
                    disp_lab = (lab[: name_max - 1] + "…") if len(lab) > name_max and name_max > 2 else lab
                    ui.put(ry, 6, disp_lab, ui.color(T.LIGHT if is_sel else T.SOFT, bg, is_sel), maxw=name_max)
                    if u_room:
                        ux = 2 + left_w - len(u_short) - 2
                        if ux > 10:
                            ui.put(ry, ux, u_short, ui.color(T.DIM, bg))
                else:
                    ui.put(ry, 4, "★" if obj.light else "·", ui.color(T.YELLOW if obj.light else T.DIM, bg, obj.light))
                    mtag_raw = obj.metric_tag
                    mtag = ((mtag_raw[:17] + "…") if len(mtag_raw) > 18 else mtag_raw) if left_w >= 38 else ""
                    t_room = (len(mtag) + 2) if mtag else 0
                    name_max = max(6, left_w - 8 - t_room)
                    disp_lab = (lab[: name_max - 1] + "…") if len(lab) > name_max and name_max > 2 else lab
                    ui.put(ry, 6, disp_lab, ui.color(T.LIGHT if is_sel else T.SOFT, bg, is_sel), maxw=name_max)
                    if mtag:
                        tx_m = 2 + left_w - len(mtag) - 2
                        if tx_m > 12:
                            ui.put(ry, tx_m, mtag, ui.color(T.CYAN if is_sel else T.MUTED, bg, is_sel))
            ui.scrollbar(body_y + 1, 2 + left_w - 1, view, top, view, len(items))

            if cur_obj:
                if split:
                    dx, dy, dw, dh = 2 + left_w + 1, body_y, max(20, w - left_w - 5), body_h
                else:
                    dx, dy, dw, dh = 2, body_y + list_h, max(12, w - 4), max(4, h - (body_y + list_h) - 2)
                pcol = T.pillar_color(cur_ci)
                cy = dy + 1
                max_cy = dy + dh - 1
                inner_w = max(10, dw - 4)

                def put_line(text, col, bold=False, indent=0):
                    nonlocal cy
                    if cy >= max_cy:
                        return
                    ui.put(cy, dx + 2 + indent, text, ui.color(col, None, bold), maxw=max(0, inner_w - indent))
                    cy += 1

                if tab == "core":
                    _sym, dlbl, dcol = self._metric_dir_badge(cur_obj.direction)
                    ui.card(dy, dx, dh, dw, title=cur_obj.name, color=pcol, title_color=pcol, badge=dlbl, badge_color=dcol)
                    if cy < max_cy:
                        ui.put(cy, dx + 2, "unit    ", ui.color(T.DIM))
                        ux = dx + 10
                        ux += ui.soft_pill(cy, ux, cur_obj.unit, fg=T.CYAN, bg=T.BAR_ALT, bold=True) + 2
                        if ux + 10 < dx + dw - 2:
                            ui.put(cy, ux, "src: " + cur_obj.source, ui.color(T.MUTED), maxw=max(0, dx + dw - 2 - ux))
                        cy += 1
                    if cy < max_cy:
                        cy += 1
                    if cy < max_cy:
                        put_line("FORMULA", T.PINK, bold=True)
                    for fl in cur_obj.formula.splitlines():
                        for part in UI.wrap(fl, max(10, inner_w - 3)):
                            if cy >= max_cy:
                                break
                            ui.put(cy, dx + 2, "▎", ui.color(T.PINK, None, True))
                            ui.put(cy, dx + 4, part, ui.color(T.LIGHT, None, True), maxw=max(0, inner_w - 2))
                            cy += 1
                    if cy < max_cy:
                        cy += 1
                    if cy < max_cy:
                        put_line("HOW IT WORKS", T.CYAN, bold=True)
                    for part in UI.wrap(cur_obj.details, inner_w):
                        if cy >= max_cy - (3 if split else 1):
                            break
                        put_line(part, T.SOFT)
                    if cy < max_cy:
                        cy += 1
                    if cy < max_cy:
                        put_line("EXAMPLE", T.GREEN, bold=True)
                    for ei, part in enumerate(UI.wrap(cur_obj.example, max(10, inner_w - 2))):
                        if cy >= max_cy:
                            break
                        ui.put(cy, dx + 2, "▸" if ei == 0 else " ", ui.color(T.GREEN, None, True))
                        ui.put(cy, dx + 4, part, ui.color(T.GREEN), maxw=max(0, inner_w - 2))
                        cy += 1
                    live_ex = self._live_metric_example(cur_obj)
                    if live_ex and cy < max_cy:
                        for li, part in enumerate(UI.wrap(live_ex, max(10, inner_w - 2))):
                            if cy >= max_cy:
                                break
                            ui.put(cy, dx + 2, "★" if li == 0 else " ", ui.color(T.YELLOW, None, True))
                            ui.put(cy, dx + 4, part, ui.color(T.YELLOW, None, True), maxw=max(0, inner_w - 2))
                            cy += 1
                    if cur_obj.caveats and cy + 1 < max_cy:
                        cy += 1
                        put_line("NOTE", T.YELLOW, bold=True)
                        for part in UI.wrap(cur_obj.caveats, inner_w):
                            if cy >= max_cy:
                                break
                            put_line(part, T.MUTED)
                else:
                    bm: metrics.BenchmarkMetric = cur_obj
                    b_badge = "★ laptop-friendly" if bm.light else (", ".join(bm.needs[:2]) or "endpoint only")
                    b_bcol = T.GREEN if bm.light else T.YELLOW
                    ui.card(dy, dx, dh, dw, title=f"{bm.name} · {bm.title}", color=pcol, title_color=pcol,
                            badge=b_badge, badge_color=b_bcol)
                    if cy < max_cy:
                        ui.put(cy, dx + 2, "pillar  ", ui.color(T.DIM))
                        ux = dx + 10
                        p_lbl = (bm.pillar[:27] + "…") if len(bm.pillar) > 28 else bm.pillar
                        ux += ui.soft_pill(cy, ux, p_lbl, fg=pcol, bg=T.BAR_ALT, bold=True) + 2
                        src_lbl = f"docs/evals/{bm.name}.md" if bm.has_doc else "built-in catalog"
                        if ux + 10 < dx + dw - 2:
                            ui.put(cy, ux, src_lbl, ui.color(T.DIM), maxw=max(0, dx + dw - 2 - ux))
                        cy += 1
                    if cy < max_cy:
                        cy += 1
                    if cy < max_cy:
                        put_line("PRIMARY METRIC", T.PINK, bold=True)
                    if cy < max_cy:
                        ui.put(cy, dx + 2, "▎", ui.color(T.PINK, None, True))
                        ui.soft_pill(cy, dx + 4, bm.metric_tag, fg=T.CYAN, bg=T.BAR_ALT, bold=True, maxw=max(8, inner_w - 4))
                        cy += 1
                    if cy < max_cy:
                        cy += 1
                    if cy < max_cy:
                        put_line("HOW THIS BENCHMARK IS SCORED", T.CYAN, bold=True)
                    for part in UI.wrap(bm.headline, inner_w):
                        if cy >= max_cy - (5 if split else 1):
                            break
                        put_line(part, T.SOFT)
                    if bm.sub_metrics and cy + 1 < max_cy:
                        cy += 1
                        put_line("SUB-METRICS REPORTED", T.GREEN, bold=True)
                        if cy < max_cy:
                            sx = dx + 2
                            for sm in bm.sub_metrics:
                                if sx + len(sm) + 3 >= dx + dw - 2:
                                    cy += 1
                                    sx = dx + 2
                                    if cy >= max_cy - (2 if split else 0):
                                        break
                                sx += ui.soft_pill(cy, sx, sm, fg=T.GREEN, bg=T.BAR_ALT) + 1
                            cy += 1
                    if cy + 1 < max_cy:
                        cy += 1
                        put_line("EVALUATION PROTOCOL", T.VIOLET, bold=True)
                        for part in UI.wrap(bm.protocol, inner_w):
                            if cy >= max_cy - (1 if bm.needs and split else 0):
                                break
                            put_line(part, T.MUTED)
                    if bm.needs and cy < max_cy:
                        nx = dx + 2
                        nx += ui.put(cy, nx, "needs: ", ui.color(T.DIM))
                        for nd in bm.needs:
                            if nx + len(nd) + 3 >= dx + dw - 2:
                                break
                            nx += ui.soft_pill(cy, nx, nd, fg=_need_color(nd), bg=T.BAR_ALT) + 1

            hint = ("↑↓ select · Tab/←→ By Benchmark · Enter full card · c category · / search · b bpw · q back"
                    if tab == "core" else
                    "↑↓ select · Tab/←→ Core Formulas · Enter full guide · c pillar · l ★ only · / search · q back")
            ui.footer(hint)
            self.scr.refresh()
            k = ui.wait_key()
            if k is None or k == "RESIZE":
                continue
            if k in ("q", ESC, SIGINT):
                return
            if k in ("\t", curses.KEY_LEFT, curses.KEY_RIGHT):
                tab = "bench" if tab == "core" else "core"
                sel = top = 0
                continue
            if k == "1":
                tab, sel, top = "core", 0, 0
                continue
            if k == "2":
                tab, sel, top = "bench", 0, 0
                continue
            if k == "c":
                if tab == "core":
                    cat_idx = (cat_idx + 1) % len(core_cats)
                else:
                    pillar_idx = (pillar_idx + 1) % len(bench_pillars)
                sel = top = 0
                continue
            if k == "l" and tab == "bench":
                light_only = not light_only
                sel = top = 0
                ui.notify("showing ★ laptop-friendly benchmarks only" if light_only else "showing all benchmarks", T.YELLOW, 1.8)
                continue
            if k == "/":
                q_in = ui.edit("Search metrics", flt, hint="matches benchmark, metric (e.g. F1, pass@1), or formula; empty clears")
                if q_in is not None:
                    flt = q_in.lower().strip()
                    sel = top = 0
                continue
            if k == "b":
                self._show_quant_bpw_table()
                continue
            if not items:
                continue
            if k in (curses.KEY_UP, "k"):
                sel = _skip_heading(items, sel - 1, -1)
            elif k in (curses.KEY_DOWN, "j"):
                sel = _skip_heading(items, sel + 1, 1)
            elif k == curses.KEY_NPAGE:
                sel = _skip_heading(items, min(len(items) - 1, sel + view), 1)
            elif k == curses.KEY_PPAGE:
                sel = _skip_heading(items, max(0, sel - view), -1)
            elif k in ("g", curses.KEY_HOME):
                sel = _skip_heading(items, 0, 1)
            elif k in ("G", curses.KEY_END):
                sel = _skip_heading(items, len(items) - 1, -1)
            elif k in ENTER and cur_obj:
                if tab == "core":
                    self._show_metric_pager(cur_obj)
                else:
                    self._show_benchmark_metric_pager(cur_obj)
