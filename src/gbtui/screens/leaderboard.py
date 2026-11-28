"""Leaderboard screen: rankings by gScore, Adapter × Harness heatmap (with Code Mode & KV cache), Pareto HTML & Report."""
from __future__ import annotations

import contextlib
import curses
import subprocess
import sys

from .. import leaderboard, report, results as R
from .. import theme as T
from ..ui import ESC, SIGINT
from .common import ENTER, _adapter_color, _clamp_top, _quant_color, _short


class LeaderboardMixin:
    def _model_breakdown(self, row: dict, benches: list, perf: list):
        c = self.ui.color
        g = row["gscore"]
        q = row.get("quant", "")
        qb = row.get("quant_bits")
        q_seg = [(f"  [{q}" + (f" · ~{qb:.1f} bpw" if qb else "") + "]", c(_quant_color(q), None, True))] if q else []
        lines = [
            [("model   ", c(T.DIM)), (row.get("base_model") or row["model"], c(T.LIGHT, None, True)), *q_seg],
            [("gScore  ", c(T.DIM)), (f"{g:.2f}" if g is not None else "—", c(T.score_color(g), None, True)),
             (f"  · {row['runs']} run(s) · {row['n_bench']} benchmarks · updated {row['when']}", c(T.MUTED))],
        ]
        eco = row.get("token_economy") or {}
        if eco.get("gscore_per_1k_tokens"):
            lines.append([("economy ", c(T.DIM)),
                          (f"{eco['gscore_per_1k_tokens']:.2f} gScore/1k tok", c(T.GREEN, None, True)),
                          (f"  · pass avg {eco.get('avg_tokens_pass') or 0:.0f}t · fail avg {eco.get('avg_tokens_fail') or 0:.0f}t", c(T.MUTED))])
        lines += [
            [("", 0)],
            [("◆ ", c(T.PINK, None, True)), ("Accuracy Benchmarks", c(T.LIGHT, None, True))],
        ]
        bw = 26
        for b in benches:
            s = row["scores"].get(b)
            if s is None:
                continue
            filled = int(round(max(0.0, min(100.0, s.value)) * bw / 100.0))
            segs = [(f"  {b:<26}", c(T.SOFT))]
            for j in range(bw):
                if j < filled:
                    col = T.grad(j, bw) if s.value >= 50 else T.score_color(s.value)
                    segs.append(("█", c(col, None, False, auto_sel=False)))
                else:
                    segs.append(("·", c(T.RULE_DIM)))
            segs.append((f"  {s.value:5.1f}%", c(T.score_color(s.value), None, True)))
            if s.n:
                segs.append((f"   n={s.n}", c(T.DIM)))
            lines.append(segs)
        if perf:
            lines.append([("", 0)])
            lines.append([("◆ ", c(T.GREEN, None, True)), ("Performance & Harness Metrics", c(T.LIGHT, None, True))])
            for b in perf:
                s = row["scores"].get(b)
                if s is None:
                    continue
                lines.append([(f"  {b:<26}", c(T.SOFT)), (f"{s.value:.2f} {s.unit}", c(T.CYAN, None, True)),
                              (f"   n={s.n}" if s.n else "", c(T.DIM))])
        self.ui.pager(f"Model · {row['model']}", lines, sub="all latest benchmark scores for this model")

    def screen_leaderboard(self):
        ui = self.ui
        comparable = False
        show_matrix = False
        flt = ""
        left = 0
        sel, top = 0, 0
        lb = ui.busy("building the leaderboard…", lambda: leaderboard.build(R.list_runs(self.s.results_path), comparable))
        while True:
            h, w = ui.size()
            self.scr.erase()
            all_models = lb["models"]
            models = [m for m in all_models if not flt or flt in f"{m['model']} {m.get('base_model', '')} {m.get('quant', '')} {m.get('adapter', '')} {m.get('harness', '')}".lower()]
            benches = lb["benches"] + lb["perf"]
            leader_g = models[0]["gscore"] if (models and models[0]["gscore"] is not None) else None
            q_cnt = len(lb.get("quants") or [])
            a_cnt = len(lb.get("adapters") or [])
            h_cnt = len(lb.get("harnesses") or [])
            sub = (f"{len(models)} variant(s)"
                   + (f" · {q_cnt} quant(s)" if q_cnt else "")
                   + (f" · {a_cnt} adapter(s)" if a_cnt else "")
                   + (f" · {h_cnt} harness(es)" if h_cnt else "")
                   + f" · updated {lb['updated']}")
            if flt:
                sub += f" · /{flt}"
            rbadge = ("◈ ADAPTER × HARNESS MATRIX", T.PINK) if show_matrix else (
                "★ COMPARABLE ONLY" if comparable else "ALL RUNS", T.YELLOW if comparable else T.PINK)
            ui.header("Leaderboard", sub, right_badge=rbadge)

            if show_matrix:
                self._draw_harness_matrix(lb, h, w)
                p = leaderboard.paths(self.s)
                ui.put(h - 3, 3, f"files: {p['md']} · .html · .json", ui.color(T.DIM), maxw=max(0, w - 6))
                ui.footer("H rankings · M run matrix · R report · o HTML (Pareto) · w write · q back")
                self.scr.refresh()
                k = ui.wait_key()
                if k is None or k == "RESIZE":
                    continue
                if k in ("q", ESC, SIGINT):
                    return
                if k in ("H", "h", "\t"):
                    show_matrix = False
                elif k == "M":
                    self.screen_matrix()
                    lb = leaderboard.build(R.list_runs(self.s.results_path), comparable)
                elif k == "R":
                    md_p, html_p = report.write_report(self.s)
                    ui.notify(f"✔ wrote {md_p.name} & {html_p.name}", T.GREEN, 3.0)
                elif k == "w":
                    leaderboard.write(self.s, comparable)
                    ui.notify("✔ wrote leaderboard.md / .html / .json", T.GREEN)
                elif k == "o":
                    leaderboard.write(self.s, comparable)
                    opener = "open" if sys.platform == "darwin" else "xdg-open"
                    with contextlib.suppress(OSError):
                        subprocess.Popen([opener, str(p["html"])], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    ui.notify("opened leaderboard.html in browser", T.CYAN)
                continue

            preview_h = 4 if (models and h >= 18) else 0
            table_y = 3
            table_h = max(6, h - table_y - 3 - preview_h)
            rows_vis = max(1, (table_h - 3) // 2)
            sel = max(0, min(sel, len(models) - 1)) if models else 0
            top = _clamp_top(top, sel, rows_vis, len(models))
            card_r = w - 3

            x0 = 4
            cw = 12
            has_q = bool(lb.get("quants")) and w >= 78
            m_w = 18 if has_q else 22
            q_w = 9 if has_q else 0
            q_x = x0 + 6 + m_w
            g_x = q_x + q_w
            cols_x = g_x + 18
            max_cols = max(0, (w - 4 - cols_x) // cw)
            left = max(0, min(left, max(0, len(benches) - max(1, max_cols))))
            vis = benches[left: left + max_cols] if max_cols > 0 else []
            nav_badge = ""
            if len(benches) > max_cols and max_cols > 0:
                nav_badge = f"{'◀ ' if left > 0 else ''}cols {left + 1}–{left + len(vis)}/{len(benches)}{' ▶' if left + len(vis) < len(benches) else ''}"
            ui.card(table_y, 2, table_h, w - 4, title="Rankings", color=T.RULE, title_color=T.YELLOW,
                    badge=nav_badge, badge_color=T.MUTED, icon="★")

            hy = table_y + 1
            hdr = f"{'#':<6}{'model':<{m_w}}" + (f"{'quant':<{q_w}}" if has_q else "") + f"{'gScore':<18}"
            ui.put(hy, x0, hdr, ui.color(T.DIM), maxw=max(0, card_r - x0 - 1))
            for j, b in enumerate(vis):
                ui.put(hy, cols_x + j * cw, _short(b)[: cw - 1],
                       ui.color(T.grad(j + left, max(2, len(benches))), None, True), maxw=max(0, card_r - (cols_x + j * cw)))

            for i in range(rows_vis):
                idx = top + i
                if idx >= len(models):
                    break
                r = models[idx]
                yy = table_y + 2 + i * 2
                if yy >= table_y + table_h - 1:
                    break
                is_sel = idx == sel
                if is_sel:
                    ui.sel_bar(yy, 3, w - 6, {0: T.YELLOW, 1: T.CYAN, 2: T.CORAL}.get(idx, T.VIOLET))
                bg = T.SEL if is_sel else None
                if idx == 0:
                    ui.put(yy, x0, " ★ 1 ", ui.color(T.contrast_fg(T.YELLOW), T.YELLOW, True, auto_sel=False))
                elif idx == 1:
                    ui.put(yy, x0, " ◈ 2 ", ui.color(T.contrast_fg(T.SOFT), T.SOFT, True, auto_sel=False))
                elif idx == 2:
                    ui.put(yy, x0, " ◇ 3 ", ui.color(T.contrast_fg(T.CORAL), T.CORAL, True, auto_sel=False))
                else:
                    ui.put(yy, x0, f"  {idx + 1:<4}", ui.color(T.MUTED, bg, True))
                m_disp = (r.get("base_model") or r["model"]) if has_q else r["model"]
                ui.put(yy, x0 + 6, f"{m_disp[:m_w - 1]:<{m_w}}", ui.color(T.LIGHT, bg, True), maxw=max(0, card_r - x0 - 7))
                if has_q:
                    rq = r.get("quant", "")
                    ui.put(yy, q_x, f"{(rq or '—')[:q_w - 1]:<{q_w}}",
                           ui.color(_quant_color(rq), bg, bool(rq)), maxw=max(0, card_r - q_x - 1))
                g = r["gscore"]
                if g is not None and g_x + 10 < card_r:
                    ui.bar(yy, g_x, 10, g / 100.0, T.score_color(g), bg=bg, gradient=(g >= 50))
                    ui.put(yy, g_x + 11, f"{g:5.1f}", ui.color(T.score_color(g), bg, True), maxw=max(0, card_r - g_x - 12))
                for j, b in enumerate(vis):
                    cx = cols_x + j * cw
                    s = r["scores"].get(b)
                    if s is None:
                        ui.put(yy, cx, "   –", ui.color(T.RULE, bg))
                    elif s.unit == "%":
                        ui.put(yy, cx, f"{s.value:5.1f}", ui.color(T.score_color(s.value), bg, True))
                        if s.n:
                            ui.put(yy, cx + 6, f"{s.n}", ui.color(T.DIM, bg))
                    elif s.unit == "tok":
                        ui.put(yy, cx, f"{s.value:.0f}t", ui.color(T.CYAN, bg, True))
                    else:
                        ui.put(yy, cx, f"{s.value:.1f}{'' if s.unit == 'tok/s' else s.unit}", ui.color(T.CYAN, bg, True))
                if yy + 1 < table_y + table_h - 1:
                    extra_bits = [x for x in (
                        r.get("quant") if not has_q else "",
                        r.get("adapter") or "",
                        f"[{r['harness']}]" if r.get("harness") else "",
                    ) if x]
                    tag_note = (" · " + " ".join(extra_bits)) if extra_bits else ""
                    ui.put(yy + 1, x0 + 6, f"{r['when']}{tag_note} · {r['runs']} run(s) · {r['n_bench']} benchmarks",
                           ui.color(T.DIM), maxw=max(0, card_r - x0 - 7))
            ui.scrollbar(table_y + 2, w - 3, max(1, table_h - 3), top, rows_vis, len(models))

            if not models:
                msg = "No models match that filter (press / to clear)." if flt else "No results yet. Run something (2 · Run): the leaderboard updates automatically."
                ui.put(table_y + 3, 5, msg, ui.color(T.MUTED), maxw=max(0, card_r - 6))
            elif preview_h:
                r = models[sel]
                py = table_y + table_h
                pct_scores = [(b, s.value) for b, s in r["scores"].items() if s.unit == "%"]
                perf_scores = [f"{_short(b)} {s.value:.1f}{'' if s.unit == 'tok/s' else s.unit}"
                               for b, s in r["scores"].items() if s.unit != "%"]
                if r["gscore"] is not None and leader_g is not None:
                    delta = "★ leader" if sel == 0 else f"{r['gscore'] - leader_g:+.1f} vs #1"
                    g_badge = f"gScore {r['gscore']:.1f} · {delta}"
                else:
                    g_badge = "no % scores"
                ui.card(py, 2, preview_h, w - 4, title=f"#{sel + 1} · {r['model']}", color=T.RULE,
                        title_color={0: T.YELLOW, 1: T.CYAN, 2: T.CORAL}.get(sel, T.VIOLET),
                        badge=g_badge, badge_color=T.score_color(r["gscore"]))
                if pct_scores:
                    best = max(pct_scores, key=lambda kv: kv[1])
                    worst = min(pct_scores, key=lambda kv: kv[1])
                    cx = 4
                    cx += ui.put(py + 1, cx, "best ", ui.color(T.DIM))
                    cx += ui.put(py + 1, cx, f"{best[0]} {best[1]:.1f}%", ui.color(T.score_color(best[1]), None, True))
                    cx += ui.put(py + 1, cx, "   ·   lowest ", ui.color(T.DIM))
                    ui.put(py + 1, cx, f"{worst[0]} {worst[1]:.1f}%", ui.color(T.score_color(worst[1]), None, True),
                           maxw=max(0, w - cx - 4))
                if perf_scores:
                    ui.put(py + 2, 4, "perf  " + "  ·  ".join(perf_scores), ui.color(T.CYAN), maxw=max(0, w - 8))
                else:
                    ui.put(py + 2, 4, "press Enter for full benchmark breakdown · R for Markdown/HTML report", ui.color(T.DIM), maxw=max(0, w - 8))

            p = leaderboard.paths(self.s)
            ui.put(h - 3, 3, f"files: {p['md']} · .html · .json", ui.color(T.DIM), maxw=max(0, w - 6))
            ui.footer("Enter breakdown · H matrix · M run matrix · R report · ←→ cols · / filter · c comp · o HTML · q back")
            self.scr.refresh()
            k = ui.wait_key()
            if k is None or k == "RESIZE":
                continue
            if k in ("q", ESC, SIGINT):
                return
            if k in ("H", "\t"):
                show_matrix = True
                continue
            if k == "M":
                self.screen_matrix()
                lb = leaderboard.build(R.list_runs(self.s.results_path), comparable)
                continue
            if k == "R":
                md_p, html_p = report.write_report(self.s)
                ui.notify(f"✔ wrote {md_p.name} & {html_p.name}", T.GREEN, 3.0)
                continue
            if k == "m":
                self.screen_metrics()
                continue
            if k == "/":
                f = ui.edit("Filter leaderboard", flt, hint="matches model, quant, adapter or harness; empty clears")
                if f is not None:
                    flt = f.lower().strip()
                    sel = top = 0
                continue
            if k == curses.KEY_RIGHT and benches:
                left = min(max(0, len(benches) - 1), left + 1)
            elif k == curses.KEY_LEFT:
                left = max(0, left - 1)
            elif k in (curses.KEY_UP, "k") and models:
                sel = (sel - 1) % len(models)
            elif k in (curses.KEY_DOWN, "j") and models:
                sel = (sel + 1) % len(models)
            elif k == curses.KEY_PPAGE and models:
                sel = max(0, sel - rows_vis)
            elif k == curses.KEY_NPAGE and models:
                sel = min(len(models) - 1, sel + rows_vis)
            elif k in ENTER and models:
                self._model_breakdown(models[sel], lb["benches"], lb["perf"])
            elif k == "c":
                comparable = not comparable
                lb = leaderboard.build(R.list_runs(self.s.results_path), comparable)
                sel = top = 0
                ui.notify("comparable-only mode " + ("ON" if comparable else "OFF"), T.YELLOW, 1.8)
            elif k == "w":
                leaderboard.write(self.s, comparable)
                ui.notify("✔ wrote leaderboard.md / .html / .json", T.GREEN)
            elif k == "o":
                leaderboard.write(self.s, comparable)
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                with contextlib.suppress(OSError):
                    subprocess.Popen([opener, str(p["html"])], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                ui.notify("opened leaderboard.html in browser", T.CYAN)

    def _draw_harness_matrix(self, lb: dict, h: int, w: int):
        """Renders the Adapter × Harness heatmap matrix (with Code Mode vs Normal & KV cache hit %) inside the TUI."""
        ui = self.ui
        mat = lb.get("matrix") or {}
        ads = mat.get("adapters") or []
        hrs = mat.get("harnesses") or []
        cells = mat.get("cells") or {}
        box_h = max(6, h - 6)
        ds_badge = ", ".join(mat.get("datasets") or []) or "no harness runs yet"
        ui.card(3, 2, box_h, w - 4, title="Adapter × Harness Heatmap Matrix", color=T.RULE, title_color=T.PINK,
                badge=ds_badge, badge_color=T.CYAN)
        if not ads or not hrs:
            ui.put(5, 5, "No Harbor harness runs yet. Press M to run an Adapter × Harness matrix (or 2 · Run → tier: harness).",
                   ui.color(T.MUTED), maxw=max(0, w - 10))
            return
        ad_w = min(18, max(12, max(len(a) for a in ads) + 2))
        col_w = max(20, min(30, (w - 8 - ad_w) // max(1, len(hrs))))
        hy = 4
        hdr_ad = "Adapter / Harness"
        ui.put(hy, 4, f"{hdr_ad:<{ad_w}}", ui.color(T.DIM, None, True))
        for j, hr in enumerate(hrs):
            cx = 4 + ad_w + j * col_w
            if cx + 6 < w - 3:
                ui.put(hy, cx, hr.upper()[:col_w - 2], ui.color(T.CYAN, None, True), maxw=max(0, w - cx - 4))
        for i, ad in enumerate(ads):
            ry = 6 + i * 4
            if ry + 2 >= 3 + box_h - 1:
                break
            ui.put(ry, 4, ad[:ad_w - 1], ui.color(_adapter_color(ad), None, True))
            for j, hr in enumerate(hrs):
                cx = 4 + ad_w + j * col_w
                if cx + 6 >= w - 3:
                    break
                c = cells.get(f"{ad}||{hr}")
                if not c:
                    ui.put(ry, cx, "–", ui.color(T.DIM))
                    continue
                pct = c.get("resolved_pct")
                col = T.score_color(pct) if pct is not None else T.DIM
                pct_s = f"{pct:5.1f}% ({c['resolved']}/{c['total']})" if pct is not None else "–"
                ui.put(ry, cx, pct_s, ui.color(col, None, True), maxw=max(0, col_w - 2))
                tok_s = f"{c['avg_tokens']:.0f}t" if c.get("avg_tokens") is not None else "–t"
                tc_s = f"{c['avg_tool_calls']:.1f}tc" if c.get("avg_tool_calls") is not None else "–tc"
                base_c = cells.get(f"base||{hr}")
                d_s = ""
                if ad != "base" and base_c and base_c.get("resolved_pct") is not None and pct is not None:
                    d_s = f" {pct - base_c['resolved_pct']:+.0f}"
                ui.put(ry + 1, cx, f"{tok_s} · {tc_s}{d_s}", ui.color(T.MUTED), maxw=max(0, col_w - 2))
                cm = c.get("avg_code_mode_calls")
                nm = c.get("avg_normal_tool_calls")
                kv = c.get("cache_hit_rate_pct")
                extra = []
                if cm is not None and nm is not None and (cm > 0 or nm > 0):
                    extra.append(f"c{cm:.0f}/n{nm:.0f}")
                if kv is not None:
                    extra.append(f"KV {kv:.0f}%")
                if extra and ry + 2 < 3 + box_h - 1:
                    ui.put(ry + 2, cx, " · ".join(extra), ui.color(T.DIM), maxw=max(0, col_w - 2))
