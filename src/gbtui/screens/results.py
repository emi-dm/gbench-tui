"""Results screen: failure taxonomy, token economy, notes/tags, archiving, cleanup, and Compare A vs B split-view diff."""
from __future__ import annotations

import curses
import json
import shutil

from .. import compare as CMP, leaderboard, results as R
from .. import theme as T
from ..ui import ESC, SIGINT, UI, dur
from .common import ENTER, _clamp_top, _quant_color


class ResultsMixin:
    def screen_results(self):
        ui, sel, top, flt = self.ui, 0, 0, ""
        show_archived = False
        all_runs = ui.busy("reading results…", lambda: R.list_runs(self.s.results_path, include_archived=True))
        while True:
            pool = all_runs if show_archived else [r for r in all_runs if not r.archived]
            runs = [
                r for r in pool
                if not flt or flt in (
                    f"{r.model} {r.base_model} {r.quant} {r.adapter} {r.eval_harness} {r.tier} {r.status} "
                    f"{r.headline} {r.id} {r.notes} {' '.join(r.tags)}"
                ).lower()
            ]
            h, w = ui.size()
            self.scr.erase()
            models_n = len({(r.base_model.lower(), r.quant, r.adapter) for r in pool})
            ok_n = sum(1 for r in pool if r.status == "ok")
            arch_n = sum(1 for r in all_runs if r.archived)
            sub = f"{len(pool)} runs · {models_n} variant(s) · {ok_n} ok"
            if arch_n:
                sub += f" · {arch_n} archived" + (" (shown)" if show_archived else "")
            if flt:
                sub += f" · /{flt}"
            rbadge = (f"◉ {len(self.marked)} marked", T.PINK) if self.marked else None
            ui.header("Results", sub, right_badge=rbadge)

            preview_h = 4 if (runs and h >= 18) else 0
            table_y = 3
            table_h = max(4, h - table_y - 2 - preview_h)
            view = max(1, table_h - 3)
            sel = max(0, min(sel, len(runs) - 1)) if runs else 0
            top = _clamp_top(top, sel, view, len(runs))
            card_r = w - 3
            show_q = w >= 96
            m_w = 16 if show_q else 18
            q_w = 9 if show_q else 0
            c_m = 23
            c_q = c_m + m_w
            c_tier = c_q + q_w
            c_st = c_tier + 10
            c_tm = c_st + 10
            c_hl = c_tm + 8

            ui.card(table_y, 2, table_h, w - 4, title="Run History", color=T.RULE, title_color=T.SKY,
                    badge=f"{len(runs)} shown" if flt else str(self.s.results_path), badge_color=T.DIM)
            hdr = f"{'date':<17}{'model':<{m_w}}" + (f"{'quant':<{q_w}}" if show_q else "") + f"{'tier':<10}{'status':<10}{'time':<8}headline"
            ui.put(table_y + 1, 6, hdr, ui.color(T.DIM), maxw=max(0, card_r - 7))
            if not runs:
                msg = "No runs match that filter." if flt else "No runs yet — start one in 2 · Run."
                ui.put(table_y + 2, 6, msg, ui.color(T.MUTED), maxw=max(0, card_r - 7))
            for i in range(view):
                j = top + i
                if j >= len(runs):
                    break
                r = runs[j]
                yy = table_y + 2 + i
                if j == sel:
                    ui.sel_bar(yy, 3, w - 6, T.tier_color(r.tier))
                bg = T.SEL if j == sel else None
                flag = "◉" if r.id in self.marked else ("▣" if r.archived else ("✎" if (r.notes or r.tags) else " "))
                fcol = T.PINK if r.id in self.marked else (T.DIM if r.archived else T.YELLOW)
                ui.put(yy, 4, flag, ui.color(fcol, bg, True))
                ui.put(yy, 6, f"{r.when:<17}", ui.color(T.MUTED, bg), maxw=max(0, card_r - 7))
                m_disp = r.base_model if show_q else r.model_label
                ui.put(yy, c_m, f"{m_disp[:m_w - 1]:<{m_w}}", ui.color(T.LIGHT if j == sel else T.SOFT, bg, j == sel),
                       maxw=max(0, card_r - c_m - 1))
                if show_q:
                    ui.put(yy, c_q, f"{(r.quant or '—')[:q_w - 1]:<{q_w}}",
                           ui.color(_quant_color(r.quant), bg, bool(r.quant)), maxw=max(0, card_r - c_q - 1))
                ui.soft_pill(yy, c_tier, f"{r.tier[:6].upper():<6}", fg=T.tier_color(r.tier), bg=T.BAR_ALT, bold=True)
                s_icon = {"ok": "✔", "failed": "✗", "cancelled": "■"}.get(r.status, "·")
                ui.put(yy, c_st, f"{s_icon} {r.status:<8}", ui.color(T.status_color(r.status), bg), maxw=max(0, card_r - c_st - 1))
                ui.put(yy, c_tm, f"{dur(r.duration) if r.duration else '–':<8}", ui.color(T.DIM, bg), maxw=max(0, card_r - c_tm - 1))
                hroom = max(0, card_r - c_hl - 1)
                htxt = r.headline
                if len(htxt) > hroom and hroom > 2:
                    htxt = htxt[: hroom - 1].rstrip(" ·") + "…"
                ui.segs(yy, c_hl, self._headline_segs(htxt, bg=bg, failed=(r.status == "failed")), maxw=hroom)
            ui.scrollbar(table_y + 2, w - 3, view, top, view, len(runs))

            if preview_h and runs:
                r = runs[sel]
                py = table_y + table_h
                badge_p = ("ARCHIVED · " if r.archived else "") + r.when
                ui.card(py, 2, preview_h, w - 4, title=r.id, color=T.RULE, title_color=T.tier_color(r.tier),
                        badge=badge_p, badge_color=T.DIM)
                ui.segs(py + 1, 4, self._headline_segs(r.headline, failed=(r.status == "failed")), maxw=max(0, w - 8))
                note_line = ""
                if r.tags:
                    note_line += " ".join(f"#{t}" for t in r.tags) + "  "
                if r.notes:
                    note_line += f"✎ {r.notes}"
                ui.put(py + 2, 4, note_line or str(r.path), ui.color(T.YELLOW if note_line else T.DIM), maxw=max(0, w - 8))

            ui.footer("Enter open · d compare A/B · n note/tags · a archive · A show arch · C clean · m mark · x export · D del · q back")
            self.scr.refresh()
            k = ui.wait_key()
            if k is None or k == "RESIZE":
                continue
            if k in ("q", ESC, SIGINT):
                return
            if k == "/":
                f = ui.edit("Filter runs", flt, hint="matches model, tier, status, score, notes or #tags; empty clears")
                if f is not None:
                    flt = f.lower().strip()
                    sel = top = 0
                continue
            if k == "A":
                show_archived = not show_archived
                sel = top = 0
                ui.notify("showing archived runs" if show_archived else "hiding archived runs", T.CYAN, 1.8)
                continue
            if k == "C":
                all_runs = self._cleanup_runs(all_runs)
                sel = top = 0
                continue
            if not runs:
                continue
            if k in (curses.KEY_UP, "k"):
                sel = (sel - 1) % len(runs)
            elif k in (curses.KEY_DOWN, "j"):
                sel = (sel + 1) % len(runs)
            elif k == curses.KEY_NPAGE:
                sel = min(len(runs) - 1, sel + view)
            elif k == curses.KEY_PPAGE:
                sel = max(0, sel - view)
            elif k in ENTER:
                self.run_detail(runs[sel])
            elif k == "d":
                marked_runs = [r for r in pool if r.id in self.marked]
                if len(marked_runs) == 2:
                    self.screen_compare(marked_runs[0], marked_runs[1])
                else:
                    run_a = runs[sel]
                    others = [(r, f"{r.when} · {r.model_label}", f"{r.tier} · {r.headline}") for r in pool if r.id != run_a.id]
                    if not others:
                        ui.message("Need at least 2 runs to compare A vs B.", title="Compare A vs B", color=T.YELLOW)
                    else:
                        run_b = ui.choose(f"Compare {run_a.model_label} against…", others)
                        if run_b is not None:
                            self.screen_compare(run_a, run_b)
            elif k == "n":
                r = runs[sel]
                new_note = ui.edit(f"Note for {r.id}", r.notes, hint="e.g. ctx 32k + flash-attn")
                if new_note is not None:
                    new_tags = ui.edit("Tags (comma or space separated)", ", ".join(r.tags), hint="e.g. baseline, q4, thinking")
                    tags_list = [t.strip().lstrip("#") for t in new_tags.replace(",", " ").split() if t.strip()] if new_tags is not None else r.tags
                    R.save_meta(r, notes=new_note.strip(), tags=tags_list)
                    ui.notify("✔ saved note & tags", T.GREEN, 2.0)
            elif k == "a":
                r = runs[sel]
                R.save_meta(r, archived=not r.archived)
                leaderboard.write(self.s)
                self.invalidate_cache()
                ui.notify(("archived " if r.archived else "restored ") + r.id, T.YELLOW, 2.0)
            elif k == "m":
                rid = runs[sel].id
                self.marked ^= {rid}
                ui.notify(f"{'◉ marked' if rid in self.marked else 'unmarked'} {rid} ({len(self.marked)} total)", T.PINK, 1.8)
            elif k == "l":
                self.show_log(runs[sel])
            elif k == "x":
                chosen = [r for r in all_runs if r.id in self.marked] or [runs[sel]]
                self.do_export(chosen)
            elif k == "D":
                target = runs[sel]
                if ui.confirm(f"Delete run {target.id} and its files?"):
                    shutil.rmtree(target.path, ignore_errors=True)
                    all_runs = [r for r in all_runs if r.id != target.id]
                    self.marked.discard(target.id)
                    sel = max(0, sel - 1)
                    leaderboard.write(self.s)
                    self.invalidate_cache()
                    ui.notify(f"deleted run {target.id}", T.YELLOW)

    def _cleanup_runs(self, all_runs: list) -> list:
        ui = self.ui
        broken = [r for r in all_runs if r.status in ("failed", "cancelled") or not r.scores]
        # Older duplicates: keep latest ok run per (base_model, quant, adapter, eval_harness, tier)
        seen: set[tuple] = set()
        dups = []
        for r in all_runs:
            if r.status != "ok":
                continue
            key = (r.base_model.lower(), r.quant, r.adapter, r.eval_harness, r.tier, r.tool_mode)
            if key in seen:
                dups.append(r)
            else:
                seen.add(key)
        choice = ui.choose("Cleanup Results", [
            ("broken", f"Delete {len(broken)} failed / cancelled / empty run(s)", "keeps all successful runs"),
            ("dups", f"Archive {len(dups)} older duplicate run(s)", "keeps the latest ok run per variant + tier"),
        ])
        if choice == "broken" and broken:
            if ui.confirm(f"Permanently delete {len(broken)} failed/empty run(s)?"):
                for r in broken:
                    shutil.rmtree(r.path, ignore_errors=True)
                    self.marked.discard(r.id)
                gone = {r.id for r in broken}
                all_runs = [r for r in all_runs if r.id not in gone]
                leaderboard.write(self.s)
                self.invalidate_cache()
                ui.notify(f"✔ deleted {len(broken)} failed/empty run(s)", T.GREEN)
        elif choice == "dups" and dups:
            for r in dups:
                R.save_meta(r, archived=True)
            leaderboard.write(self.s)
            self.invalidate_cache()
            ui.notify(f"✔ archived {len(dups)} older duplicate run(s)", T.GREEN)
        return all_runs

    def show_log(self, run):
        p = run.path / "gbench.log"
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = "(no gbench.log in this run)"
        styled = [self._log_segs(ln) for ln in text.splitlines()]
        self.ui.pager(f"Log · {run.id}", styled, sub=str(p))

    def run_detail(self, run):
        ui, sel, top = self.ui, 0, 0
        diag = ui.busy("analyzing run diagnostics…", CMP.diagnose_run, run)
        while True:
            h, w = ui.size()
            self.scr.erase()
            ui.header(f"Run · {run.id}", f"{run.model_label} · {run.tier} · {run.when}" + (f" · {dur(run.duration)}" if run.duration else ""),
                      crumbs=["Results", run.id],
                      right_badge=(f"● {run.status.upper()}", T.status_color(run.status)))
            spec = run.meta.get("spec") or {}
            q_info = f"  · quant {run.quant}" if run.quant else ""
            ad_info = f"  · adapter {run.adapter}" if run.adapter else ""
            info = [
                ("endpoint", f"{run.endpoint.get('base_url', '?')}  · model {run.endpoint.get('model', '?')}{q_info}{ad_info}"),
                ("settings", f"limit {spec.get('eval_limit')} · max tok {spec.get('max_output_tokens')} · thinking {'on' if spec.get('thinking') else 'off'} · mode {run.tool_mode}"),
            ]
            if run.notes or run.tags:
                tag_s = " ".join(f"#{t}" for t in run.tags)
                info.append(("notes", f"{tag_s}  {run.notes}".strip()))
            if run.meta.get("error"):
                info.append(("error", run.meta["error"]))
            meta_h = len(info) + 2
            card_r = w - 3
            ui.card(3, 2, meta_h, w - 4, title="Run Metadata", color=T.RULE, title_color=T.VIOLET,
                    badge=run.tier.upper(), badge_color=T.tier_color(run.tier))
            for idx, (k_, v) in enumerate(info):
                ui.put(4 + idx, 4, f"{k_:<10}", ui.color(T.DIM))
                ui.put(4 + idx, 15, v, ui.color(T.RED if k_ == "error" else (T.YELLOW if k_ == "notes" else T.SOFT)),
                       maxw=max(0, card_r - 16))

            # Diagnostics strip: Failure Taxonomy + Token Economy OR Harbor Code Mode / KV Cache telemetry
            dy = 3 + meta_h + 1
            he = run.harness_entry
            diag_h = 0
            if he and h >= 22:
                diag_h = 5
                ui.card(dy, 2, diag_h, w - 4, title=f"Harbor Telemetry · {run.eval_harness} ({run.tool_mode})",
                        color=T.RULE, title_color=T.PINK,
                        badge=f"KV cache {he['cache_hit_rate_pct']:.0f}%" if he.get("cache_hit_rate_pct") is not None else "",
                        badge_color=T.GREEN)
                cm = he.get("avg_code_mode_calls")
                nm = he.get("avg_normal_tool_calls")
                cm_r = he.get("code_mode_ratio_pct")
                line1 = (
                    f"tool calls: avg {he.get('avg_tool_calls') or 0:.1f}  "
                    f"(code_mode {cm or 0:.1f} vs normal {nm or 0:.1f}"
                    + (f" · {cm_r:.0f}% code" if cm_r is not None else "") + ")"
                    + (f"  · tool errors {he['tool_error_rate_pct']:.1f}%" if he.get("tool_error_rate_pct") is not None else "")
                )
                t_dist = he.get("tool_distribution") or {}
                top_tools = ", ".join(f"{k}:{v}" for k, v in list(t_dist.items())[:6]) or "—"
                line2 = (
                    f"steps: {he.get('avg_steps') or 0:.1f}  · tok/turn: {he.get('tokens_per_turn') or 0:.0f}  "
                    f"· top tools: {top_tools}"
                )
                ui.put(dy + 1, 4, line1, ui.color(T.CYAN), maxw=max(0, w - 8))
                ui.put(dy + 2, 4, line2, ui.color(T.SOFT), maxw=max(0, w - 8))
                if he.get("tokens_per_resolved") is not None:
                    ui.put(dy + 3, 4, f"efficiency: {he['tokens_per_resolved']:,.0f} tokens per resolved task",
                           ui.color(T.GREEN), maxw=max(0, w - 8))
            elif diag.get("total", 0) > 0 and h >= 22:
                diag_h = 4
                fails = diag["causes"]
                eco_badge = f"{diag['gscore_per_1k_tokens']:.2f} gScore/1k tok" if diag.get("gscore_per_1k_tokens") else f"{diag['total_tokens']:,} tok"
                ui.card(dy, 2, diag_h, w - 4, title="Diagnostics · Failure Taxonomy & Token Economy",
                        color=T.RULE, title_color=T.YELLOW, badge=eco_badge, badge_color=T.CYAN)
                fail_txt = (
                    f"failures ({diag['failed']}): "
                    f"truncated={fails['truncated']} · tool/format={fails['tool_format']} · "
                    f"timeout/err={fails['timeout_error']} · wrong={fails['wrong_answer']}"
                )
                ap = f"{diag['avg_tokens_pass']:.0f}" if diag.get("avg_tokens_pass") is not None else "—"
                af = f"{diag['avg_tokens_fail']:.0f}" if diag.get("avg_tokens_fail") is not None else "—"
                eco_txt = f"token economy: avg pass {ap} tok · avg fail {af} tok · total {diag['total_tokens']:,} tok"
                ui.put(dy + 1, 4, fail_txt, ui.color(T.YELLOW if diag["failed"] else T.GREEN), maxw=max(0, w - 8))
                ui.put(dy + 2, 4, eco_txt, ui.color(T.SOFT), maxw=max(0, w - 8))

            sc = run.scores
            sy = dy + (diag_h + 1 if diag_h else 0)
            sh = max(4, h - sy - 2)
            view = max(1, sh - 3)
            sel = max(0, min(sel, len(sc) - 1)) if sc else 0
            top = _clamp_top(top, sel, view, len(sc))
            pct_vals = [s.value for s in sc if s.unit == "%" and s.value is not None]
            avg_badge = f"mean {sum(pct_vals) / len(pct_vals):.1f}%" if pct_vals else f"{len(sc)} metric(s)"
            avg_col = T.score_color(sum(pct_vals) / len(pct_vals)) if pct_vals else T.DIM
            ui.card(sy, 2, sh, w - 4, title="Scores", color=T.RULE, title_color=T.CYAN,
                    badge=avg_badge, badge_color=avg_col)
            ui.put(sy + 1, 5, f"{'kind':<8}{'benchmark':<26}{'score':<29}{'n':<9}detail",
                   ui.color(T.DIM), maxw=max(0, card_r - 6))
            if not sc:
                ui.put(sy + 2, 5, "No scores in this run (press l to inspect gbench.log).",
                       ui.color(T.MUTED), maxw=max(0, card_r - 6))
            for i in range(view):
                j = top + i
                if j >= len(sc):
                    break
                s = sc[j]
                yy = sy + 2 + i
                sc_col = T.score_color(s.value) if (s.unit == "%" and s.value is not None) else T.CYAN
                if j == sel:
                    ui.sel_bar(yy, 3, w - 6, sc_col)
                bg = T.SEL if j == sel else None
                ui.put(yy, 5, f"{s.kind:<8}", ui.color(T.tier_color(s.kind if s.kind != "eval" else "evals"), bg, True))
                ui.put(yy, 13, f"{s.name[:25]:<26}", ui.color(T.LIGHT, bg, True), maxw=max(0, card_r - 14))
                if s.value is None:
                    ui.put(yy, 39, f"{s.status or 'no score'}", ui.color(T.RED, bg), maxw=max(0, card_r - 40))
                elif s.unit == "%":
                    bw = max(6, min(20, card_r - 49))
                    if bw >= 6:
                        ui.bar(yy, 39, bw, s.value / 100.0, sc_col, bg=bg, gradient=(s.value >= 55))
                    ui.put(yy, 40 + bw, f"{s.value:5.1f}%", ui.color(sc_col, bg, True), maxw=max(0, card_r - 41 - bw))
                    if 49 + bw < card_r:
                        ui.put(yy, 49 + bw, f"{s.correct if s.correct is not None else '?'}/{s.n}",
                               ui.color(T.SOFT, bg), maxw=max(0, card_r - 50 - bw))
                else:
                    ui.put(yy, 39, f"{s.value:.2f} {s.unit}", ui.color(T.CYAN, bg, True), maxw=max(0, card_r - 40))
                    if s.n and 69 < card_r:
                        ui.put(yy, 69, f"n={s.n}", ui.color(T.MUTED, bg), maxw=max(0, card_r - 70))
                if 78 < card_r:
                    det = s.detail or ("" if s.comparable is None else ("★ comparable" if s.comparable else "subset (not comparable)"))
                    ui.put(yy, 78, det, ui.color(T.MUTED, bg), maxw=max(0, card_r - 79))
            ui.scrollbar(sy + 2, w - 3, view, top, view, len(sc))
            ui.footer("Enter samples · f failures only · l log · x export · m metrics · q back")
            self.scr.refresh()
            k = ui.wait_key()
            if k is None or k == "RESIZE":
                continue
            if k in ("q", ESC, SIGINT):
                return
            if k in (curses.KEY_UP, "k") and sc:
                sel = (sel - 1) % len(sc)
            elif k in (curses.KEY_DOWN, "j") and sc:
                sel = (sel + 1) % len(sc)
            elif k == curses.KEY_NPAGE and sc:
                sel = min(len(sc) - 1, sel + view)
            elif k == curses.KEY_PPAGE and sc:
                sel = max(0, sel - view)
            elif k in ENTER and sc:
                self.samples(run, sc[sel], False)
            elif k == "f" and sc:
                self.samples(run, sc[sel], True)
            elif k == "l":
                self.show_log(run)
            elif k == "x":
                self.do_export([run])
            elif k == "m":
                focus = sc[sel].name if sc else None
                self.screen_metrics(initial_tab="bench" if focus else "core", focus_name=focus)

    def screen_compare(self, run_a, run_b):
        """Compare Run A vs Run B: score deltas, regressions, improvements, and split-view sample diff."""
        ui = self.ui
        diff = ui.busy(f"comparing {run_a.model_label} vs {run_b.model_label}…", CMP.compare_runs, run_a, run_b)
        flt = "diffs"  # diffs | reg | imp | all
        sel, top = 0, 0
        while True:
            all_rows = diff.samples
            if flt == "reg":
                rows = [x for x in all_rows if x.verdict == "regression"]
            elif flt == "imp":
                rows = [x for x in all_rows if x.verdict == "improvement"]
            elif flt == "diffs":
                rows = [x for x in all_rows if x.verdict in ("regression", "improvement")]
            else:
                rows = all_rows

            h, w = ui.size()
            self.scr.erase()
            rbadge = (f"▼ {diff.regressions} reg · ▲ {diff.improvements} imp",
                      T.RED if diff.regressions > diff.improvements else T.GREEN)
            ui.header("Compare A vs B", f"A: {run_a.model_label} ({run_a.when})   vs   B: {run_b.model_label} ({run_b.when})",
                      crumbs=["Results", "Compare"], right_badge=rbadge)

            sum_h = min(6, max(4, len(diff.score_deltas) + 3))
            ui.card(3, 2, sum_h, w - 4, title="Score & Token Economy Summary", color=T.RULE, title_color=T.VIOLET,
                    badge=f"filter: {flt.upper()} ({len(rows)}/{len(all_rows)})", badge_color=T.CYAN)
            for idx, sc in enumerate(diff.score_deltas[: max(1, sum_h - 3)]):
                av = f"{sc['a']:.1f}" if sc["a"] is not None else "—"
                bv = f"{sc['b']:.1f}" if sc["b"] is not None else "—"
                dv = f"{sc['delta']:+.1f}" if sc["delta"] is not None else "—"
                dcol = T.GREEN if (sc["delta"] or 0) > 0.05 else (T.RED if (sc["delta"] or 0) < -0.05 else T.MUTED)
                ui.put(4 + idx, 4, f"{sc['name'][:24]:<25} A={av:>6}  B={bv:>6}  Δ=", ui.color(T.SOFT))
                ui.put(4 + idx, 46, f"{dv} {sc['unit']}", ui.color(dcol, None, True), maxw=max(0, w - 50))
            da, db = diff.diag_a, diff.diag_b
            if sum_h >= 4:
                ey = 3 + sum_h - 2
                eco = (f"A: {da['passed']}/{da['total']} ok · trunc={da['causes']['truncated']} · {da['total_tokens']:,}t   │   "
                       f"B: {db['passed']}/{db['total']} ok · trunc={db['causes']['truncated']} · {db['total_tokens']:,}t")
                ui.put(ey, 4, eco, ui.color(T.MUTED), maxw=max(0, w - 8))

            ly = 3 + sum_h + 1
            lh = max(5, h - ly - 2)
            view = max(1, lh - 3)
            sel = max(0, min(sel, len(rows) - 1)) if rows else 0
            top = _clamp_top(top, sel, view, len(rows))
            ui.card(ly, 2, lh, w - 4, title="Sample-by-Sample Diff (Enter for split view)", color=T.RULE, title_color=T.PINK)
            ui.put(ly + 1, 4, f"{'kind':<12}{'suite':<14}{'id':<20}{'A → B':<10}{'Δ tok':<10}prompt", ui.color(T.DIM), maxw=max(0, w - 8))
            if not rows:
                ui.put(ly + 2, 4, "No samples in this filter (press f to cycle diffs / reg / imp / all).", ui.color(T.MUTED), maxw=max(0, w - 8))
            for i in range(view):
                j = top + i
                if j >= len(rows):
                    break
                x = rows[j]
                yy = ly + 2 + i
                k_col = {"regression": T.RED, "improvement": T.GREEN, "both_pass": T.DIM, "both_fail": T.YELLOW}.get(x.verdict, T.MUTED)
                if j == sel:
                    ui.sel_bar(yy, 3, w - 6, k_col)
                bg = T.SEL if j == sel else None
                k_lbl = {"regression": "▼ REGRESS", "improvement": "▲ IMPROVE", "both_pass": "= pass", "both_fail": "= fail"}.get(x.verdict, x.verdict)
                ca = (x.a or {}).get("correct")
                cb = (x.b or {}).get("correct")
                am = "✔" if ca else ("✗" if ca is False else "–")
                bm = "✔" if cb else ("✗" if cb is False else "–")
                dt = f"{x.token_delta:+d}t" if x.token_delta is not None else "—"
                pr_txt = str((x.a or x.b or {}).get("prompt") or "").replace("\n", " ")
                ui.put(yy, 4, f"{k_lbl:<12}", ui.color(k_col, bg, True))
                ui.put(yy, 16, f"{x.suite[:13]:<14}", ui.color(T.PALE, bg))
                ui.put(yy, 30, f"{x.sample_id[:19]:<20}", ui.color(T.LIGHT if j == sel else T.SOFT, bg, j == sel))
                ui.put(yy, 50, f"{am} → {bm:<5}", ui.color(k_col, bg, True))
                ui.put(yy, 60, f"{dt:<10}", ui.color(T.CYAN, bg))
                ui.put(yy, 70, pr_txt, ui.color(T.MUTED, bg), maxw=max(0, w - 74))
            ui.scrollbar(ly + 2, w - 3, view, top, view, len(rows))
            ui.footer("Enter split-view diff · f filter (diffs/reg/imp/all) · ↑↓ move · q back")
            self.scr.refresh()
            k = ui.wait_key()
            if k is None or k == "RESIZE":
                continue
            if k in ("q", ESC, SIGINT):
                return
            if k == "f":
                flt = {"diffs": "reg", "reg": "imp", "imp": "all", "all": "diffs"}[flt]
                sel = top = 0
            elif not rows:
                continue
            elif k in (curses.KEY_UP, "k"):
                sel = (sel - 1) % len(rows)
            elif k in (curses.KEY_DOWN, "j"):
                sel = (sel + 1) % len(rows)
            elif k == curses.KEY_NPAGE:
                sel = min(len(rows) - 1, sel + view)
            elif k == curses.KEY_PPAGE:
                sel = max(0, sel - view)
            elif k in ENTER:
                sel = self._sample_diff_view(rows, sel, run_a.model_label, run_b.model_label)

    def _sample_diff_view(self, rows: list, idx: int, label_a: str, label_b: str) -> int:
        """Side-by-side split view of the exact same sample in Run A (left) vs Run B (right)."""
        ui = self.ui
        scroll = 0
        while True:
            x = rows[idx]
            h, w = ui.size()
            self.scr.erase()
            k_col = {"regression": T.RED, "improvement": T.GREEN}.get(x.verdict, T.CYAN)
            ui.header(f"Diff · {x.suite} · {x.sample_id}", f"sample {idx + 1} of {len(rows)} · {x.verdict.upper()}",
                      crumbs=["Compare", x.sample_id[:20]], right_badge=(x.verdict.upper(), k_col))

            def pane_lines(item, col_w):
                if not item:
                    return [("(not present in this run)", T.DIM)]
                out = []
                ok = item.get("correct")
                v_txt = "✔ PASS" if ok else ("✗ FAIL" if ok is False else "– N/A")
                v_col = T.GREEN if ok else (T.RED if ok is False else T.DIM)
                out.append((f"{v_txt} · tok={item.get('completion_tokens') or '?'} · finish={item.get('finish_reason') or '?'}", v_col))
                if item.get("detail"):
                    for ln in UI.wrap("detail: " + str(item["detail"]), col_w):
                        out.append((ln, T.YELLOW))
                if item.get("reasoning"):
                    out.append(("── reasoning ──", T.YELLOW))
                    for raw in str(item["reasoning"]).splitlines()[:20]:
                        for ln in UI.wrap(raw, col_w) or [""]:
                            out.append((ln, T.MUTED))
                out.append(("── response ──", T.PINK))
                for raw in str(item.get("response") or "(empty)").splitlines():
                    for ln in UI.wrap(raw, col_w) or [""]:
                        out.append((ln, T.SOFT))
                return out

            pr_h = 4
            ui.card(3, 2, pr_h, w - 4, title="Prompt", color=T.RULE, title_color=T.CYAN)
            pr_txt = str((x.a or x.b or {}).get("prompt") or "(no prompt)")
            for li, ln in enumerate(UI.wrap(pr_txt, max(12, w - 8))[:2]):
                ui.put(4 + li, 4, ln, ui.color(T.LIGHT), maxw=max(0, w - 8))

            py = 3 + pr_h + 1
            ph = max(6, h - py - 2)
            half = (w - 5) // 2
            cw_a = max(10, half - 4)
            cw_b = max(10, (w - 5 - half) - 4)
            la = pane_lines(x.a, cw_a)
            lb = pane_lines(x.b, cw_b)
            max_lines = max(len(la), len(lb))
            view = max(1, ph - 2)
            scroll = max(0, min(scroll, max(0, max_lines - view)))

            ui.card(py, 2, ph, half, title=f"A · {label_a[:max(8, half - 10)]}", color=T.RULE, title_color=T.SKY)
            ui.card(py, 3 + half, ph, w - 5 - half, title=f"B · {label_b[:max(8, w - 5 - half - 10)]}", color=T.RULE, title_color=T.PINK)
            for i in range(view):
                j = scroll + i
                if j < len(la):
                    ui.put(py + 1 + i, 4, la[j][0], ui.color(la[j][1]), maxw=cw_a)
                if j < len(lb):
                    ui.put(py + 1 + i, 5 + half, lb[j][0], ui.color(lb[j][1]), maxw=cw_b)
            ui.scrollbar(py + 1, w - 3, view, scroll, view, max_lines)
            ui.footer("↑↓ scroll · n/p next/prev sample · q back")
            self.scr.refresh()
            k = ui.wait_key()
            if k is None or k == "RESIZE":
                continue
            if k in ("q", ESC, SIGINT):
                return idx
            if k in (curses.KEY_UP, "k"):
                scroll = max(0, scroll - 1)
            elif k in (curses.KEY_DOWN, "j"):
                scroll = min(max(0, max_lines - view), scroll + 1)
            elif k == curses.KEY_PPAGE:
                scroll = max(0, scroll - view)
            elif k == curses.KEY_NPAGE:
                scroll = min(max(0, max_lines - view), scroll + view)
            elif k in ("n", "]") and len(rows) > 1:
                idx = (idx + 1) % len(rows)
                scroll = 0
            elif k in ("p", "[") and len(rows) > 1:
                idx = (idx - 1) % len(rows)
                scroll = 0

    def samples(self, run, score, failures: bool):
        ui = self.ui
        all_its = R.items(run, score)
        if not all_its:
            ui.message("No per-sample records for this score.")
            return
        mode = "fail" if failures else "all"
        flt = ""
        sel, top = 0, 0
        while True:
            its = all_its
            if mode == "fail":
                its = [i for i in its if i["correct"] is False]
            elif mode == "pass":
                its = [i for i in its if i["correct"] is True]
            if flt:
                its = [i for i in its if flt in f"{i['id']} {i.get('category', '')} {i.get('prompt', '')} {i.get('detail', '')}".lower()]

            h, w = ui.size()
            self.scr.erase()
            ok = sum(1 for i in all_its if i["correct"] is True)
            bad = sum(1 for i in all_its if i["correct"] is False)
            na = sum(1 for i in all_its if i["correct"] is None)
            pass_rate = (ok * 100.0 / (ok + bad)) if (ok + bad) else 0.0
            mode_lbl = {"all": "ALL", "fail": "FAILURES ONLY", "pass": "PASSED ONLY"}[mode]
            ui.header(f"Samples · {score.name}", f"{len(all_its)} total · {ok} passed · {bad} failed · {na} N/A"
                      + (f" · /{flt}" if flt else ""),
                      crumbs=["Results", score.name],
                      right_badge=(f"{mode_lbl} ({len(its)})", T.YELLOW if mode == "fail" else T.GREEN if mode == "pass" else T.CYAN))

            bar_w = min(40, max(14, w // 3))
            ui.stacked_bar(3, 3, bar_w, [(ok, T.GREEN), (bad, T.RED), (na, T.DIM)])
            sx = 5 + bar_w
            sx += ui.soft_pill(3, sx, f"{pass_rate:.1f}%", fg=T.score_color(pass_rate), bg=T.BAR_ALT) + 1
            sx += ui.put(3, sx, f"✔ {ok}", ui.color(T.GREEN, None, True)) + 2
            sx += ui.put(3, sx, f"✗ {bad}", ui.color(T.RED, None, True)) + 2
            if na:
                ui.put(3, sx, f"– {na} N/A", ui.color(T.DIM))

            preview_h = 4 if (its and h >= 18) else 0
            list_y = 5
            list_h = max(4, h - list_y - 2 - preview_h)
            view = max(1, list_h - 3)
            sel = max(0, min(sel, len(its) - 1)) if its else 0
            top = _clamp_top(top, sel, view, len(its))
            card_r = w - 3
            det_col_w = min(38, max(24, w // 3)) if w >= 100 else 0
            det_x = (card_r - det_col_w - 1) if det_col_w else card_r

            ui.card(list_y, 2, list_h, w - 4, title="Per-Sample Verdicts", color=T.RULE, title_color=T.VIOLET,
                    badge=f"{sel + 1}/{len(its)}" if its else "0", badge_color=T.DIM)
            ui.put(list_y + 1, 6, f"{'id':<22}{'category':<15}prompt", ui.color(T.DIM), maxw=max(0, det_x - 8))
            if det_col_w:
                ui.put(list_y + 1, det_x - 2, "│ ", ui.color(T.RULE))
                ui.put(list_y + 1, det_x, "verdict detail", ui.color(T.DIM), maxw=det_col_w)
            if not its:
                ui.put(list_y + 2, 6, "No samples match this filter (press f or / to change).",
                       ui.color(T.MUTED), maxw=max(0, card_r - 7))
            for n in range(view):
                j = top + n
                if j >= len(its):
                    break
                it = its[j]
                y = list_y + 2 + n
                mark = "–" if it["correct"] is None else ("✔" if it["correct"] else "✗")
                mcol = T.DIM if it["correct"] is None else (T.GREEN if it["correct"] else T.RED)
                if j == sel:
                    ui.sel_bar(y, 3, w - 6, mcol if it["correct"] is not None else T.VIOLET)
                bg = T.SEL if j == sel else None
                ui.put(y, 4, mark, ui.color(mcol, bg, True))
                sid = str(it["id"])
                sid = sid[7:19] if sid.startswith("sha256:") else sid[:21]
                ui.put(y, 6, f"{sid:<22}", ui.color(T.LIGHT if j == sel else T.SOFT, bg, j == sel), maxw=max(0, card_r - 7))
                cat_s = str(it.get("category") or it.get("status") or "")[:14]
                ui.put(y, 28, f"{cat_s:<15}", ui.color(T.PALE, bg), maxw=max(0, card_r - 29))
                q = (it["prompt"] or "").rsplit("[user]", 1)[-1].strip().replace("\n", " ")
                det = (it["detail"] or "").replace("\n", " ")
                det = "" if det in ("ok", it.get("status")) else det
                q_max = max(0, (det_x - 45) if det_col_w else (card_r - 44))
                if q_max > 0:
                    ui.put(y, 43, q, ui.color(T.SOFT, bg), maxw=q_max)
                if det_col_w:
                    ui.put(y, det_x - 2, "│ ", ui.color(T.RULE, bg))
                    ui.put(y, det_x, det, ui.color(T.YELLOW if it["correct"] is False else T.MUTED, bg), maxw=det_col_w)
            ui.scrollbar(list_y + 2, w - 3, view, top, view, len(its))

            if preview_h and its:
                it = its[sel]
                py = list_y + list_h
                q = (it["prompt"] or "").rsplit("[user]", 1)[-1].strip().replace("\n", " ")
                det = (it["detail"] or "").replace("\n", " ")
                st_str = str(it.get("status") or "")
                ui.card(py, 2, preview_h, w - 4, title=str(it["id"])[:40], color=T.RULE,
                        title_color=T.status_color(st_str), badge=st_str, badge_color=T.status_color(st_str))
                ui.put(py + 1, 4, q or "(prompt unavailable)", ui.color(T.SOFT if q else T.DIM), maxw=max(0, w - 8))
                ui.put(py + 2, 4, det or "(no extra verdict detail)",
                       ui.color(T.YELLOW if it["correct"] is False else T.MUTED), maxw=max(0, w - 8))

            ui.footer("Enter open · f cycle filter (all/fail/pass) · / search · ↑↓ move · q back")
            self.scr.refresh()
            k = ui.wait_key()
            if k is None or k == "RESIZE":
                continue
            if k in ("q", ESC, SIGINT):
                return
            if k == "f":
                mode = {"all": "fail", "fail": "pass", "pass": "all"}[mode]
                sel = top = 0
                continue
            if k == "/":
                q_in = ui.edit("Search samples", flt, hint="matches id, category, prompt or detail; empty clears")
                if q_in is not None:
                    flt = q_in.lower().strip()
                    sel = top = 0
                continue
            if not its:
                continue
            if k in (curses.KEY_UP, "k"):
                sel = (sel - 1) % len(its)
            elif k in (curses.KEY_DOWN, "j"):
                sel = (sel + 1) % len(its)
            elif k == curses.KEY_NPAGE:
                sel = min(len(its) - 1, sel + view)
            elif k == curses.KEY_PPAGE:
                sel = max(0, sel - view)
            elif k in ENTER:
                sel = self.sample_view(its, sel)

    def _build_sample_lines(self, it, w: int):
        c = self.ui.color
        verdict = "✔ PASSED" if it["correct"] else ("– N/A" if it["correct"] is None else f"✗ {it['status'] or 'FAILED'}")
        vcol = T.GREEN if it["correct"] else (T.DIM if it["correct"] is None else T.RED)
        fail_kind = CMP.classify_failure(it)
        fk_seg = [(f"  [{fail_kind}]", c(T.YELLOW, None, True))] if fail_kind else []
        lines = [[("verdict ", c(T.DIM)), (f" {verdict} ", c(T.INK, vcol, True, auto_sel=False)),
                  *fk_seg, (f"  category: {it.get('category') or '—'}", c(T.PALE))]]
        if it.get("finish_reason") or it.get("completion_tokens"):
            lines.append([("tokens  ", c(T.DIM)), (f"{it.get('completion_tokens')} · finish {it.get('finish_reason')}", c(T.SOFT))])
        if it.get("detail"):
            for n, ln in enumerate(UI.wrap(it["detail"], max(20, w - 16))):
                lines.append([("detail  " if n == 0 else "        ", c(T.DIM)), (ln, c(T.YELLOW))])
        lines.append([("─" * max(10, w - 6), c(T.RULE_DIM))])
        lines.append([("", 0)])

        def block(title, text, col):
            if not text:
                return
            lines.append([("╭─ ", c(col)), (title, c(col, None, True))])
            for ln in str(text).splitlines() or [""]:
                for part in UI.wrap(ln, max(20, w - 8)) or [""]:
                    lines.append([("│ ", c(col)), (part, c(T.SOFT))])
            lines.append([("╰─", c(col))])
            lines.append([("", 0)])

        if it.get("messages"):
            for m in it["messages"] if isinstance(it["messages"], list) else []:
                cont = m.get("content") if isinstance(m, dict) else m
                if isinstance(cont, list):
                    cont = " ".join(p.get("text", f"[{p.get('type')}]") for p in cont if isinstance(p, dict))
                block(f"{m.get('role', '?') if isinstance(m, dict) else '?'}", cont, T.CYAN)
        else:
            block("prompt", it.get("prompt") or it.get("title"), T.CYAN)
        block("reasoning", it.get("reasoning"), T.YELLOW)
        block("model response", it.get("response"), T.PINK)
        if it.get("tool_calls"):
            block("tool calls", json.dumps(it["tool_calls"], indent=2, default=str), T.VIOLET)
        gold = it.get("gold")
        if gold not in (None, ""):
            block("expected", gold if isinstance(gold, str) else json.dumps(gold, indent=2, default=str), T.GREEN)
        return lines

    def sample_view(self, its, idx: int = 0) -> int:
        if isinstance(its, dict):
            its, idx = [its], 0
        state = {"idx": idx}

        def make(i: int):
            it = its[i]
            w = self.ui.size()[1]
            return (f"Sample · {it['suite']} · {it['id']}", self._build_sample_lines(it, w),
                    f"sample {i + 1} of {len(its)}")

        def step(delta: int):
            if len(its) <= 1:
                return None
            state["idx"] = (state["idx"] + delta) % len(its)
            t, lns, sb = make(state["idx"])
            return ("replace", t, lns, sb)

        title, lines, sub = make(state["idx"])
        hints = "↑↓ scroll · n/p or [/] next/prev sample · / search · ←→ sideways · q back" if len(its) > 1 else \
                "↑↓ scroll · / search · ←→ sideways · q back"
        self.ui.pager(title, lines, sub=sub, hints=hints,
                      keys={"n": lambda: step(1), "p": lambda: step(-1), "]": lambda: step(1), "[": lambda: step(-1)})
        return state["idx"]
