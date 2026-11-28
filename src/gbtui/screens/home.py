"""Home screen: cleanly ordered workspace menu, live KPIs, host status, and instant non-hanging quit."""
from __future__ import annotations

import contextlib
import curses
import time
from pathlib import Path

from .. import __version__, harness_lens, results as R, runner, servers, sysmem
from .. import theme as T
from ..ui import ESC, PULSE, SIGINT, SPIN, UI, dur
from .common import ENTER, _clamp_top


class HomeMixin:
    # Strictly ordered digits 1–9, 0 followed by 2 clearly separated system items (d, s)
    MENU = [
        ("1", "◎", "Endpoints", "Servers, GGUF folder import (F), local discovery (D) & LoRA picker (l)"),
        ("2", "▶", "Run", "Launch Golden Set, eval suites, perf, presets (P) or Harbor harness"),
        ("3", "⚡", "Sweep & Matrix", "Quantization Sweep across GGUFs or Adapter × Harness Matrix batch"),
        ("4", "◉", "Live run", "Follow the live benchmark pipeline, sweep step and streamed logs"),
        ("5", "≡", "Results", "Failure breakdown, token economy, notes/tags and Compare A vs B diff"),
        ("6", "★", "Leaderboard", "Rankings by gScore, Adapter × Harness matrix, Pareto HTML & reports"),
        ("7", "⟐", "Harness Lens", "Inspect & compare harnesses (Pi, OpenCode, Hermes…) & launch interactive play"),
        ("8", "◈", "Suites", "Browse all eval suites by pillar and Golden Set cases with guides & needs"),
        ("9", "∑", "Metrics", "Inspect core formulas and exact scoring metrics used by each benchmark"),
        ("0", "↓", "Export", "Package runs into a Hugging Face dataset (JSONL/Parquet) or Markdown/HTML report"),
        ("d", "◇", "Sandbox", "Hardened Docker image, live containers, memory caps and pruning"),
        ("s", "•", "Settings", "16 color themes, sandbox RAM/CPU, notifications and results folder"),
    ]

    def _intro_reveal(self):
        """Quick, immediately-skippable aurora prism sweep over the wordmark on first launch."""
        h, w = self.ui.size()
        if w < 92 or h < 24:
            return
        self._refresh_summary(force=True)
        for step in range(1, 9):
            self.draw_home(0, reveal=step / 8.0)
            k = self.ui.key(24)
            if k is not None and k != "RESIZE":
                with contextlib.suppress(curses.error, TypeError, ValueError, OverflowError):
                    if isinstance(k, int):
                        curses.ungetch(k)
                    elif isinstance(k, str) and len(k) == 1:
                        curses.unget_wch(k)
                return

    def main(self):
        sel = 0
        actions = [
            self.screen_endpoints,
            self.screen_run,
            self.screen_sweep,
            self.screen_live,
            self.screen_results,
            self.screen_leaderboard,
            self.screen_harness_lens,
            self.screen_suites,
            self.screen_metrics,
            self.screen_export,
            self.screen_sandbox,
            self.screen_settings,
        ]
        key_idx = {item[0]: idx for idx, item in enumerate(self.MENU)}
        # Convenient mnemonic aliases that don't clash with menu keys
        key_idx["m"] = key_idx["9"]   # Metrics
        key_idx["h"] = key_idx["7"]   # Harness Lens
        self._intro_reveal()
        while True:
            self._refresh_summary()
            self.draw_home(sel)
            k = self.ui.key(140)
            self.tick += 1
            if k is None or k == "RESIZE":
                continue
            if k in (curses.KEY_UP, "k"):
                sel = (sel - 1) % len(self.MENU)
            elif k in (curses.KEY_DOWN, "j"):
                sel = (sel + 1) % len(self.MENU)
            elif isinstance(k, str) and k in key_idx:
                sel = key_idx[k]
                actions[sel]()
                self.invalidate_cache()
            elif k in ENTER:
                actions[sel]()
                self.invalidate_cache()
            elif k == "t":
                new_t = T.cycle_theme()
                self.s.theme = new_t
                self.save()
                info = T.theme_info(new_t)
                self.ui.notify(f"✦ Theme ({T.THEME_ORDER.index(new_t) + 1}/{len(T.THEME_ORDER)}): {info['label']} — {info['desc']}", T.PINK, 2.8)
            elif k == "T":
                prev_t = T.current_theme()
                items = [(name, T.THEMES[name]["label"], T.THEMES[name]["desc"]) for name in T.THEME_ORDER]
                cur_idx = T.THEME_ORDER.index(prev_t) if prev_t in T.THEME_ORDER else 0
                picked = self.ui.choose(
                    f"Select Color Theme ({len(T.THEME_ORDER)} palettes · live preview)",
                    items,
                    sel=cur_idx,
                    hint="↑↓ preview live · Enter apply · Esc restore",
                    on_move=T.set_theme,
                    on_redraw=lambda: self.draw_home(sel),
                )
                if picked:
                    T.set_theme(picked)
                    self.s.theme = picked
                    self.save()
                    info = T.theme_info(picked)
                    self.ui.notify(f"✦ Theme: {info['label']} — {info['desc']}", T.PINK, 2.8)
                else:
                    T.set_theme(prev_t)
            elif k == "?":
                self.show_help()
            elif k == "r":
                snap = runner.snapshot()
                if not snap["active"] and snap["result_dir"]:
                    run = R.load_run(Path(snap["result_dir"]))
                    if run:
                        self.run_detail(run)
                        self.invalidate_cache()
            elif k in ("q", ESC, SIGINT):
                if self.quit():
                    return

    def _menu_badge(self, num: str, live: dict) -> tuple[str, int]:
        ep = self.s.endpoint()
        if num == "1":
            if not ep:
                return ("none", T.YELLOW)
            pr = self.st.ep_ok.get(ep.name)
            return ("● up" if pr and pr["ok"] else "○ down" if pr else "…", T.GREEN if pr and pr["ok"] else T.DIM)
        if num == "2":
            return (self.s.run.tier, T.tier_color(self.s.run.tier))
        if num == "3":
            return (f"{len(self.s.endpoints)} eps", T.PINK if len(self.s.endpoints) > 1 else T.DIM)
        if num == "4":
            if live["active"]:
                d, t = live["progress"]
                step = f" {live['matrix_pos']}" if live.get("matrix_pos") else ""
                return (f"{SPIN[self.tick % len(SPIN)]} {d}/{t}{step}" if t else f"{SPIN[self.tick % len(SPIN)]} {live['phase']}{step}", T.PINK)
            if live["phase"] in ("done", "failed", "cancelled"):
                return (live["phase"], T.status_color(live["phase"]))
            return ("idle", T.DIM)
        if num == "5":
            n = len(self._runs_cache)
            return (f"{n} run{'s' if n != 1 else ''}", T.SKY if n else T.DIM)
        if num == "6":
            top = self._lb_cache["models"][0] if self._lb_cache["models"] else None
            if top and top["gscore"] is not None:
                return (f"★ {top['gscore']:.1f}", T.score_color(top["gscore"]))
            return ("—", T.DIM)
        if num == "7":
            return (f"{len(harness_lens.HARNESS_SPECS)} agents", T.VIOLET)
        if num == "8" and self._cat_count and (self._cat_count[0] or self._cat_count[1]):
            return (f"{self._cat_count[0]}+{self._cat_count[1]}", T.VIOLET)
        if num == "9":
            return ("core+evals", T.CORAL)
        if num == "d":
            if self.st.sbx == "ready":
                return ("● ready", T.GREEN)
            if self.st.sbx == "checking…":
                return (f"{SPIN[self.tick % len(SPIN)]} check", T.CYAN)
            return ("⚠ setup", T.YELLOW)
        if num == "s":
            return (f"✦ {T.current_theme()}", T.PALE)
        return ("", T.DIM)

    def draw_home(self, sel: int, reveal: float = 1.0):
        ui = self.ui
        h, w = ui.size()
        self.scr.erase()
        ui.header("Home", tick=self.tick)
        live = runner.snapshot()
        y = 2
        wm = T.WORDMARK
        wm_w = len(wm[0])
        two_col = w >= 78
        show_wm = w >= wm_w + 8 and (w >= 92 or h >= 30) and h >= 26
        if show_wm:
            x0 = 3
            cols_shown = max(1, int(wm_w * reveal))
            for i, row in enumerate(wm):
                for j, ch in enumerate(row[:cols_shown]):
                    if ch == " ":
                        continue
                    shade = ch in "╗╔╝╚═║"
                    col = T.shimmer_2d(i, j, len(wm), len(row), self.tick) if reveal >= 1.0 else T.grad(j, len(row))
                    ui.put(y + i, x0 + j, ch, ui.color(col, None, not shade, shade))
            ui.put(y + len(wm), x0 + 1, "terminal UI for Google DeepMind's gbench & Harbor", ui.color(T.MUTED))
            vx = x0 + 52
            vx += ui.soft_pill(y + len(wm), vx, f"v{__version__}", fg=T.CYAN, bg=T.BAR_ALT, bold=True) + 1
            if vx + 14 < w:
                ui.soft_pill(y + len(wm), vx, f"✦ {T.theme_info()['label']}", fg=T.PINK, bg=T.BAR_ALT, bold=False)
            sx = x0 + wm_w + 4
            sw = w - sx - 2
            if sw >= 34:
                self._draw_kpi_tiles(y, sx, sw, len(wm))
            y += len(wm) + 2
        else:
            ui.put(y, 3, "◆ g·bench tui", ui.color(T.PINK, None, True))
            ui.soft_pill(y, 18, f"v{__version__}", fg=T.CYAN, bg=T.BAR_ALT)
            y += 2

        left_w = min(46, max(32, int(w * 0.41))) if two_col else max(10, w - 4)
        max_menu_h = max(4, h - y - 2)
        menu_h = min(len(self.MENU) + 2, max_menu_h)
        view_m = max(1, menu_h - 2)
        m_top = _clamp_top(0, sel, view_m, len(self.MENU))
        card_y = y
        ui.card(card_y, 2, menu_h, left_w, title="Workspace", color=T.RULE, title_color=T.VIOLET, badge="1–0 · d · s", badge_color=T.DIM)
        for i in range(view_m):
            idx = m_top + i
            if idx >= len(self.MENU):
                break
            num, icon, name, _desc = self.MENU[idx]
            ry = card_y + 1 + i
            is_sel = idx == sel
            item_col = T.menu_color(idx)
            if is_sel:
                ui.sel_bar(ry, 3, left_w - 2, item_col)
            bg = T.SEL if is_sel else None
            live_icon = SPIN[self.tick % len(SPIN)] if (num == "4" and live["active"]) else icon
            ui.put(ry, 5, num, ui.color(item_col, bg, True))
            ui.put(ry, 7, live_icon, ui.color(item_col if (is_sel or (num == "4" and live["active"])) else T.DIM, bg, is_sel))
            ui.put(ry, 9, ("▸ " if is_sel else "") + name, ui.color(T.LIGHT if is_sel else T.SOFT, bg, is_sel),
                   maxw=max(0, left_w - 11))
            btxt, bcol = self._menu_badge(num, live)
            if btxt and left_w > 24:
                bx = 2 + left_w - len(btxt) - 2
                if bx > 22:
                    ui.put(ry, bx, btxt, ui.color(bcol, bg, is_sel or (num == "4" and live["active"])))
        ui.scrollbar(card_y + 1, 2 + left_w - 1, view_m, m_top, view_m, len(self.MENU))

        dy = card_y + menu_h + 1
        desc_used = 0
        if dy + 1 < h - 2:
            accent = T.menu_color(sel)
            desc_lines = UI.wrap(self.MENU[sel][3], max(12, left_w - 6))[:2]
            for i, line in enumerate(desc_lines):
                if dy + i >= h - 2:
                    break
                ui.put(dy + i, 3, "▎", ui.color(accent, None, True))
                ui.put(dy + i, 5, line, ui.color(T.MUTED), maxw=max(0, left_w - 5))
                desc_used += 1

        if two_col:
            px = left_w + 4
            pw = w - px - 2
            self.draw_status(card_y, px, pw, max_y=h - 2, live=live)
        elif dy + desc_used + 1 + 4 <= h - 2:
            self.draw_status(dy + desc_used + 1, 2, w - 4, max_y=h - 2, live=live)

        hints = "↑↓ move · Enter open · 1–0/d/s jump · t/T themes · ? help"
        if not live["active"] and live["result_dir"]:
            hints += " · r last run"
        hints += " · q quit"
        ui.footer(hints)
        self.scr.refresh()

    def _draw_kpi_tiles(self, y: int, x: int, w: int, hh: int):
        """3 jewel-toned KPI metric tiles aligned next to the GBENCH wordmark."""
        ui = self.ui
        runs = self._runs_cache
        n_runs = len(runs)
        ok_runs = sum(1 for r in runs if r.status == "ok")
        n_models = len(self._lb_cache["models"])
        top = self._lb_cache["models"][0] if n_models else None
        top_g = top["gscore"] if top else None
        ep = self.s.endpoint()

        if w >= 54:
            t1_w = max(14, (w - 2) // 3 - 2)
            t2_w = max(18, (w - 2) // 3 + 2)
            t3_w = w - t1_w - t2_w - 2
            ui.card(y, x, hh, t1_w, title="Runs", color=T.RULE, title_color=T.VIOLET, icon="≡")
            ui.put(y + 1, x + 2, f"{n_runs}", ui.color(T.LIGHT, None, True))
            ui.put(y + 1, x + 3 + len(str(n_runs)), "total", ui.color(T.DIM), maxw=max(0, t1_w - 5 - len(str(n_runs))))
            ui.put(y + 2, x + 2, f"{n_models} model{'s' if n_models != 1 else ''}", ui.color(T.PALE), maxw=t1_w - 4)
            ok_pct = int(round(ok_runs * 100 / n_runs)) if n_runs else 0
            ui.put(y + 3, x + 2, f"✔ {ok_pct}% ok" if n_runs else "ready", ui.color(T.GREEN if n_runs else T.DIM), maxw=t1_w - 4)
            if hh >= 6:
                ui.bar(y + 4, x + 2, max(4, t1_w - 4), ok_runs / n_runs if n_runs else 0.0, T.GREEN)

            x2 = x + t1_w + 1
            ui.card(y, x2, hh, t2_w, title="Leader", color=T.RULE, title_color=T.YELLOW, icon="★")
            if top and top_g is not None:
                ui.put(y + 1, x2 + 2, f"{top_g:5.1f}", ui.color(T.score_color(top_g), None, True))
                ui.put(y + 1, x2 + 8, "gScore", ui.color(T.DIM), maxw=max(0, t2_w - 10))
                ui.put(y + 2, x2 + 2, top["model"], ui.color(T.LIGHT, None, True), maxw=t2_w - 4)
                ui.put(y + 3, x2 + 2, f"{top['n_bench']} bench · {top['runs']} run(s)", ui.color(T.MUTED), maxw=t2_w - 4)
                if hh >= 6:
                    ui.bar(y + 4, x2 + 2, max(6, t2_w - 4), top_g / 100.0, T.score_color(top_g), gradient=True)
            else:
                ui.put(y + 2, x2 + 2, "no scores yet", ui.color(T.MUTED), maxw=t2_w - 4)
                ui.put(y + 3, x2 + 2, "press 2 to run", ui.color(T.DIM), maxw=t2_w - 4)

            x3 = x2 + t2_w + 1
            ui.card(y, x3, hh, t3_w, title="Target", color=T.RULE, title_color=T.CYAN, icon="◎")
            if ep:
                pr = self.st.ep_ok.get(ep.name)
                up = bool(pr and pr["ok"])
                dot = PULSE[(self.tick // 3) % len(PULSE)] if up else "○"
                ui.put(y + 1, x3 + 2, f"{dot} {ep.name}", ui.color(T.GREEN if up else T.YELLOW, None, True), maxw=t3_w - 4)
                ad_s = f" · {ep.effective_adapter}" if ep.effective_adapter else ""
                ui.put(y + 2, x3 + 2, (f"● {pr['latency_ms']}ms{ad_s}" if up else (f"GGUF{ad_s} · stopped" if ep.gguf else "○ offline")),
                       ui.color(T.GREEN if up else T.MUTED), maxw=t3_w - 4)
                ui.put(y + 3, x3 + 2, f"tier: {self.s.run.tier}", ui.color(T.tier_color(self.s.run.tier)), maxw=t3_w - 4)
                if hh >= 6:
                    sb_ok = self.st.sbx == "ready"
                    ui.put(y + 4, x3 + 2, "docker ● ready" if sb_ok else "docker ⚠ check",
                           ui.color(T.GREEN if sb_ok else T.YELLOW), maxw=t3_w - 4)
            else:
                ui.put(y + 2, x3 + 2, "none (press 1)", ui.color(T.YELLOW), maxw=t3_w - 4)
        else:
            ui.card(y, x, hh, w, title="Snapshot", color=T.RULE, title_color=T.PINK,
                    badge=f"{n_runs} runs" if n_runs else "new")
            ui.put(y + 1, x + 2, f"{'models':<10}", ui.color(T.DIM))
            ui.put(y + 1, x + 12, f"{n_models} ranked · {n_runs} runs", ui.color(T.SOFT), maxw=max(0, w - 14))
            ui.put(y + 2, x + 2, f"{'leader':<10}", ui.color(T.DIM))
            ui.put(y + 2, x + 12, top["model"] if top else "—", ui.color(T.LIGHT, None, True), maxw=max(0, w - 14))
            ui.put(y + 3, x + 2, f"{'gScore':<10}", ui.color(T.DIM))
            if top_g is not None:
                bw = max(6, min(14, w - 22))
                ui.bar(y + 3, x + 12, bw, top_g / 100.0, T.score_color(top_g), gradient=True)
                ui.put(y + 3, x + 13 + bw, f"{top_g:5.1f}", ui.color(T.score_color(top_g), None, True))
            else:
                ui.put(y + 3, x + 12, "—", ui.color(T.DIM))
            if hh >= 6:
                ui.put(y + 4, x + 2, f"{'active':<10}", ui.color(T.DIM))
                ui.put(y + 4, x + 12, ep.name if ep else "none", ui.color(T.CYAN if ep else T.YELLOW), maxw=max(0, w - 14))

    def draw_status(self, y: int, x: int, w: int, max_y: int, live: dict):
        ui = self.ui
        if w < 24 or y + 4 > max_y:
            return
        ep = self.s.endpoint()
        sb = self.st.sbx
        env_rows = 5 + (1 if ep and ep.gguf else 0) + (1 if self.st.sbx_list else 0)
        env_h = min(env_rows + 2, max(4, max_y - y))
        if sb == "ready":
            badge_txt, badge_col = "● ready", T.GREEN
        elif sb == "checking…":
            badge_txt, badge_col = f"{SPIN[self.tick % len(SPIN)]} checking", T.CYAN
        else:
            badge_txt, badge_col = "⚠ setup", T.YELLOW
        ui.card(y, x, env_h, w, title="Environment", color=T.RULE, title_color=T.GREEN, badge=badge_txt, badge_color=badge_col)
        ry = y + 1

        def row(label, text, color, bold=False):
            nonlocal ry
            if ry >= y + env_h - 1:
                return
            ui.put(ry, x + 2, f"{label:<9}", ui.color(T.DIM))
            ui.put(ry, x + 12, text, ui.color(color, None, bold), maxw=max(0, w - 14))
            ry += 1

        if ep:
            pr = self.st.ep_ok.get(ep.name)
            up = bool(pr and pr["ok"])
            state = (f"● up ({pr['latency_ms']}ms)" if up else "○ down" if pr else f"{SPIN[self.tick % len(SPIN)]} probing")
            ad_lbl = f" [{ep.effective_adapter}]" if ep.effective_adapter else ""
            row("endpoint", f"{ep.name}{ad_lbl}  {state}", T.GREEN if up else T.YELLOW, bold=up)
            row("", ep.base_url, T.CYAN)
            if ep.gguf:
                pid = servers.server_pid(ep)
                row("llama", f"pid {pid} · {sysmem.fmt(sysmem.rss_bytes(pid))} RSS" if pid else "stopped (1 → s to start)",
                    T.SOFT if pid else T.MUTED)
        else:
            row("endpoint", "none configured — press 1", T.YELLOW)
        sb_line = "● ready" if sb == "ready" else (f"{SPIN[self.tick % len(SPIN)]} checking…" if sb == "checking…" else "⚠ " + sb)
        sb_col = T.GREEN if sb == "ready" else (T.CYAN if sb == "checking…" else T.YELLOW)
        row("sandbox", sb_line, sb_col)
        if self.st.sbx_list:
            row("", f"{len(self.st.sbx_list)} gbtui container(s) alive", T.PINK)
        tot = self.st.mem_total or 1
        av = self.st.mem_avail
        used_frac = max(0.0, min(1.0, 1.0 - av / tot))
        mem_txt = f"{sysmem.fmt(av)} free of {sysmem.fmt(tot)}" + (f" · swap {sysmem.fmt(self.st.swap)}" if self.st.swap else "")
        row("memory", mem_txt, T.SOFT)
        if ry < y + env_h - 1:
            pct_s = f"{int(used_frac * 100):2d}% used"
            hist = self.st.mem_hist[-10:]
            sp_w = (len(hist) + 2) if (len(hist) >= 2 and w >= 52) else 0
            bw = max(6, min(28, w - 16 - len(pct_s) - sp_w))
            ui.heat_bar(ry, x + 12, bw, used_frac)
            px_pct = x + 13 + bw
            ui.put(ry, px_pct, pct_s, ui.color(T.mem_color(used_frac), None, True), maxw=max(0, x + w - 2 - px_pct))
            if sp_w and px_pct + len(pct_s) + 2 + len(hist) < x + w - 1:
                ui.sparkline(ry, px_pct + len(pct_s) + 2, hist)
            ry += 1

        y += env_h + 1
        if (live["active"] or live["phase"] != "idle") and y + 4 <= max_y:
            run_h = min(5, max_y - y)
            col = T.status_color(live["phase"])
            el = (live["finished"] or time.time()) - live["started"] if live["started"] else 0
            ui.card(y, x, run_h, w, title="Active Run" if live["active"] else "Last Run",
                    color=T.PINK if live["active"] else T.RULE, title_color=col, badge=f"⏱ {dur(el)}", badge_color=T.MUTED)
            ry = y + 1
            ph = (f"{SPIN[self.tick % len(SPIN)]} " if live["active"] else "✔ " if live["phase"] == "done" else "✗ ") + live["phase"].upper()
            ui.put(ry, x + 2, ph, ui.color(col, None, True))
            ui.put(ry, x + 16, live["run_id"], ui.color(T.SOFT), maxw=max(0, w - 18))
            ry += 1
            if ry < y + run_h - 1:
                d, t = live["progress"]
                if live["active"] and t:
                    ui.progress(ry, x + 2, max(8, w - 22), d / t, f"{d}/{t}", show_pct=True)
                elif live["active"]:
                    ui.wave_bar(ry, x + 2, max(8, min(28, w - 22)), self.tick, label=(live["current"] or live["phase"])[:18])
                else:
                    hint = "press r for full results · 4 for log" if live["result_dir"] else "press 4 for log"
                    ui.put(ry, x + 2, hint, ui.color(T.MUTED), maxw=max(0, w - 4))
            y += run_h + 1

        runs = self._runs_cache
        if runs and y + 4 <= max_y:
            avail_rows = max(1, min(5, max_y - y - 2))
            rc_h = avail_rows + 2
            ui.card(y, x, rc_h, w, title="Recent Runs", color=T.RULE, title_color=T.CYAN,
                    badge=f"{len(runs)} total", badge_color=T.DIM)
            for i, r in enumerate(runs[:avail_rows]):
                ry = y + 1 + i
                when_s = r.when[5:]
                model_w = 12 if w < 62 else 13
                tier_tag = r.tier[:6].upper()
                ui.put(ry, x + 2, when_s, ui.color(T.DIM))
                ui.put(ry, x + 14, f"{r.model[:model_w - 1]:<{model_w}}", ui.color(T.LIGHT))
                tx = x + 14 + model_w
                ui.soft_pill(ry, tx, f"{tier_tag:<6}", fg=T.tier_color(r.tier), bg=T.BAR_ALT, bold=True)
                hx = tx + 9
                hroom = max(0, x + w - 2 - hx)
                htxt = r.headline
                if len(htxt) > hroom and hroom > 2:
                    htxt = htxt[: hroom - 1].rstrip(" ·") + "…"
                ui.segs(ry, hx, self._headline_segs(htxt, failed=(r.status == "failed")), maxw=hroom)
        elif not runs and y + 4 <= max_y:
            ui.card(y, x, 4, w, title="Quick Start", color=T.RULE, title_color=T.PALE)
            ui.put(y + 1, x + 2, "1. Press 1 to add an endpoint, GGUF folder (F) or discover (D)", ui.color(T.MUTED), maxw=max(0, w - 4))
            ui.put(y + 2, x + 2, "2. Press 2 to run, 3 for Quant Sweep, or 7 for Harness Lens", ui.color(T.MUTED), maxw=max(0, w - 4))

    def quit(self) -> bool:
        """Fast, non-hanging exit: immediately wakes Status poller and stops servers in parallel."""
        if runner.snapshot()["active"]:
            if not self.ui.confirm("A run is in progress. Quit and cancel it? (its sandbox is removed)"):
                return False
            runner.cancel()
            self.ui.busy("cancelling the run and removing its sandbox…", runner.wait, 15.0)
        # Wake the background Status thread immediately so it exits without waiting 5s
        if hasattr(self.st, "stop"):
            self.st.stop()
        else:
            self.st._stop = True
            if hasattr(self.st, "_kick"):
                self.st._kick.set()
        started = servers.started_servers(self.s.endpoints)
        if started and self.s.stop_server_on_exit:
            self.ui.busy(f"stopping {len(started)} llama-server(s)…", servers.stop_all_servers, started)
        return True
