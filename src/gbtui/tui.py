"""gbench-tui: the screens. Home · Endpoints · Run · Live run · Suites · Results · Leaderboard · Export · Sandbox · Settings."""
from __future__ import annotations

import contextlib
import curses
import json
import locale
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__, catalog, config, export, leaderboard, metrics, results as R, runner, sandbox, servers, sysmem
from . import theme as T
from .ui import ESC, PULSE, SIGINT, SPIN, UI, dur

ENTER = ("\n", "\r", curses.KEY_ENTER)
_TS_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})(\s+-\s+.*)$")
_PCT_TOKEN = re.compile(r"(\d+(?:\.\d+)?%)")


def _need_color(need: str) -> int:
    return {
        "docker image": T.YELLOW,
        "Gemini key": T.PINK,
        "logprobs": T.VIOLET,
        "long context": T.CYAN,
        "vision": T.SKY,
        "tool calls": T.CORAL,
    }.get(need, T.YELLOW)


def _quant_color(quant: str) -> int:
    bits = config.quant_bits(quant)
    if bits is None:
        return T.PALE if quant else T.DIM
    if bits <= 3.95:
        return T.CORAL
    if bits <= 5.2:
        return T.YELLOW
    if bits <= 7.0:
        return T.SKY
    if bits <= 9.0:
        return T.VIOLET
    return T.GREEN


@dataclass
class Field:
    key: str
    label: str
    kind: str = "text"            # text | int | float | bool | choice | action | info | picker
    choices: list = field(default_factory=list)
    help: str = ""
    min_val: float | None = None
    max_val: float | None = None


class Status:
    """Host / endpoint / sandbox state, refreshed in a background thread so screens never block on a probe."""

    def __init__(self, app):
        self.app = app
        self.mem_avail = 0
        self.mem_total = sysmem.total_bytes()
        self.swap = 0
        self.mem_hist: list[float] = []
        self.ep_ok: dict = {}
        self.sbx = "checking…"
        self.sbx_list: list = []
        self._stop = False
        self._kick = threading.Event()
        self._probe_all_flag = False
        threading.Thread(target=self._loop, daemon=True).start()

    def kick(self):
        self._kick.set()

    def probe_all(self):
        self._probe_all_flag = True
        self._kick.set()

    def _loop(self):
        n = 0
        while not self._stop:
            kicked = self._kick.is_set()
            self._kick.clear()
            try:
                self.mem_avail = sysmem.available_bytes()
                self.swap = sysmem.swap_used_bytes()
                if self.mem_total > 0 and self.mem_avail > 0:
                    used = max(0.0, min(1.0, 1.0 - self.mem_avail / self.mem_total))
                    self.mem_hist = (self.mem_hist + [used])[-16:]
                eps = list(self.app.s.endpoints)
                if self._probe_all_flag:
                    self._probe_all_flag = False
                    for ep in eps:
                        if self._stop:
                            break
                        self.ep_ok[ep.name] = servers.probe(ep, timeout=1.2)
                else:
                    active_name = self.app.s.active
                    ep = next((e for e in eps if e.name == active_name), eps[0] if eps else None)
                    if ep:
                        self.ep_ok[ep.name] = servers.probe(ep, timeout=1.5)
                if n % 6 == 0 or kicked:
                    self.sbx = sandbox.status() or "ready"
                    self.sbx_list = [s for s in sandbox.sandboxes() if str(s.get("name", "")).startswith("gbtui-")]
            except Exception:                       # noqa: BLE001 — background poller must never die silently
                pass
            n += 1
            self._kick.wait(5)


def _short(bench: str) -> str:
    """A column header that fits: 'decode tok/s (in512)' -> 'tok/s·512', 'TTFT (in512)' -> 'TTFT·512'."""
    m = re.match(r"(.+?) \(in(\d+)\)$", bench)
    name, n = (m.group(1), m.group(2)) if m else (bench, "")
    name = {"decode tok/s": "tok/s", "output tok/s": "out t/s"}.get(name, name)
    return f"{name}·{n}" if n else name


def _friendly_probe_err(ep: config.Endpoint, err: str) -> str:
    if ep.gguf and ("Connection refused" in err or "Errno 61" in err or "Errno 111" in err):
        return "llama-server stopped (press s to start)"
    return err


def _skip_heading(items, sel: int, direction: int = 1) -> int:
    """Move `sel` to the nearest non-heading item (items[i][0] is not None)."""
    if not items:
        return 0
    sel = max(0, min(sel, len(items) - 1))
    if items[sel][0] is not None:
        return sel
    n = len(items)
    for step in range(1, n):
        idx = sel + direction * step
        if 0 <= idx < n and items[idx][0] is not None:
            return idx
    for step in range(1, n):
        idx = sel - direction * step
        if 0 <= idx < n and items[idx][0] is not None:
            return idx
    return sel


def _clamp_top(top: int, sel: int, view: int, total: int, items=None) -> int:
    """Keep viewport scroll `top` tightly clamped on resize, deletion, and top headings."""
    max_top = max(0, total - view)
    top = max(0, min(top, max_top))
    top = min(max(top, sel - view + 1), sel)
    if items and all(items[i][0] is None for i in range(sel)):
        top = 0
    return max(0, top)


class App:
    def __init__(self, scr):
        self.ui = UI(scr)
        self.scr = scr
        self.s = config.load()
        T.set_theme(getattr(self.s, "theme", "aurora"))
        self.st = Status(self)
        self.ui.status_fn = self.footer_status
        self.marked: set = set()
        self.tick = 0
        self._cache_ts = 0.0
        self._runs_cache: list = []
        self._lb_cache: dict = {"models": [], "benches": [], "perf": [], "updated": ""}
        self._cat_count: tuple[int, int] | None = None
        self._last_live_id: str = ""
        self._last_live_headline: str = ""

    def invalidate_cache(self):
        self._cache_ts = 0.0

    def _refresh_summary(self, force: bool = False):
        now = time.monotonic()
        if not force and (now - self._cache_ts) < 6.0:
            return
        self._cache_ts = now
        self._runs_cache = R.list_runs(self.s.results_path)
        self._lb_cache = leaderboard.build(self._runs_cache)
        if self._cat_count is None:
            try:
                cat = catalog.load()
                self._cat_count = (len(cat.get("suites", [])), len(cat.get("golden", [])))
            except Exception:                       # noqa: BLE001
                self._cat_count = (0, 0)

    def _headline_segs(self, headline: str, bg=None, failed: bool = False):
        """Colorize percentage scores inside a run headline using the vivid score heat spectrum."""
        c = self.ui.color
        if failed:
            return [(headline, c(T.RED, bg))]
        segs = []
        for tok in _PCT_TOKEN.split(headline):
            if not tok:
                continue
            if tok.endswith("%"):
                try:
                    val = float(tok[:-1])
                    segs.append((tok, c(T.score_color(val), bg, True)))
                    continue
                except ValueError:
                    pass
            segs.append((tok, c(T.SOFT if bg is None else T.LIGHT, bg)))
        return segs

    # ------------------------------------------------------------------ chrome
    def footer_status(self):
        out = []
        L = runner.snapshot()
        if L["active"]:
            d, t = L["progress"]
            prog = f" {d}/{t}" if t else ""
            out.append((f"{SPIN[self.tick % len(SPIN)]} {L['current'] or L['phase']}{prog}", T.PINK))
        elif L["phase"] in ("done", "failed", "cancelled"):
            out.append(({"done": "✔ run finished", "failed": "✗ run failed", "cancelled": "■ cancelled"}[L["phase"]],
                        T.GREEN if L["phase"] == "done" else T.RED if L["phase"] == "failed" else T.YELLOW))
        if self.st.mem_avail and self.st.mem_total:
            free = self.st.mem_avail / sysmem.GB
            used_f = max(0.0, min(1.0, 1.0 - self.st.mem_avail / self.st.mem_total))
            n_blk = int(round(used_f * 4))
            mini = "█" * n_blk + "░" * (4 - n_blk)
            col = T.GREEN if free > self.s.min_free_gb * 2 else T.YELLOW if free > self.s.min_free_gb else T.RED
            out.append((f"RAM {mini} {free:.0f}G", col))
        return out

    def save(self):
        config.save(self.s)

    def show_help(self):
        txt = (
            "Global shortcuts\n"
            "  1–9, m    Jump directly to any screen from Home (m = Metrics)\n"
            "  ↑↓ / j k  Move selection    PgUp/PgDn  Page scroll\n"
            "  Enter     Open / Edit       Space      Toggle / Activate\n"
            "  t         Cycle color theme on Home (5 palettes)\n"
            "  /         Filter or search  q / Esc    Back / Quit\n\n"
            "Workflow\n"
            "  1 Endpoints → add a /v1 server or .gguf (s starts llama-server)\n"
            "  2 Run       → configure Golden Set, evals or perf & launch\n"
            "  3 Live run  → watch pipeline & output (b leaves it running)\n"
            "  4 Suites    → browse eval suites · m Metrics → formulas & scoring\n"
            "  5 Results   → explore scores, samples & failures (n/p steps)\n"
            "  6 Leaderbd  → rank models by gScore, Enter for full breakdown"
        )
        self.ui.message(txt, title="gbench-tui · shortcuts", color=T.VIOLET)

    # ------------------------------------------------------------------ home
    MENU = [
        ("1", "◎", "Endpoints", "OpenAI-compatible /v1 servers and local GGUF models via llama-server"),
        ("2", "▶", "Run", "Launch Golden Set, academic eval suites, perf or preflight in a Docker Sandbox"),
        ("3", "◉", "Live run", "Follow the live benchmark pipeline, progress and streamed gbench logs"),
        ("4", "◈", "Suites", "Browse all eval suites by pillar and Golden Set cases with guides & needs"),
        ("m", "∑", "Metrics", "Inspect core formulas and the exact scoring metrics used by each benchmark"),
        ("5", "≡", "Results", "Inspect every run: scores, per-sample prompts, reasoning, failures and logs"),
        ("6", "★", "Leaderboard", "Models ranked by gScore with benchmark heatmap and HTML/Markdown export"),
        ("7", "↓", "Export", "Package runs into a Hugging Face dataset (JSONL / Parquet + dataset card)"),
        ("8", "◇", "Sandbox", "Docker Sandboxes template, live microVMs, memory caps and pruning"),
        ("9", "•", "Settings", "Color theme, sandbox RAM/CPU defaults, host memory guard and results folder"),
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
            self.screen_endpoints, self.screen_run, self.screen_live, self.screen_suites, self.screen_metrics,
            self.screen_results, self.screen_leaderboard, self.screen_export, self.screen_sandbox, self.screen_settings,
        ]
        key_idx = {item[0]: idx for idx, item in enumerate(self.MENU)}
        key_idx["0"] = key_idx["m"]
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
                self.ui.notify(f"✦ Theme: {info['label']} — {info['desc']}", T.PINK, 2.8)
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
            if live["active"]:
                d, t = live["progress"]
                return (f"{SPIN[self.tick % len(SPIN)]} {d}/{t}" if t else f"{SPIN[self.tick % len(SPIN)]} {live['phase']}", T.PINK)
            if live["phase"] in ("done", "failed", "cancelled"):
                return (live["phase"], T.status_color(live["phase"]))
            return ("idle", T.DIM)
        if num == "4" and self._cat_count and (self._cat_count[0] or self._cat_count[1]):
            return (f"{self._cat_count[0]}+{self._cat_count[1]}", T.VIOLET)
        if num == "5":
            n = len(self._runs_cache)
            return (f"{n} run{'s' if n != 1 else ''}", T.SKY if n else T.DIM)
        if num == "6":
            top = self._lb_cache["models"][0] if self._lb_cache["models"] else None
            if top and top["gscore"] is not None:
                return (f"★ {top['gscore']:.1f}", T.score_color(top["gscore"]))
            return ("—", T.DIM)
        if num == "8":
            if self.st.sbx == "ready":
                return ("● ready", T.GREEN)
            if self.st.sbx == "checking…":
                return (f"{SPIN[self.tick % len(SPIN)]} check", T.CYAN)
            return ("⚠ setup", T.YELLOW)
        if num == "9":
            return (f"✦ {T.current_theme()}", T.PALE)
        if num == "m":
            return ("core+evals", T.CORAL)
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
            ui.put(y + len(wm), x0 + 1, "terminal UI for Google DeepMind's gbench", ui.color(T.MUTED))
            vx = x0 + 43
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
        ui.card(card_y, 2, menu_h, left_w, title="Workspace", color=T.RULE, title_color=T.VIOLET, badge="1–9 · m", badge_color=T.DIM)
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
            live_icon = SPIN[self.tick % len(SPIN)] if (num == "3" and live["active"]) else icon
            ui.put(ry, 5, num, ui.color(item_col, bg, True))
            ui.put(ry, 7, live_icon, ui.color(item_col if (is_sel or (num == "3" and live["active"])) else T.DIM, bg, is_sel))
            ui.put(ry, 9, ("▸ " if is_sel else "") + name, ui.color(T.LIGHT if is_sel else T.SOFT, bg, is_sel),
                   maxw=max(0, left_w - 11))
            btxt, bcol = self._menu_badge(num, live)
            if btxt and left_w > 24:
                bx = 2 + left_w - len(btxt) - 2
                if bx > 22:
                    ui.put(ry, bx, btxt, ui.color(bcol, bg, is_sel or (num == "3" and live["active"])))
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

        hints = "↑↓ move · Enter open · 1-9/m jump · t theme · ? help"
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
            # 3 sleek side-by-side KPI tiles
            t1_w = max(14, (w - 2) // 3 - 2)
            t2_w = max(18, (w - 2) // 3 + 2)
            t3_w = w - t1_w - t2_w - 2
            # Tile 1: Runs & Models (Violet)
            ui.card(y, x, hh, t1_w, title="Runs", color=T.RULE, title_color=T.VIOLET, icon="≡")
            ui.put(y + 1, x + 2, f"{n_runs}", ui.color(T.LIGHT, None, True))
            ui.put(y + 1, x + 3 + len(str(n_runs)), "total", ui.color(T.DIM), maxw=max(0, t1_w - 5 - len(str(n_runs))))
            ui.put(y + 2, x + 2, f"{n_models} model{'s' if n_models != 1 else ''}", ui.color(T.PALE), maxw=t1_w - 4)
            ok_pct = int(round(ok_runs * 100 / n_runs)) if n_runs else 0
            ui.put(y + 3, x + 2, f"✔ {ok_pct}% ok" if n_runs else "ready", ui.color(T.GREEN if n_runs else T.DIM), maxw=t1_w - 4)
            if hh >= 6:
                ui.bar(y + 4, x + 2, max(4, t1_w - 4), ok_runs / n_runs if n_runs else 0.0, T.GREEN)

            # Tile 2: Top gScore (Gold / Emerald)
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

            # Tile 3: Active Endpoint (Cyan / Mint)
            x3 = x2 + t2_w + 1
            ui.card(y, x3, hh, t3_w, title="Target", color=T.RULE, title_color=T.CYAN, icon="◎")
            if ep:
                pr = self.st.ep_ok.get(ep.name)
                up = bool(pr and pr["ok"])
                dot = PULSE[(self.tick // 3) % len(PULSE)] if up else "○"
                ui.put(y + 1, x3 + 2, f"{dot} {ep.name}", ui.color(T.GREEN if up else T.YELLOW, None, True), maxw=t3_w - 4)
                ui.put(y + 2, x3 + 2, f"● {pr['latency_ms']}ms" if up else ("GGUF · stopped" if ep.gguf else "○ offline"),
                       ui.color(T.GREEN if up else T.MUTED), maxw=t3_w - 4)
                ui.put(y + 3, x3 + 2, f"tier: {self.s.run.tier}", ui.color(T.tier_color(self.s.run.tier)), maxw=t3_w - 4)
                if hh >= 6:
                    sb_ok = self.st.sbx == "ready"
                    ui.put(y + 4, x3 + 2, "sbx ● ready" if sb_ok else "sbx ⚠ check",
                           ui.color(T.GREEN if sb_ok else T.YELLOW), maxw=t3_w - 4)
            else:
                ui.put(y + 2, x3 + 2, "none (press 1)", ui.color(T.YELLOW), maxw=t3_w - 4)
        else:
            # Compact single Snapshot card on medium widths
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
            row("endpoint", f"{ep.name}  {state}", T.GREEN if up else T.YELLOW, bold=up)
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
            row("", f"{len(self.st.sbx_list)} gbtui microVM(s) alive", T.PINK)
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
                    hint = "press r for full results · 3 for log" if live["result_dir"] else "press 3 for log"
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
            ui.put(y + 1, x + 2, "1. Press 1 to configure an endpoint or GGUF", ui.color(T.MUTED), maxw=max(0, w - 4))
            ui.put(y + 2, x + 2, "2. Press 2 to run the Golden Set or evals", ui.color(T.MUTED), maxw=max(0, w - 4))

    def quit(self) -> bool:
        if runner.snapshot()["active"]:
            if not self.ui.confirm("A run is in progress. Quit and cancel it? (its sandbox is removed)"):
                return False
            runner.cancel()
            self.ui.busy("cancelling the run and removing its sandbox…", runner.wait, 30.0)
        started = servers.started_servers(self.s.endpoints)
        if started and self.s.stop_server_on_exit:
            for ep in started:
                servers.stop_server(ep)
        self.st._stop = True
        return True

    # ------------------------------------------------------------------ endpoints
    def screen_endpoints(self):
        ui, sel, top = self.ui, 0, 0
        self.st.probe_all()
        while True:
            eps = self.s.endpoints
            h, w = ui.size()
            self.scr.erase()
            up_n = sum(1 for e in eps if (self.st.ep_ok.get(e.name) or {}).get("ok"))
            ui.header("Endpoints", "OpenAI-compatible /v1 servers reached from the sandbox (--remote-endpoint)",
                      right_badge=(f"● {up_n}/{len(eps)} online", T.GREEN if up_n else T.MUTED))
            sel = max(0, min(sel, len(eps) - 1)) if eps else 0
            detail_h = 9 if eps else 0
            table_y = 3
            table_h = max(4, min(len(eps) + 3, h - detail_h - 6)) if eps else 5
            view = max(1, table_h - 3)
            top = _clamp_top(top, sel, view, len(eps))
            card_r = w - 3

            ui.card(table_y, 2, table_h, w - 4, title="Configured Endpoints", color=T.RULE, title_color=T.CYAN,
                    badge=f"★ {self.s.active}" if self.s.active else "none", badge_color=T.YELLOW)
            hdr_y = table_y + 1
            name_w = 18 if w < 90 else 20
            quant_w = 10 if w >= 88 else 0
            model_w = 15 if w < 100 else 18
            tok_w = 0 if w < 106 else (18 if w < 124 else 22)
            col_name = 8
            col_quant = col_name + name_w
            col_model = col_quant + quant_w
            col_tok = col_model + model_w
            col_url = col_tok + tok_w
            hdr_str = (f"{'name':<{name_w}}"
                       + (f"{'quant':<{quant_w}}" if quant_w else "")
                       + f"{'model':<{model_w}}"
                       + (f"{'tokenizer':<{tok_w}}" if tok_w else "")
                       + "url")
            ui.put(hdr_y, col_name, hdr_str, ui.color(T.DIM), maxw=max(0, card_r - col_name - 1))
            if not eps:
                ui.put(table_y + 2, 5, "No endpoints yet. Press a to add a server or g to pick a local .gguf.",
                       ui.color(T.MUTED), maxw=max(0, card_r - 6))
            for i in range(view):
                j = top + i
                if j >= len(eps):
                    break
                e = eps[j]
                ry = table_y + 2 + i
                active = e.name == self.s.active
                if j == sel:
                    ui.sel_bar(ry, 3, w - 6, T.CYAN)
                bg = T.SEL if j == sel else None
                pr = self.st.ep_ok.get(e.name)
                up = bool(pr and pr["ok"])
                ui.put(ry, 4, "●" if up else "○", ui.color(T.GREEN if up else T.DIM, bg, up))
                ui.put(ry, 6, "★" if active else " ", ui.color(T.YELLOW, bg, True))
                ui.put(ry, col_name, f"{e.name[:name_w - 1]:<{name_w}}",
                       ui.color(T.LIGHT if (active or j == sel) else T.SOFT, bg, active or j == sel),
                       maxw=max(0, card_r - col_name - 1))
                if quant_w:
                    eq = e.effective_quant
                    ui.put(ry, col_quant, f"{(eq or '—')[:quant_w - 1]:<{quant_w}}",
                           ui.color(_quant_color(eq), bg, bool(eq)), maxw=max(0, card_r - col_quant - 1))
                ui.put(ry, col_model, f"{(e.model or 'auto')[:model_w - 1]:<{model_w}}",
                       ui.color(T.PALE, bg), maxw=max(0, card_r - col_model - 1))
                if tok_w:
                    ui.put(ry, col_tok, f"{(e.tokenizer or '—')[:tok_w - 1]:<{tok_w}}",
                           ui.color(T.MUTED, bg), maxw=max(0, card_r - col_tok - 1))
                if e.gguf:
                    pid = servers.server_pid(e)
                    tag, tag_col = ("GGUF ●" if pid else "GGUF ○"), (T.GREEN if pid else T.YELLOW)
                    tag_bold = bool(pid)
                else:
                    tag, tag_col, tag_bold = "HTTP", (T.CYAN if up else T.VIOLET), False
                tag_x = max(col_url + 2, w - len(tag) - 5)
                url_max = max(0, tag_x - col_url - 2)
                if url_max > 0:
                    ui.put(ry, col_url, e.base_url, ui.color(T.CYAN, bg), maxw=url_max)
                if tag_x + len(tag) <= card_r:
                    ui.soft_pill(ry, tag_x - 1, tag, fg=tag_col, bg=T.BAR_ALT, bold=tag_bold)
            ui.scrollbar(table_y + 2, w - 3, view, top, view, len(eps))

            if eps:
                e = eps[sel]
                dy = table_y + table_h + 1
                dh = min(detail_h, h - dy - 2)
                if dh >= 4:
                    pr = self.st.ep_ok.get(e.name)
                    up = bool(pr and pr["ok"])
                    btxt = f"● online · {pr['latency_ms']}ms" if up else "○ offline" if pr else "probing…"
                    ui.card(dy, 2, dh, w - 4, title=e.name, color=T.VIOLET if e.name == self.s.active else T.RULE,
                            title_color=T.PINK if e.name == self.s.active else T.CYAN,
                            badge=btxt, badge_color=T.GREEN if up else T.YELLOW)
                    right_panel = e.gguf and w >= 98 and dh >= 7
                    info_maxw = (w - 48) if right_panel else max(0, w - 19)
                    eq = e.effective_quant
                    q_bits = config.quant_bits(eq)
                    q_desc = (f"{eq}" + (f" (~{q_bits:.1f} bpw)" if q_bits else "") + ("" if e.quant else " · auto")) if eq else "(unspecified)"
                    info = [
                        ("url", e.base_url, T.CYAN),
                        ("model", e.model or "(auto: the only model the endpoint serves)", T.SOFT),
                        ("quant", q_desc, _quant_color(eq) if eq else T.DIM),
                        ("tokenizer", e.tokenizer or "(none — needed for Golden chat-template checks and perf)",
                         T.PALE if e.tokenizer else T.MUTED),
                        ("api key", f"${e.api_key_env}" if e.api_key_env else "none", T.YELLOW if e.api_key_env else T.DIM),
                        ("gguf", f"{Path(e.gguf).name}  · ctx {e.ctx} · {e.parallel} slots"
                         if e.gguf else "external server (vLLM / Ollama / LM Studio / API)", T.SOFT if e.gguf else T.DIM),
                    ]
                    if pr:
                        if up:
                            served = ", ".join(pr["models"][:5]) or "(empty)"
                            mismatch = e.model and pr["models"] and e.model not in pr["models"]
                            p_txt = f"✔ serves {served}" + (f"  ⚠ '{e.model}' not listed" if mismatch else "")
                            info.append(("probe", p_txt, T.YELLOW if mismatch else T.GREEN))
                        else:
                            info.append(("probe", _friendly_probe_err(e, pr["error"]), T.YELLOW if e.gguf else T.RED))
                    for i, (k_, v, col_) in enumerate(info):
                        if dy + 1 + i >= dy + dh - 1:
                            break
                        ui.put(dy + 1 + i, 4, f"{k_:<10}", ui.color(T.DIM))
                        ui.put(dy + 1 + i, 15, v, ui.color(col_), maxw=max(0, info_maxw))
                    if right_panel:
                        rx = w - 31
                        for row_i in range(1, dh - 1):
                            ui.put(dy + row_i, rx - 2, "│", ui.color(T.RULE_DIM))
                        est = servers.estimate_bytes(e)
                        tot = self.st.mem_total or 1
                        ui.put(dy + 1, rx, "RAM Footprint", ui.color(T.PALE, None, True))
                        ui.put(dy + 2, rx, f"~{sysmem.fmt(est)} estimated", ui.color(T.CYAN, None, True))
                        ui.heat_bar(dy + 3, rx, 22, min(1.0, est / tot))
                        ui.put(dy + 4, rx, f"host free: {sysmem.fmt(self.st.mem_avail)}", ui.color(T.MUTED))
                        pid = servers.server_pid(e)
                        ui.put(dy + 5, rx, f"● pid {pid} running" if pid else "○ press s to launch",
                               ui.color(T.GREEN if pid else T.YELLOW))

            ui.footer("Enter edit · Space active · a add · g GGUF · s start/stop · t test · r refresh · d delete · q back")
            self.scr.refresh()
            k = ui.key(500)
            self.tick += 1
            if k is None or k == "RESIZE":
                continue
            if k in ("q", ESC, SIGINT):
                return
            if k in (curses.KEY_UP, "k") and eps:
                sel = (sel - 1) % len(eps)
            elif k in (curses.KEY_DOWN, "j") and eps:
                sel = (sel + 1) % len(eps)
            elif k == curses.KEY_PPAGE and eps:
                sel = max(0, sel - view)
            elif k == curses.KEY_NPAGE and eps:
                sel = min(len(eps) - 1, sel + view)
            elif k == "a":
                before = len(self.s.endpoints)
                self.edit_endpoint(None)
                if len(self.s.endpoints) > before:
                    sel = len(self.s.endpoints) - 1
            elif k == "g":
                before = len(self.s.endpoints)
                self.add_gguf()
                if len(self.s.endpoints) > before:
                    sel = len(self.s.endpoints) - 1
            elif k == "r":
                self.st.probe_all()
                ui.notify("probing all endpoints…", T.CYAN, 1.8)
            elif not eps:
                continue
            elif k in ENTER:
                self.edit_endpoint(eps[sel])
            elif k == " ":
                self.s.active = eps[sel].name
                self.save()
                self.st.kick()
                ui.notify(f"★ active endpoint set to {eps[sel].name}", T.YELLOW)
            elif k == "d":
                if ui.confirm(f"Delete endpoint {eps[sel].name}? (its saved results stay)"):
                    servers.stop_server(eps[sel])
                    name = eps[sel].name
                    del eps[sel]
                    self.st.ep_ok.pop(name, None)
                    sel = max(0, sel - 1)
                    if self.s.active not in [e.name for e in eps]:
                        self.s.active = eps[0].name if eps else ""
                    self.save()
                    ui.notify(f"deleted endpoint {name}", T.YELLOW)
            elif k == "s":
                e = eps[sel]
                if not e.gguf:
                    ui.message("This endpoint has no GGUF: start its server yourself (vLLM, Ollama, LM Studio…).")
                elif servers.server_pid(e):
                    servers.stop_server(e)
                    ui.notify(f"■ llama-server for {e.name} stopped", T.YELLOW)
                else:
                    why = ui.busy(f"starting llama-server for {e.name}…", servers.start_server, e, self.s.min_free_gb)
                    if why:
                        ui.message(why, title="llama-server", color=T.RED)
                    else:
                        ui.notify(f"✔ llama-server ready on {e.base_url}", T.GREEN, 3.0)
                self.st.probe_all()
            elif k == "t":
                self.test_endpoint(eps[sel])

    def test_endpoint(self, e):
        ui = self.ui
        pr = ui.busy(f"GET {e.base_url}/models…", servers.probe, e)
        if not pr["ok"]:
            ui.message(f"✗ {e.base_url} not reachable: {pr['error']}" + ("\n\nStart its llama-server with s." if e.gguf else ""),
                       title="Endpoint test", color=T.RED)
            return
        chat_model = e.model or (pr["models"][0] if pr["models"] else None)
        r = ui.busy("sending one chat completion…", servers.chat_once, e, model=chat_model)
        txt = (f"✔ serves: {', '.join(pr['models'])} ({pr['latency_ms']} ms)\n\n"
               + (f"✔ chat: “{r['text'][:200]}” ({r['seconds']}s)" if r["ok"] else f"✗ chat: {r['text']}"))
        if e.model and pr["models"] and e.model not in pr["models"]:
            txt += f"\n\n⚠ '{e.model}' is not in the served ids: gbench matches by name, or set the model to one of them."
        ui.message(txt, title=f"Endpoint test · {e.name}", color=T.GREEN if r["ok"] else T.YELLOW)

    def add_gguf(self):
        ui = self.ui
        found = sorted({str(p) for d in (Path.home(), Path.home() / "Downloads", Path.home() / "models")
                        if d.is_dir() for p in d.glob("*.gguf")})
        items = []
        for p in found:
            q = config.detect_quant(p)
            sz = f"{os.path.getsize(p) / 1e9:.1f} GB"
            items.append((p, Path(p).name, f"{q} · {sz}" if q else sz))
        items.append(("", "other path…", ""))
        path = ui.choose("Select GGUF file", items)
        if path is None:
            return
        if path == "":
            path = ui.edit("path to the .gguf", str(Path.home()))
            if not path:
                return
        base, quant = config.split_model_quant(Path(path).stem)
        base_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", base.lower()).strip(".") or "gguf-model"
        name = f"{base_slug}-{quant.lower()}" if quant else base_slug
        port = 8080 + len([e for e in self.s.endpoints if e.gguf])
        tok = ui.edit("tokenizer (HF id, for gbench's --tokenizer)", f"LiquidAI/{base}" if base.upper().startswith("LFM") else "",
                      hint="the original model's repo, e.g. LiquidAI/LFM3-1.7B; empty = none")
        ep = config.Endpoint(name=name, base_url=f"http://127.0.0.1:{port}/v1", model=name, tokenizer=tok or "",
                             quant=quant, gguf=path)
        if any(e.name == ep.name for e in self.s.endpoints):
            ep.name += f"-{port}"
        self.s.endpoints.append(ep)
        self.s.active = self.s.active or ep.name
        self.save()
        self.st.probe_all()
        q_msg = f" ({quant})" if quant else ""
        ui.notify(f"✔ added GGUF endpoint {ep.name}{q_msg}", T.GREEN)

    def edit_endpoint(self, e):
        new = e is None
        e = e or config.Endpoint(name=f"endpoint-{len(self.s.endpoints) + 1}")
        fields = [
            Field("name", "Name", help="unique identifier (no / or ..)"),
            Field("base_url", "Base URL (/v1)", help="what gbench gets as --remote-endpoint"),
            Field("model", "Model id (--models)", help="the id the endpoint serves; empty = its only model"),
            Field("quant", "Quantization", help="e.g. Q4_K_M, Q8_0, BF16, FP8, AWQ (auto-detected if empty)"),
            Field("tokenizer", "Tokenizer (--tokenizer)", help="HF id of the original model, e.g. LiquidAI/LFM3-1.7B"),
            Field("api_key_env", "API key env var", help="name of the environment variable with the key (never the key)"),
            Field("gguf", "GGUF file (optional)", help="gbench-tui starts llama-server for it on the URL's port"),
            Field("ctx", "llama-server ctx", "int", help="total context; small = less memory", min_val=512),
            Field("parallel", "llama-server slots", "int", help="concurrent requests (each gets ctx/slots)", min_val=1),
            Field("extra_args", "llama-server extra args"),
            Field("notes", "Notes"),
        ]
        vals = {f.key: getattr(e, f.key) for f in fields}
        old = e.name
        while self.form(("New endpoint" if new else f"Endpoint · {old}"), fields, vals, done_label="Save"):
            new_name = str(vals["name"] or "").strip()
            if not new_name or "/" in new_name or "\\" in new_name or new_name in (".", ".."):
                self.ui.message("Endpoint name must be non-empty and cannot contain '/' or '..'.", color=T.RED)
                continue
            others = {x.name for x in self.s.endpoints if x is not e}
            if new_name in others:
                self.ui.message(f"An endpoint named '{new_name}' already exists.", color=T.RED)
                continue
            vals["name"] = new_name
            vals["quant"] = config.normalize_quant(vals.get("quant", ""))
            if not new and new_name != old and servers.server_pid(e):
                servers.stop_server(e)
            for k, v in vals.items():
                setattr(e, k, v)
            if new:
                self.s.endpoints.append(e)
            elif new_name != old:
                self.st.ep_ok.pop(old, None)
            if self.s.active in ("", old):
                self.s.active = e.name
            self.save()
            self.st.probe_all()
            self.ui.notify(f"✔ saved endpoint {e.name}", T.GREEN)
            return

    # ------------------------------------------------------------------ generic form
    def form(self, title, fields, vals, done_label="Done", sub="", on_change=None, extra=None, extra_h: int = 0) -> bool:
        ui, sel, top = self.ui, 0, 0
        make = fields if callable(fields) else (lambda: fields)
        while True:
            fl = make() + [Field("__done", f"▶ {done_label}", "action")]
            sel = max(0, min(sel, len(fl) - 1))
            if fl[sel].kind == "info":
                for step in range(len(fl)):
                    cand = (sel + step) % len(fl)
                    if fl[cand].kind != "info":
                        sel = cand
                        break
            h, w = ui.size()
            self.scr.erase()
            ui.header(title, sub)
            card_y = 3
            reserve_bottom = 4 + (extra_h + 1 if extra else 0)
            max_card_h = max(3, min(h - card_y - 4, h - card_y - reserve_bottom)) if h > card_y + 6 else max(3, h - card_y - 3)
            card_h = min(len(fl) + 2, max_card_h)
            view = max(1, card_h - 2)
            top = _clamp_top(top, sel, view, len(fl))
            ui.card(card_y, 2, card_h, w - 4, title="Configuration", color=T.RULE, title_color=T.VIOLET)
            lw = min(max(12, w // 2 - 4), max((len(f.label) for f in fl if f.kind != "info"), default=12) + 4)
            for i in range(view):
                j = top + i
                if j >= len(fl):
                    break
                f = fl[j]
                ry = card_y + 1 + i
                if f.kind == "info":
                    if f.label:
                        ui.put(ry, 4, f.label, ui.color(T.PALE, None, True), maxw=max(0, w - 8))
                    else:
                        ui.hline(ry, 4, max(0, w - 8), "─", ui.color(T.RULE_DIM))
                    continue
                is_sel = j == sel
                if is_sel:
                    ui.sel_bar(ry, 3, w - 6, T.GREEN if f.key == "__done" else T.PINK)
                bg = T.SEL if is_sel else None
                if f.kind == "action":
                    if is_sel:
                        ui.pill(ry, 5, f.label, fg=T.INK, bg=T.GREEN if f.key == "__done" else T.CYAN, bold=True)
                    else:
                        ui.put(ry, 5, f.label, ui.color(T.GREEN if f.key == "__done" else T.CYAN, bg, True))
                    continue
                ui.put(ry, 5, f.label, ui.color(T.LIGHT if is_sel else T.SOFT, bg, is_sel), maxw=max(0, lw - 1))
                v = vals.get(f.key)
                vx = 5 + lw
                vmax = max(0, w - vx - 4)
                if f.kind == "bool":
                    txt, col = ("[● ON ]" if v else "[○ OFF]"), (T.GREEN if v else T.DIM)
                    ui.put(ry, vx, txt, ui.color(col, bg, bool(v) or is_sel), maxw=vmax)
                elif f.kind == "choice":
                    lab = next((c[1] for c in f.choices if c[0] == v), str(v))
                    col = T.tier_color(v) if f.key == "tier" else T.CYAN
                    ui.put(ry, vx, f"◀ {lab} ▶", ui.color(col, bg, True), maxw=vmax)
                elif f.kind == "picker":
                    txt = (f"({len(v)}) {', '.join(v)}" if v else "(all)") if isinstance(v, list) else str(v)
                    ui.put(ry, vx, txt, ui.color(T.PALE, bg, is_sel), maxw=vmax)
                else:
                    txt = str(v) if v not in (None, "") else "—"
                    ui.put(ry, vx, txt, ui.color(T.LIGHT if txt != "—" else T.DIM, bg, is_sel), maxw=vmax)
            ui.scrollbar(card_y + 1, w - 3, view, top, view, len(fl))

            if extra:
                ey = card_y + card_h + 1
                eh = max(0, h - 4 - ey)
                if eh >= 4:
                    extra(ey, eh)

            f = fl[sel]
            if f.help and card_y + card_h < h - 3:
                ui.put(h - 3, 2, "▎", ui.color(T.VIOLET, None, True))
                ui.put(h - 3, 4, f.help, ui.color(T.MUTED), maxw=max(0, w - 6))
            ui.footer("↑↓ move · Enter edit/toggle · ←→ change choice · Esc cancel")
            self.scr.refresh()
            k = ui.key(500)
            self.tick += 1
            if k is None or k == "RESIZE":
                continue
            if k in (ESC, SIGINT, "q"):
                return False
            if k in (curses.KEY_UP, "k"):
                sel = (sel - 1) % len(fl)
                while fl[sel].kind == "info":
                    sel = (sel - 1) % len(fl)
                continue
            if k in (curses.KEY_DOWN, "j", "\t"):
                sel = (sel + 1) % len(fl)
                while fl[sel].kind == "info":
                    sel = (sel + 1) % len(fl)
                continue
            if k == curses.KEY_PPAGE:
                sel = max(0, sel - view)
                while fl[sel].kind == "info" and sel < len(fl) - 1:
                    sel += 1
                continue
            if k == curses.KEY_NPAGE:
                sel = min(len(fl) - 1, sel + view)
                while fl[sel].kind == "info" and sel > 0:
                    sel -= 1
                continue
            changed = False
            if f.kind == "choice" and k in (curses.KEY_LEFT, curses.KEY_RIGHT, " "):
                keys = [c[0] for c in f.choices]
                idx = keys.index(vals[f.key]) if vals[f.key] in keys else 0
                vals[f.key] = keys[(idx + (-1 if k == curses.KEY_LEFT else 1)) % len(keys)]
                changed = True
            elif f.kind == "bool" and k in (curses.KEY_LEFT, curses.KEY_RIGHT, " ") + ENTER:
                vals[f.key] = not vals[f.key]
                changed = True
            elif k in ENTER:
                if f.key == "__done":
                    return True
                if f.kind == "action":
                    f.choices[0]()
                elif f.kind == "choice":
                    keys = [c[0] for c in f.choices]
                    v = ui.choose(f.label, [(c[0], c[1], c[2] if len(c) > 2 else "") for c in f.choices],
                                  sel=keys.index(vals[f.key]) if vals[f.key] in keys else 0)
                    if v is not None:
                        vals[f.key] = v
                        changed = True
                elif f.kind == "picker":
                    v = f.choices[0](vals[f.key])
                    if v is not None:
                        vals[f.key] = v
                        changed = True
                else:
                    v = ui.edit(f.label, "" if vals[f.key] is None else str(vals[f.key]), hint=f.help)
                    if v is not None:
                        if f.kind == "int":
                            try:
                                iv = int(v.strip())
                            except ValueError:
                                ui.notify("✗ please enter a whole number", T.RED)
                                continue
                            if f.min_val is not None and iv < f.min_val:
                                ui.notify(f"✗ must be ≥ {int(f.min_val)}", T.RED)
                                continue
                            if f.max_val is not None and iv > f.max_val:
                                ui.notify(f"✗ must be ≤ {int(f.max_val)}", T.RED)
                                continue
                            vals[f.key] = iv
                            changed = True
                        elif f.kind == "float":
                            try:
                                fv = float(v.strip())
                            except ValueError:
                                ui.notify("✗ please enter a number", T.RED)
                                continue
                            if not math.isfinite(fv):
                                ui.notify("✗ number must be finite", T.RED)
                                continue
                            if f.min_val is not None and fv < f.min_val:
                                ui.notify(f"✗ must be ≥ {f.min_val:g}", T.RED)
                                continue
                            if f.max_val is not None and fv > f.max_val:
                                ui.notify(f"✗ must be ≤ {f.max_val:g}", T.RED)
                                continue
                            vals[f.key] = fv
                            changed = True
                        else:
                            vals[f.key] = v.strip()
                            changed = True
            if changed and on_change:
                on_change(vals)

    # ------------------------------------------------------------------ run
    def _draw_cmd_highlighted(self, y: int, x: int, cmd: str, width: int, max_lines: int) -> int:
        ui = self.ui
        lines = UI.wrap(cmd, max(20, width))[:max_lines]
        for i, ln in enumerate(lines):
            segs = []
            for tok in re.split(r"(\s+)", ln):
                if tok == "gbench":
                    segs.append((tok, ui.color(T.GREEN, None, True)))
                elif tok.startswith("--"):
                    segs.append((tok, ui.color(T.VIOLET, None, True)))
                elif tok.startswith("http"):
                    segs.append((tok, ui.color(T.CYAN)))
                elif "<id>" in tok:
                    segs.append((tok, ui.color(T.YELLOW)))
                else:
                    segs.append((tok, ui.color(T.SOFT)))
            ui.segs(y + i, x, segs, maxw=width)
        return len(lines)

    def screen_run(self):
        ui = self.ui
        if runner.snapshot()["active"]:
            return self.screen_live()
        if not self.s.endpoints:
            ui.message("Add an endpoint first (1 · Endpoints).")
            return
        d = self.s.run
        active_ep = self.s.endpoint() or self.s.endpoints[0]
        vals = {
            "endpoint": active_ep.name, "tier": d.tier, "evals": list(d.evals),
            "golden_tasks": list(d.golden_tasks), "eval_limit": d.eval_limit, "max_output_tokens": d.max_output_tokens,
            "thinking": d.thinking, "temperature": d.temperature, "concurrency": d.concurrency,
            "suite_timeout": d.suite_timeout, "perf_mode": d.perf_mode,
            "memory": self.s.sandbox_memory_gb,
        }
        tiers = [(k, k, v) for k, v in runner.TIERS.items()]

        def fields_for(v):
            fl = [
                Field("endpoint", "Endpoint", "choice", [(e.name, e.name, e.base_url) for e in self.s.endpoints],
                      help="the server gbench measures (--remote-endpoint, --models, --tokenizer)"),
                Field("tier", "What to run", "choice", tiers, help=runner.TIERS.get(v["tier"], "")),
            ]
            if v["tier"] == "golden":
                fl.append(Field("golden_tasks", "Golden Set tasks", "picker", [self.pick_golden],
                                help="empty = all 16 (vision / audio cases are N/A for a text-only model)"))
            if v["tier"] in ("evals", "preflight"):
                fl.append(Field("evals", "Eval suites", "picker", [self.pick_suites],
                                help="★ = runs on a laptop with no extra setup; see 4 · Suites for each one's needs"))
            if v["tier"] == "evals":
                fl += [
                    Field("eval_limit", "Samples per suite", "int", help="--eval-limit: stratified subset (0 = all)", min_val=0),
                    Field("concurrency", "Concurrent requests", "int", help="--eval-concurrency (≤ llama-server slots)", min_val=1),
                    Field("suite_timeout", "Suite timeout (s)", "int", help="--suite-timeout: TIMEOUT instead of hanging", min_val=0),
                ]
            if v["tier"] in ("evals", "preflight", "golden"):
                fl += [
                    Field("max_output_tokens", "Max output tokens", "int",
                          help="--max-output-tokens: truncation counts as a failure", min_val=1),
                    Field("thinking", "Thinking / reasoning", "bool", help="--eval-thinking"),
                    Field("temperature", "Temperature", help="empty = gbench's default (0.0, 1.0 with thinking)"),
                ]
            if v["tier"] == "perf":
                fl.append(Field("perf_mode", "Mode", "choice", [
                    ("personal", "personal", "single stream: decode tok/s, TTFT"),
                    ("quick", "quick", "serving preset quick, text only"),
                ]))
            fl += [Field("memory", "Sandbox RAM (GB)", "int", help="microVM cap; gbench + datasets fit in 4–6 GB", min_val=2),
                   Field("sep", "", "info")]
            return fl

        def preview(y: int, max_h: int):
            ep = self.s.endpoint(vals["endpoint"]) or self.s.endpoints[0]
            spec = self.spec_from(vals, ep)
            h, w = ui.size()
            ch = min(max_h, 9)
            tcol = T.tier_color(vals["tier"])
            ui.card(y, 2, ch, w - 4, title="Sandbox Command & Preflight", color=T.RULE, title_color=tcol,
                    badge=f"{vals['tier'].upper()} · {vals['memory']}GB", badge_color=tcol)
            used = self._draw_cmd_highlighted(y + 1, 4, runner.command_preview(spec), max(10, w - 8), max(1, min(3, ch - 4)))
            cy = y + 1 + used + 1
            up = (self.st.ep_ok.get(ep.name) or {}).get("ok")
            need = max(2, int(vals["memory"])) * sysmem.GB + (0 if not ep.gguf or up or servers.server_pid(ep) else servers.estimate_bytes(ep))
            why = sysmem.check(need, self.s.min_free_gb)
            if cy < y + ch - 1:
                cx = 4
                cx += ui.soft_pill(cy, cx, f"{'✔' if not why else '✗'} RAM {sysmem.fmt(need)}",
                                   fg=T.GREEN if not why else T.RED, bg=T.BAR_ALT) + 1
                if up:
                    cx += ui.soft_pill(cy, cx, f"● {ep.name} online", fg=T.GREEN, bg=T.BAR_ALT) + 1
                elif ep.gguf:
                    cx += ui.soft_pill(cy, cx, "▸ llama auto-start", fg=T.CYAN, bg=T.BAR_ALT) + 1
                else:
                    cx += ui.soft_pill(cy, cx, "⚠ endpoint offline", fg=T.YELLOW, bg=T.BAR_ALT) + 1
                sb_ok = self.st.sbx == "ready"
                if cx + 16 < w - 4:
                    ui.soft_pill(cy, cx, "✔ sandbox ready" if sb_ok else "⚠ sandbox setup",
                                 fg=T.GREEN if sb_ok else T.YELLOW, bg=T.BAR_ALT, maxw=max(0, w - cx - 4))
                cy += 1
            if cy < y + ch - 1:
                tot = self.st.mem_total or 1
                proj = max(0.0, min(1.0, (tot - self.st.mem_avail + need) / tot))
                bw = max(8, min(28, w - 42))
                ui.heat_bar(cy, 4, bw, proj)
                info_m = f"projected {int(proj * 100)}% RAM ({sysmem.fmt(self.st.mem_avail)} free)" if not why else why
                ui.put(cy, 6 + bw, info_m, ui.color(T.MUTED if not why else T.RED), maxw=max(0, w - 10 - bw))
                cy += 1
            if cy < y + ch - 1 and vals["tier"] in ("evals", "preflight"):
                heavy = [f"{n} ({', '.join(catalog.suite(n)['needs'])})" for n in vals["evals"]
                         if catalog.suite(n) and catalog.suite(n)["needs"]]
                if heavy:
                    ui.put(cy, 4, "⚠ extra needs: " + "; ".join(heavy) + " — try Preflight first",
                           ui.color(T.YELLOW), maxw=max(0, w - 8))

        if not self.form("Run gbench", lambda: fields_for(vals), vals, done_label="Start run", extra=preview, extra_h=9,
                         sub="every run gets a fresh Docker Sandbox microVM, removed automatically when it ends"):
            return
        ep = self.s.endpoint(vals["endpoint"]) or self.s.endpoints[0]
        self.s.active = ep.name
        self.s.sandbox_memory_gb = max(2, int(vals["memory"]))
        for k in ("tier", "evals", "golden_tasks", "eval_limit", "max_output_tokens", "thinking", "temperature",
                  "concurrency", "suite_timeout", "perf_mode"):
            setattr(self.s.run, k, vals[k])
        self.save()
        why = runner.start(self.spec_from(vals, ep), self.s)
        if why:
            ui.message(why, title="Cannot start run", color=T.RED)
            return
        self.screen_live()

    @staticmethod
    def spec_from(v, ep):
        return runner.RunSpec(
            endpoint=ep, tier=v["tier"], evals=list(v["evals"]), golden_tasks=list(v["golden_tasks"]),
            eval_limit=max(0, int(v["eval_limit"] or 0)), max_output_tokens=max(1, int(v["max_output_tokens"] or 4096)),
            thinking=bool(v["thinking"]), temperature=str(v["temperature"] or ""),
            concurrency=max(1, int(v["concurrency"] or 1)), suite_timeout=max(0, int(v["suite_timeout"] or 0)),
            perf_mode=v["perf_mode"],
        )

    def pick_golden(self, current):
        gold = catalog.load()["golden"]
        items = [(g["id"], g.get("id", ""), f"{g.get('category', '')} · {g.get('description', '')}") for g in gold]
        return self.multi_pick("Golden Set tasks (none = all)", items, set(current))

    def pick_suites(self, current):
        cat = catalog.load()
        items = []
        for p in cat["pillars"] + ["Other"]:
            group = [s for s in cat["suites"] if s["pillar"] == p]
            if not group:
                continue
            items.append((None, p, ""))
            for s in sorted(group, key=lambda s: (not s["light"], s["name"])):
                badge = ("★ " if s["light"] else "  ")
                items.append((s["name"], badge + s["name"], " · ".join(s["needs"]) or s["title"]))
        return self.multi_pick("Eval suites (Space toggles · ★ laptop-friendly)", items, set(current))

    def multi_pick(self, title, items, chosen: set):
        """items: [(value|None for a heading, label, note)] → list of chosen values, or None on Esc."""
        ui, sel, top, flt = self.ui, 0, 0, ""
        while True:
            vis = [it for it in items if it[0] is None or not flt or flt in it[1].lower() or flt in it[2].lower()]
            if flt:
                cleaned = []
                for idx, it in enumerate(vis):
                    if it[0] is None:
                        if idx + 1 < len(vis) and vis[idx + 1][0] is not None:
                            cleaned.append(it)
                    else:
                        cleaned.append(it)
                vis = cleaned
            h, w = ui.size()
            self.scr.erase()
            ui.header(title, f"{len(chosen)} selected" + (f" · filter: '{flt}'" if flt else ""),
                      right_badge=(f"{sum(1 for it in vis if it[0] is not None)} shown", T.CYAN))
            view = max(1, h - 5)
            sel = _skip_heading(vis, sel, 1) if vis else 0
            top = _clamp_top(top, sel, view, len(vis), items=vis)
            if not vis:
                ui.put(5, 4, "No items match that filter. Press / to edit or clear it.", ui.color(T.MUTED), maxw=max(0, w - 8))
            p_idx = 0
            for i in range(view):
                j = top + i
                if j >= len(vis):
                    break
                v, lab, note = vis[j]
                y = 3 + i
                if v is None:
                    ui.section(y, 3, lab, w - 7, T.pillar_color(p_idx))
                    p_idx += 1
                    continue
                on = v in chosen
                if j == sel:
                    ui.sel_bar(y, 2, w - 5, T.GREEN if on else T.PINK)
                bg = T.SEL if j == sel else None
                ui.put(y, 4, "◉" if on else "○", ui.color(T.GREEN if on else T.DIM, bg, on))
                ui.put(y, 6, lab[:28], ui.color(T.LIGHT if (on or j == sel) else T.SOFT, bg, on or j == sel))
                ui.put(y, 35, note, ui.color(T.MUTED, bg), maxw=max(0, w - 38))
            ui.scrollbar(3, w - 2, view, top, view, len(vis))
            ui.footer("Space toggle · Enter done · / filter · a all/none · i info · q/Esc cancel")
            self.scr.refresh()
            k = ui.wait_key()
            if k is None or k == "RESIZE":
                continue
            if k in (ESC, SIGINT, "q"):
                return None
            if k in ENTER:
                return [it[0] for it in items if it[0] in chosen]
            if k == "/":
                f = ui.edit("Filter", flt, hint="empty clears the filter")
                if f is not None:
                    flt = f.lower().strip()
                    sel = top = 0
                continue
            if not vis:
                continue
            if k in (curses.KEY_UP, "k"):
                sel = _skip_heading(vis, sel - 1, -1)
            elif k in (curses.KEY_DOWN, "j"):
                sel = _skip_heading(vis, sel + 1, 1)
            elif k == curses.KEY_NPAGE:
                sel = _skip_heading(vis, min(len(vis) - 1, sel + view), 1)
            elif k == curses.KEY_PPAGE:
                sel = _skip_heading(vis, max(0, sel - view), -1)
            elif k in ("g", curses.KEY_HOME):
                sel = _skip_heading(vis, 0, 1)
            elif k in ("G", curses.KEY_END):
                sel = _skip_heading(vis, len(vis) - 1, -1)
            elif k == " " and vis[sel][0] is not None:
                chosen ^= {vis[sel][0]}
                sel = _skip_heading(vis, min(len(vis) - 1, sel + 1), 1)
            elif k == "a":
                vals_vis = {it[0] for it in vis if it[0] is not None}
                chosen = (chosen - vals_vis) if vals_vis <= chosen else (chosen | vals_vis)
            elif k == "i" and vis[sel][0] is not None:
                self.show_doc(vis[sel][0])

    # ------------------------------------------------------------------ live
    def _draw_pipeline(self, y: int, x: int, w: int, phase: str):
        ui = self.ui
        stages = [("server", "Server"), ("sandbox", "Sandbox"), ("running", "Running"), ("collecting", "Collect"), ("done", "Done")]
        order = {k: i for i, (k, _) in enumerate(stages)}
        cur_idx = order.get(phase, 4 if phase in ("failed", "cancelled") else 0)
        cx = x
        for i, (_key, label) in enumerate(stages):
            if cx >= x + w - 10:
                break
            if i:
                conn_col = T.GREEN if i <= cur_idx and phase not in ("failed", "cancelled") else T.RULE_DIM
                cx += ui.put(y, cx, " ── ", ui.color(conn_col))
            if phase in ("failed", "cancelled") and i == 4:
                icon = "✗" if phase == "failed" else "■"
                col = T.RED if phase == "failed" else T.YELLOW
                lab = "Failed" if phase == "failed" else "Cancelled"
                cx += ui.soft_pill(y, cx, f"{icon} {lab}", fg=col, bg=T.BAR_ALT, bold=True)
            elif i < cur_idx or (phase == "done" and i == 4):
                cx += ui.soft_pill(y, cx, f"✔ {label}", fg=T.GREEN, bg=T.BAR_ALT, bold=(i == 4))
            elif i == cur_idx:
                cx += ui.pill(y, cx, f"{SPIN[self.tick % len(SPIN)]} {label}", fg=T.INK, bg=T.PINK, bold=True)
            else:
                cx += ui.put(y, cx, f"○ {label}", ui.color(T.DIM))

    def _log_segs(self, ln: str):
        c = self.ui.color
        if ln.startswith("✗") or "ERROR" in ln or "Traceback" in ln or ln.lstrip().startswith("FAIL "):
            return [(ln, c(T.RED))]
        if ln.startswith("✔"):
            return [(ln, c(T.GREEN, None, True))]
        if ln.startswith("$ "):
            segs = [("$ ", c(T.PINK, None, True))]
            for tok in re.split(r"(\s+)", ln[2:]):
                if tok == "gbench":
                    segs.append((tok, c(T.GREEN, None, True)))
                elif tok.startswith("--"):
                    segs.append((tok, c(T.VIOLET, None, True)))
                elif tok.startswith("http"):
                    segs.append((tok, c(T.CYAN)))
                else:
                    segs.append((tok, c(T.SOFT)))
            return segs
        if ln.startswith("▸"):
            return [(ln, c(T.CYAN, None, True))]
        if "WARNING" in ln or ln.startswith("⚠"):
            return [(ln, c(T.YELLOW))]
        if set(ln.strip()) <= {"=", "-"} and len(ln.strip()) >= 10:
            return [(ln, c(T.RULE))]
        if "PASS" in ln and ln.lstrip().startswith("- "):
            idx = ln.rfind("PASS")
            return [(ln[:idx], c(T.SOFT)), ("PASS", c(T.GREEN, None, True)), (ln[idx + 4:], c(T.SOFT))]
        if "FAIL" in ln and ln.lstrip().startswith("- "):
            idx = ln.rfind("FAIL")
            return [(ln[:idx], c(T.SOFT)), ("FAIL", c(T.RED, None, True)), (ln[idx + 4:], c(T.SOFT))]
        if "N/A" in ln and ln.lstrip().startswith("- "):
            return [(ln, c(T.DIM))]
        m = _TS_LINE.match(ln)
        if m:
            rest = m.group(2)
            if " - INFO - " in rest:
                a, b = rest.split(" - INFO - ", 1)
                return [(m.group(1), c(T.DIM)), (a + " - ", c(T.DIM)), ("INFO", c(T.CYAN)), (" - " + b, c(T.SOFT))]
            return [(m.group(1), c(T.DIM)), (rest, c(T.SOFT))]
        return [(ln, c(T.SOFT))]

    def screen_live(self):
        ui = self.ui
        if runner.snapshot()["phase"] == "idle":
            ui.message("No run in progress. Start one in 2 · Run.")
            return
        follow = True
        wrap_lines = False
        top = 0
        left = 0
        prev_dropped = runner.snapshot()["dropped"]
        while True:
            L = runner.snapshot()
            h, w = ui.size()
            self.scr.erase()
            spec = L["spec"]
            sub = f"{spec.endpoint.name} · {spec.tier}" if spec else ""
            el = (L["finished"] or time.time()) - L["started"] if L["started"] else 0
            ui.header("Live run", sub, right_badge=(f"⏱ {dur(el)}", T.PINK if L["active"] else T.MUTED))

            top_h = 6
            pcol = T.VIOLET if L["active"] else T.status_color(L["phase"])
            ui.card(3, 2, top_h, w - 4, title=L["run_id"] or "Benchmark", color=pcol, title_color=T.PINK if L["active"] else pcol,
                    badge=L["phase"].upper(), badge_color=T.status_color(L["phase"]))
            self._draw_pipeline(4, 4, w - 8, L["phase"])
            sb_info = (f"sandbox {L['sandbox']}" + (" (removed)" if not L["active"] else "")) if L["sandbox"] else "preparing…"
            ui.put(5, 4, sb_info, ui.color(T.MUTED), maxw=max(0, w - 10))

            d, t = L["progress"]
            if L["active"]:
                if t:
                    rate_s = ""
                    if d > 0 and d <= t and el > 2:
                        rate = d / el
                        rem = (el / d) * (t - d)
                        rate_s = f" · {rate:.2f}/s · ETA {dur(rem)}"
                    lbl = f"{d}/{t} {L['current']}{rate_s}"
                    ui.progress(6, 4, max(8, w - len(lbl) - 16), d / t, lbl[: max(10, w - 24)], show_pct=True)
                else:
                    lbl = L["current"] or {"server": "starting llama-server…", "sandbox": "booting microVM…",
                                           "running": "running gbench…", "collecting": "saving results…"}.get(L["phase"], "")
                    ui.wave_bar(6, 4, max(8, min(44, w - len(lbl) - 14)), self.tick, label=lbl[: max(10, w - 24)])
            else:
                if L["run_id"] != self._last_live_id:
                    self._last_live_id = L["run_id"]
                    self._last_live_headline = ""
                    if L["result_dir"]:
                        run_obj = R.load_run(Path(L["result_dir"]))
                        self._last_live_headline = run_obj.headline if run_obj else ""
                verdict = self._last_live_headline or L["error"] or f"finished ({L['phase']})"
                ui.put(6, 4, ("✔ " if L["phase"] == "done" else "✗ " if L["phase"] == "failed" else "■ ") + verdict,
                       ui.color(T.status_color(L["phase"]), None, True), maxw=max(0, w - 28))
                if L["result_dir"]:
                    ui.pill(6, max(6, w - 23), "r ▸ inspect", fg=T.INK, bg=T.GREEN, bold=True)

            oy = 3 + top_h + 1
            oh = max(4, h - oy - 2)
            raw_lines = L["lines"]
            if not follow and L["dropped"] > prev_dropped:
                top = max(0, top - (L["dropped"] - prev_dropped))
            prev_dropped = L["dropped"]
            if wrap_lines:
                lines = [part for ln in raw_lines for part in UI.wrap(ln, max(20, w - 8))]
            else:
                lines = raw_lines
            view = max(1, oh - 2)
            max_top = max(0, len(lines) - view)
            if follow:
                top = max_top
            else:
                top = max(0, min(top, max_top))
            max_lw = max((len(ln) for ln in lines[top: top + view]), default=0)
            left = max(0, min(left, max(0, max_lw - max(10, w - 10))))
            obadge = (f"● FOLLOW · {len(raw_lines)} lines" if follow else f"PAUSED {top + 1}/{max(1, len(lines))}")
            ui.card(oy, 2, oh, w - 4, title="gbench output", color=T.RULE, title_color=T.CYAN,
                    badge=obadge, badge_color=T.GREEN if follow else T.YELLOW)
            for i in range(view):
                j = top + i
                if j >= len(lines):
                    break
                ui.segs(oy + 1 + i, 4, self._log_segs(lines[j]), maxw=max(0, w - 8), skip=0 if wrap_lines else left)
            ui.scrollbar(oy + 1, w - 3, view, top, view, len(lines))

            if L["active"]:
                hint = "c cancel · b back (keeps running) · ↑↓ scroll · ←→ pan · w wrap · f follow"
            else:
                hint = ("r results · " if L["result_dir"] else "") + "b back · ↑↓ scroll · ←→ pan · w wrap"
            ui.footer(hint)
            self.scr.refresh()
            k = ui.key(150 if L["active"] else 500)
            self.tick += 1
            if k is None or k == "RESIZE":
                continue
            if k in ("b", "q", ESC, SIGINT):
                return
            if k == "c" and L["active"]:
                if ui.confirm("Cancel the run? gbench stops and its sandbox is removed."):
                    runner.cancel()
                    ui.notify("■ cancelling run…", T.YELLOW)
            elif k in (curses.KEY_UP, "k"):
                follow, top = False, max(0, top - 1)
            elif k in (curses.KEY_DOWN, "j"):
                top = min(max_top, top + 1)
                if top >= max_top:
                    follow = True
            elif k == curses.KEY_PPAGE:
                follow, top = False, max(0, top - view)
            elif k == curses.KEY_NPAGE:
                top = min(max_top, top + view)
                if top >= max_top:
                    follow = True
            elif k in ("g", curses.KEY_HOME):
                follow, top = False, 0
            elif k == curses.KEY_LEFT and not wrap_lines:
                left = max(0, left - 8)
            elif k == curses.KEY_RIGHT and not wrap_lines:
                left += 8
            elif k == "w":
                wrap_lines = not wrap_lines
                left = 0
                ui.notify("line wrap " + ("ON" if wrap_lines else "OFF"), T.CYAN, 1.5)
            elif k in ("f", "G", curses.KEY_END):
                follow = True
            elif k == "r" and not L["active"] and L["result_dir"]:
                run = R.load_run(Path(L["result_dir"]))
                if run:
                    self.run_detail(run)

    # ------------------------------------------------------------------ suites
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

    # ------------------------------------------------------------------ results
    def screen_results(self):
        ui, sel, top, flt = self.ui, 0, 0, ""
        all_runs = ui.busy("reading results…", R.list_runs, self.s.results_path)
        while True:
            runs = [r for r in all_runs if not flt or flt in (f"{r.model} {r.base_model} {r.quant} {r.tier} {r.status} {r.headline} {r.id}").lower()]
            h, w = ui.size()
            self.scr.erase()
            models_n = len({(r.base_model.lower(), r.quant) for r in all_runs})
            ok_n = sum(1 for r in all_runs if r.status == "ok")
            sub = f"{len(all_runs)} runs · {models_n} variant(s) · {ok_n} ok" + (f" · filter: '{flt}'" if flt else "")
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
                ui.put(yy, 4, "◉" if r.id in self.marked else " ", ui.color(T.PINK, bg, True))
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
                ui.card(py, 2, preview_h, w - 4, title=r.id, color=T.RULE, title_color=T.tier_color(r.tier),
                        badge=r.when, badge_color=T.DIM)
                ui.segs(py + 1, 4, self._headline_segs(r.headline, failed=(r.status == "failed")), maxw=max(0, w - 8))
                ui.put(py + 2, 4, str(r.path), ui.color(T.DIM), maxw=max(0, w - 8))

            ui.footer("Enter open · m mark · x export · l log · / filter · D delete · q back")
            self.scr.refresh()
            k = ui.wait_key()
            if k is None or k == "RESIZE":
                continue
            if k in ("q", ESC, SIGINT):
                return
            if k == "/":
                f = ui.edit("Filter runs", flt, hint="matches model, tier, status or score; empty clears")
                if f is not None:
                    flt = f.lower().strip()
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
        while True:
            h, w = ui.size()
            self.scr.erase()
            ui.header(f"Run · {run.id}", f"{run.model_label} · {run.tier} · {run.when}" + (f" · {dur(run.duration)}" if run.duration else ""),
                      crumbs=["Results", run.id],
                      right_badge=(f"● {run.status.upper()}", T.status_color(run.status)))
            spec = run.meta.get("spec") or {}
            q_info = f"  · quant {run.quant}" if run.quant else ""
            info = [
                ("endpoint", f"{run.endpoint.get('base_url', '?')}  · model {run.endpoint.get('model', '?')}{q_info}  · tokenizer {run.endpoint.get('tokenizer') or '—'}"),
                ("settings", f"limit {spec.get('eval_limit')} · max tokens {spec.get('max_output_tokens')} · thinking {'on' if spec.get('thinking') else 'off'} · temp {spec.get('temperature') or 'default'}"),
                ("sandbox", f"{(run.meta.get('sandbox') or {}).get('template', '?')} · {(run.meta.get('sandbox') or {}).get('memory_gb', '?')} GB · gbench {run.meta.get('gbench_commit', '?')}"),
            ]
            if run.meta.get("error"):
                info.append(("error", run.meta["error"]))
            meta_h = len(info) + 2
            card_r = w - 3
            ui.card(3, 2, meta_h, w - 4, title="Run Metadata", color=T.RULE, title_color=T.VIOLET,
                    badge=run.tier.upper(), badge_color=T.tier_color(run.tier))
            for idx, (k_, v) in enumerate(info):
                ui.put(4 + idx, 4, f"{k_:<10}", ui.color(T.DIM))
                ui.put(4 + idx, 15, v, ui.color(T.RED if k_ == "error" else T.SOFT), maxw=max(0, card_r - 16))

            sc = run.scores
            sy = 3 + meta_h + 1
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

    def samples(self, run, score, failures: bool):
        ui = self.ui
        all_its = R.items(run, score)
        if not all_its:
            ui.message("No per-sample records for this score.")
            return
        mode = "fail" if failures else "all"   # all | fail | pass
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
        lines = [[("verdict ", c(T.DIM)), (f" {verdict} ", c(T.INK, vcol, True, auto_sel=False)),
                  (f"  category: {it.get('category') or '—'}", c(T.PALE))]]
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
        """View a sample in full, with n/p (or [/]) stepping directly to adjacent samples."""
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

    # ------------------------------------------------------------------ leaderboard
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
            lines.append([("◆ ", c(T.GREEN, None, True)), ("Performance Metrics", c(T.LIGHT, None, True))])
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
        flt = ""
        left = 0
        sel, top = 0, 0
        lb = ui.busy("building the leaderboard…", lambda: leaderboard.build(R.list_runs(self.s.results_path), comparable))
        while True:
            h, w = ui.size()
            self.scr.erase()
            all_models = lb["models"]
            models = [m for m in all_models if not flt or flt in f"{m['model']} {m.get('base_model', '')} {m.get('quant', '')}".lower()]
            benches = lb["benches"] + lb["perf"]
            leader_g = models[0]["gscore"] if (models and models[0]["gscore"] is not None) else None
            q_cnt = len(lb.get("quants") or [])
            sub = f"{len(models)} variant(s)" + (f" · {q_cnt} quant(s)" if q_cnt else "") + f" · updated {lb['updated']}"
            if flt:
                sub += f" · /{flt}"
            ui.header("Leaderboard", sub,
                      right_badge=("★ COMPARABLE ONLY" if comparable else "ALL RUNS", T.YELLOW if comparable else T.PINK))

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
                    ui.put(yy, x0, " ★ 1 ", ui.color(T.INK, T.YELLOW, True, auto_sel=False))
                elif idx == 1:
                    ui.put(yy, x0, " ◈ 2 ", ui.color(T.INK, T.SOFT, True, auto_sel=False))
                elif idx == 2:
                    ui.put(yy, x0, " ◇ 3 ", ui.color(T.INK, T.CORAL, True, auto_sel=False))
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
                    else:
                        ui.put(yy, cx, f"{s.value:.1f}{'' if s.unit == 'tok/s' else s.unit}", ui.color(T.CYAN, bg, True))
                if yy + 1 < table_y + table_h - 1:
                    q_note = f" · {r['quant']}" if (r.get("quant") and not has_q) else ""
                    ui.put(yy + 1, x0 + 6, f"{r['when']}{q_note} · {r['runs']} run(s) · {r['n_bench']} benchmarks",
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
                    ui.put(py + 2, 4, "press Enter for full benchmark breakdown", ui.color(T.DIM), maxw=max(0, w - 8))

            p = leaderboard.paths(self.s)
            ui.put(h - 3, 3, f"files: {p['md']} · .html · .json", ui.color(T.DIM), maxw=max(0, w - 6))
            ui.footer("Enter breakdown · ←→ cols · / filter · c comparable · m metrics · o HTML · w write · q back")
            self.scr.refresh()
            k = ui.wait_key()
            if k is None or k == "RESIZE":
                continue
            if k in ("q", ESC, SIGINT):
                return
            if k == "m":
                self.screen_metrics()
                continue
            if k == "/":
                f = ui.edit("Filter leaderboard", flt, hint="matches model or quantization (e.g. Q4_K_M); empty clears")
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

    # ------------------------------------------------------------------ export
    def screen_export(self):
        runs = self.ui.busy("reading results…", R.list_runs, self.s.results_path)
        if not runs:
            self.ui.message("No runs to export yet.")
            return
        chosen = [r for r in runs if r.id in self.marked]
        self.do_export(chosen or runs)

    def do_export(self, runs):
        ui = self.ui
        out_default = str(Path.home() / "gbench-datasets" / f"gbench-{time.strftime('%Y%m%d-%H%M')}")
        vals = {
            "out": out_default, "fmt": "both", "failures": False,
            "runs": f"{len(runs)} run(s): " + ", ".join(r.id for r in runs[:3]) + ("…" if len(runs) > 3 else ""),
        }
        fields = [
            Field("runs", vals["runs"], "info"),
            Field("out", "Output folder", help="a dataset folder: README.md + data/"),
            Field("fmt", "Format", "choice", [
                ("jsonl", "JSONL", ""),
                ("parquet", "Parquet", "pyarrow, else in a sandbox"),
                ("both", "JSONL + Parquet", ""),
            ]),
            Field("failures", "Failures only", "bool", help="only the samples the model got wrong"),
        ]
        while True:
            if not self.form("Export as a dataset", fields, vals, done_label="Export",
                             sub="samples (prompt, response, gold, correct) + scores, with a Hugging Face dataset card"):
                return
            if not str(vals["out"] or "").strip():
                ui.message("Output folder cannot be empty.", color=T.RED)
                continue
            break
        log: list = []
        out = Path(vals["out"]).expanduser()
        try:
            ui.busy("exporting dataset…", export.export, runs, out, vals["fmt"], vals["failures"], log.append)
        except Exception as e:                       # noqa: BLE001
            log.append(f"✗ {e}")
        log += ["", f"Upload with:  huggingface-cli upload <you>/<name> {out} --repo-type dataset"]
        ui.pager("Export", [self._log_segs(ln) for ln in log], sub=str(out))

    # ------------------------------------------------------------------ sandbox
    def screen_sandbox(self):
        ui = self.ui
        log: list = []
        while True:
            h, w = ui.size()
            self.scr.erase()
            st = self.st.sbx
            btxt = "● READY" if st == "ready" else ("CHECKING" if st == "checking…" else "⚠ SETUP NEEDED")
            bcol = T.GREEN if st == "ready" else (T.CYAN if st == "checking…" else T.YELLOW)
            ui.header("Sandbox", "gbench always runs inside Docker Sandboxes (sbx): microVM with RAM cap and network policy",
                      right_badge=(btxt, bcol))
            rows = [
                ("sbx", shutil.which("sbx") or "not installed", T.SOFT if shutil.which("sbx") else T.RED),
                ("status", st, T.GREEN if st == "ready" else (T.CYAN if st == "checking…" else T.YELLOW)),
                ("template", sandbox.TEMPLATE + ("  ✔ built" if st == "ready" else ""), T.GREEN if st == "ready" else T.SOFT),
                ("gbench", f"{sandbox.GBENCH_REPO} @ {sandbox.gbench_commit() or sandbox.GBENCH_REF}", T.PALE),
                ("run cap", f"{self.s.sandbox_memory_gb} GB RAM · {self.s.cpus()} CPUs per run sandbox (one at a time)", T.CYAN),
                ("network", "deny-all + endpoint + Hugging Face / GitHub raw (datasets)", T.SOFT),
                ("host ↔", f"localhost endpoints reached as {sandbox.host_alias()}", T.MUTED),
                ("cache", str(sandbox.work_dir()) + "  (HF datasets, in-flight runs)", T.DIM),
            ]
            env_h = len(rows) + 2
            ui.card(3, 2, env_h, w - 4, title="Sandbox Configuration", color=T.RULE, title_color=T.VIOLET)
            for idx, (k_, v, col_) in enumerate(rows):
                ui.put(4 + idx, 4, f"{k_:<10}", ui.color(T.DIM))
                ui.put(4 + idx, 15, v, ui.color(col_), maxw=max(0, w - 19))

            ly = 3 + env_h + 1
            live_list = self.st.sbx_list
            live_h = min(max(3, len(live_list) + 2), max(3, h - ly - (7 if log else 3)))
            if ly + live_h <= h - 2:
                ui.card(ly, 2, live_h, w - 4, title="Live gbtui Sandboxes", color=T.RULE, title_color=T.CYAN,
                        badge=f"{len(live_list)} alive", badge_color=T.GREEN if live_list else T.DIM)
                if not live_list:
                    ui.put(ly + 1, 4, "(none — sandboxes are created on demand and removed after each run)",
                           ui.color(T.DIM), maxw=max(0, w - 8))
                else:
                    for idx, s in enumerate(live_list[: live_h - 2]):
                        ui.put(ly + 1 + idx, 4, f"● {s.get('name')}  {s.get('status', '')}  {s.get('memory', '')}",
                               ui.color(T.GREEN), maxw=max(0, w - 8))
            logy = ly + live_h + 1
            if log and logy + 3 <= h - 2:
                logh = h - 2 - logy
                ui.card(logy, 2, logh, w - 4, title="Activity Log", color=T.RULE, title_color=T.PINK)
                for i, ln in enumerate(log[-(logh - 2):]):
                    ui.segs(logy + 1 + i, 4, self._log_segs(ln), maxw=max(0, w - 8))

            ui.footer("b build template · c refresh catalog · p prune dead · s sbx login help · q back")
            self.scr.refresh()
            k = ui.key(1000)
            self.tick += 1
            if k is None or k == "RESIZE":
                continue
            if k in ("q", ESC, SIGINT):
                return
            if k == "b":
                self.live_task("Building the sandbox template", lambda say: sandbox.setup(say, cpus=self.s.cpus()))
                catalog.reload()
                metrics.reload()
                self._cat_count = None
                self.st.kick()
            elif k == "c":
                self.live_task("Refreshing gbench's catalog", sandbox.refresh_catalog)
                catalog.reload()
                metrics.reload()
                self._cat_count = None
            elif k == "p":
                gone = ui.busy("pruning dead sandboxes…", sandbox.prune)
                msg = f"✔ pruned {len(gone)} sandbox(es)"
                log.append(msg)
                ui.notify(msg, T.GREEN)
                self.st.kick()
            elif k == "s":
                ui.message("Docker Sandboxes needs one sign-in on this machine. Quit gbench-tui and run:\n\n    sbx login\n\n"
                           "Then build the template here (b). Every run then creates its own sandbox from it.",
                           title="sbx sign-in", color=T.VIOLET)

    def live_task(self, title, fn):
        """Runs fn(say) in a thread and shows its lines live with an animated prism wave bar and scroll."""
        ui = self.ui
        lines: list = []
        done = {"v": None, "end": False, "t1": 0.0}
        t0 = time.monotonic()
        follow = True
        top = 0

        def work():
            try:
                done["v"] = fn(lines.append)
            except Exception as e:                   # noqa: BLE001
                lines.append(f"✗ {e}")
            done["t1"] = time.monotonic()
            done["end"] = True
        threading.Thread(target=work, daemon=True).start()
        while True:
            h, w = ui.size()
            self.scr.erase()
            el = (done["t1"] if done["end"] else time.monotonic()) - t0
            st_txt = "running…" if not done["end"] else ("finished ✔" if done["v"] else "finished ✗")
            ui.header(title, st_txt, right_badge=(f"⏱ {dur(el)}", T.PINK if not done["end"] else T.MUTED))
            if not done["end"]:
                ui.wave_bar(3, 3, max(8, w - 42), self.tick, label=f"{SPIN[self.tick % len(SPIN)]} working…")
            else:
                ui.put(3, 3, "✔ Task completed" if done["v"] else "✗ Task failed",
                       ui.color(T.GREEN if done["v"] else T.RED, None, True))
            oy = 5
            oh = max(4, h - oy - 2)
            view = max(1, oh - 2)
            snap_lines = list(lines)
            max_top = max(0, len(snap_lines) - view)
            top = max_top if follow else max(0, min(top, max_top))
            ui.card(oy, 2, oh, w - 4, title="Output", color=T.RULE, title_color=T.CYAN,
                    badge=f"{len(snap_lines)} lines", badge_color=T.DIM)
            for i in range(view):
                j = top + i
                if j >= len(snap_lines):
                    break
                ui.segs(oy + 1 + i, 4, self._log_segs(snap_lines[j]), maxw=max(0, w - 8))
            ui.scrollbar(oy + 1, w - 3, view, top, view, len(snap_lines))
            ui.footer("↑↓ scroll · Enter / q back" if done["end"] else "↑↓ scroll · please wait…")
            self.scr.refresh()
            k = ui.key(150 if not done["end"] else 500)
            self.tick += 1
            if k in (curses.KEY_UP, "k"):
                follow, top = False, max(0, top - 1)
            elif k in (curses.KEY_DOWN, "j"):
                top = min(max_top, top + 1)
                if top >= max_top:
                    follow = True
            elif k == curses.KEY_PPAGE:
                follow, top = False, max(0, top - view)
            elif k == curses.KEY_NPAGE:
                top = min(max_top, top + view)
                if top >= max_top:
                    follow = True
            elif done["end"] and k in ("q", ESC, SIGINT) + ENTER:
                return done["v"]

    # ------------------------------------------------------------------ settings
    def screen_settings(self):
        s = self.s
        prev_theme = T.current_theme()
        theme_choices = [(name, T.THEMES[name]["label"], T.THEMES[name]["desc"]) for name in T.THEME_ORDER]
        vals = {
            "theme": getattr(s, "theme", "aurora"),
            "sandbox_memory_gb": s.sandbox_memory_gb, "sandbox_cpus": s.sandbox_cpus, "min_free_gb": s.min_free_gb,
            "results_dir": s.results_dir or str(s.results_path), "stop_server_on_exit": s.stop_server_on_exit,
        }
        fields = [
            Field("theme", "Color theme", "choice", theme_choices, help="live preview with ←→ · also press t on Home"),
            Field("sandbox_memory_gb", "Sandbox RAM (GB)", "int", help="RAM cap of each run's microVM", min_val=2),
            Field("sandbox_cpus", "Sandbox CPUs (0 = auto)", "int",
                  help=f"auto = {max(2, min(6, (os.cpu_count() or 4) // 2))}", min_val=0),
            Field("min_free_gb", "Keep free for system (GB)", "float",
                  help="no sandbox or llama-server starts if host memory would drop below this", min_val=0.5),
            Field("results_dir", "Results folder", help="where runs and leaderboards are stored"),
            Field("stop_server_on_exit", "Stop llama-server on exit", "bool", help="stop servers started by gbench-tui on quit"),
        ]

        def on_change(v):
            if v["theme"] != T.current_theme():
                T.set_theme(v["theme"])

        def preview(y: int, max_h: int):
            w = self.ui.size()[1]
            ch = min(max_h, 6)
            tot = self.st.mem_total or 1
            av = self.st.mem_avail
            used = max(0.0, min(1.0, 1.0 - av / tot))
            info = T.theme_info(vals["theme"])
            self.ui.card(y, 2, ch, w - 4, title=f"Preview · {info['label']}", color=T.RULE, title_color=T.PINK,
                         badge=f"RAM {sysmem.fmt(av)} free", badge_color=T.GREEN)
            if ch >= 4:
                self.ui.prism_rule(y + 1, 4, max(8, w - 8), "█")
            if ch >= 5:
                self.ui.put(y + 2, 4, info["desc"], self.ui.color(T.MUTED), maxw=max(0, w - 8))
            if ch >= 6:
                bw = max(8, min(32, w - 26))
                self.ui.heat_bar(y + 3, 4, bw, used)
                self.ui.put(y + 3, 6 + bw, f"{int(used * 100)}% host RAM used", self.ui.color(T.DIM), maxw=max(0, w - 10 - bw))

        if self.form("Settings", fields, vals, done_label="Save", on_change=on_change, extra=preview, extra_h=6):
            s.theme = vals["theme"]
            T.set_theme(s.theme)
            s.sandbox_memory_gb = max(2, int(vals["sandbox_memory_gb"]))
            s.sandbox_cpus = max(0, int(vals["sandbox_cpus"]))
            s.min_free_gb = float(vals["min_free_gb"])
            s.results_dir = vals["results_dir"]
            s.stop_server_on_exit = vals["stop_server_on_exit"]
            self.save()
            self.ui.notify(f"✔ settings saved (theme: {T.theme_info()['label']})", T.GREEN)
        else:
            T.set_theme(prev_theme)

    # ------------------------------------------------------------------ metrics & formulas
    @staticmethod
    def _metric_dir_badge(direction: str) -> tuple[str, str, int]:
        return {
            "higher": ("↑", "↑ higher is better", T.GREEN),
            "lower": ("↓", "↓ lower is better", T.CYAN),
            "dimension": ("◈", "◈ grouping dimension", T.VIOLET),
            "guard": ("◆", "◆ host safety guard", T.YELLOW),
        }.get(direction, ("·", direction, T.PALE))

    def _live_metric_example(self, m: metrics.MetricDef) -> str:
        """If the user has runs in the leaderboard, compute a live example for gScore."""
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

            # Mode tab strip on row 3
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


def _utf8():
    with contextlib.suppress(locale.Error):
        locale.setlocale(locale.LC_ALL, "")


def run_tui():
    _utf8()
    os.environ.setdefault("TERM", "xterm-256color")

    def main(scr):
        App(scr).main()
    curses.wrapper(main)
