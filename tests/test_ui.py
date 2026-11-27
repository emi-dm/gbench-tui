"""Tests for theme palette, animations, progress parsing, UI primitives, and full TUI screens."""
from __future__ import annotations

import curses
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("GBTUI_HOME", tempfile.mkdtemp())
os.environ.setdefault("GBTUI_CACHE", tempfile.mkdtemp())

from gbtui import cli, config, metrics, results as R, runner, theme as T, tui, ui as U  # noqa: E402


class FakeScreen:
    """Strict curses window stub: rejects NUL bytes, out-of-bounds writes, and bottom-right cell writes."""

    def __init__(self, h: int = 36, w: int = 120, keys: list | None = None):
        self.h = h
        self.w = w
        self.keys = list(keys or [])
        self.grid: dict[tuple[int, int], str] = {}
        self.calls: list[tuple[int, int, str, int]] = []

    def getmaxyx(self):
        return (self.h, self.w)

    def keypad(self, _flag):
        pass

    def erase(self):
        self.grid.clear()
        self.calls.clear()

    def refresh(self):
        pass

    def timeout(self, _ms):
        pass

    def move(self, _y, _x):
        pass

    def addstr(self, y, x, text, attr=0):
        if "\x00" in text:
            raise ValueError("embedded null character")
        if y < 0 or y >= self.h or x < 0 or x >= self.w:
            raise curses.error("out of bounds")
        dw = U.disp_w(text)
        if y == self.h - 1 and x + dw >= self.w:
            raise curses.error("bottom-right cell")
        if x + dw > self.w:
            raise curses.error("overflow right edge")
        self.calls.append((y, x, text, attr))
        cx = x
        for ch in text:
            cw = U._char_w(ch)
            if cx < self.w:
                self.grid[(y, cx)] = ch
            cx += cw

    def get_wch(self):
        if not self.keys:
            return "q"
        k = self.keys.pop(0)
        if k is None:
            raise curses.error("timeout")
        if k == "KBINT":
            raise KeyboardInterrupt
        return k

    def row_text(self, y: int) -> str:
        return "".join(self.grid.get((y, x), " ") for x in range(self.w))

    def all_text(self) -> str:
        return "\n".join(self.row_text(y) for y in range(self.h))


def _curses_ctx(colors: int = 256, pairs: int = 32767):
    return [
        mock.patch.object(curses, "curs_set"),
        mock.patch.object(curses, "start_color"),
        mock.patch.object(curses, "use_default_colors"),
        mock.patch.object(curses, "init_pair"),
        mock.patch.object(curses, "color_pair", side_effect=lambda n: n << 8),
        mock.patch.object(curses, "COLORS", colors, create=True),
        mock.patch.object(curses, "COLOR_PAIRS", pairs, create=True),
    ]


class _CursesEnv:
    def __init__(self, colors: int = 256, pairs: int = 32767):
        self.patches = _curses_ctx(colors, pairs)

    def __enter__(self):
        for p in self.patches:
            p.start()

    def __exit__(self, *a):
        for p in reversed(self.patches):
            p.stop()


class TestThemeAndAnimations(unittest.TestCase):
    def tearDown(self):
        T.set_theme("aurora")

    def test_gradient_and_shimmer(self):
        self.assertEqual(T.grad(0, 10), T.GRADIENT[0])
        self.assertEqual(T.grad(9, 10), T.GRADIENT[-1])
        self.assertEqual(T.grad(0, 1), T.GRADIENT[0])
        cols = {T.shimmer(i, 50, tick) for tick in range(60) for i in range(0, 50, 5)}
        cols2d = {T.shimmer_2d(r, c, 6, 50, tick) for tick in range(0, 55, 5) for r in range(6) for c in range(0, 50, 10)}
        self.assertTrue(all(0 <= c <= 255 for c in cols | cols2d))
        self.assertIn(T.LIGHT, cols)
        self.assertIn(T.LIGHT, cols2d)

    def test_all_themes_and_cycle(self):
        seen = []
        T.set_theme("aurora")
        for _ in range(len(T.THEME_ORDER)):
            seen.append(T.current_theme())
            self.assertTrue(len(T.GRADIENT) >= 8)
            self.assertTrue(len(T.MENU_COLORS) >= len(tui.App.MENU))
            self.assertTrue(len(T.PILLAR_COLORS) >= 6)
            T.cycle_theme()
        self.assertEqual(tuple(seen), T.THEME_ORDER)
        self.assertEqual(T.current_theme(), "aurora")
        # Unknown theme falls back cleanly to aurora
        self.assertEqual(T.set_theme("nonexistent"), "aurora")

    def test_wave_cell(self):
        chars = {T.wave_cell(i, 30, t)[0] for t in range(40) for i in range(30)}
        self.assertTrue({"█", "▓", "▒", "░", "─"} <= chars)

    def test_on_sel_contrast_promotion(self):
        T.set_theme("deepmind")
        for dark in (T.DEEP, 26, 27, T.BLUE, T.CYAN):
            self.assertNotEqual(T.on_sel(dark), dark)
        T.set_theme("aurora")
        self.assertNotEqual(T.on_sel(T.DIM), T.DIM)
        self.assertEqual(T.on_sel(None), T.LIGHT)
        self.assertEqual(T.on_sel(T.LIGHT), T.LIGHT)

    def test_score_and_tier_colors(self):
        T.set_theme("aurora")
        self.assertEqual(T.score_color(95), T.GREEN)
        self.assertEqual(T.score_color(75), T.CYAN)
        self.assertEqual(T.score_color(60), T.VIOLET)
        self.assertEqual(T.score_color(40), T.YELLOW)
        self.assertEqual(T.score_color(10), T.RED)
        self.assertEqual(T.score_color(None), T.MUTED)
        self.assertEqual(T.tier_color("golden"), T.YELLOW)
        self.assertEqual(T.status_color("ok"), T.GREEN)
        self.assertEqual(T.status_color("failed"), T.RED)
        T.set_theme("deepmind")
        self.assertEqual(T.score_color(95), T.LIGHT)
        self.assertEqual(T.score_color(10), T.RED)


class TestProgressParsing(unittest.TestCase):
    def setUp(self):
        runner.LIVE.__init__()

    def test_golden_progress_no_double_count(self):
        runner._parse_progress("Running Golden Set evaluation for lfm3-1.7b (16 tasks)")
        self.assertEqual(runner.LIVE.current, "Golden Set")
        self.assertEqual(runner.LIVE.progress, (0, 16))
        runner._parse_progress("Golden case 'math-canonical' starting")
        runner._parse_progress("Golden case 'math-canonical' passed")
        self.assertEqual(runner.LIVE.current, "math-canonical")
        self.assertEqual(runner.LIVE.progress, (1, 16))

    def test_tqdm_clamped_and_suite_resets_progress(self):
        runner._parse_progress("gsm8k: 100%|█████| 25/20 [00:12<00:00]")
        self.assertEqual(runner.LIVE.progress, (20, 20))
        runner._parse_progress("Running eval suite 'ifeval'")
        self.assertEqual(runner.LIVE.current, "ifeval")
        self.assertEqual(runner.LIVE.progress, (0, 0))
        # Noise words must not clobber current suite
        runner._parse_progress("Evaluating model on 20 prompts")
        self.assertEqual(runner.LIVE.current, "ifeval")

    def test_live_dropped_counter(self):
        for i in range(4005):
            runner.LIVE.add(f"line-{i}")
        snap = runner.snapshot()
        self.assertEqual(snap["dropped"], 1000)
        self.assertEqual(len(snap["lines"]), 3005)
        self.assertEqual(snap["lines"][0], "line-1000")


class TestUIPrimitives(unittest.TestCase):
    def test_nul_ansi_and_cjk_safe(self):
        scr = FakeScreen(24, 80)
        with _CursesEnv():
            u = U.UI(scr)
            # NUL + ANSI + CJK must never raise ValueError or overflow width
            u.put(1, 0, "hello\x00\x1b[31mred\x1b[0m世界テスト", u.color(T.LIGHT), maxw=16)
            u.segs(2, 0, [("日本語テスト", u.color(T.ICE)), (" after", u.color(T.SOFT))], maxw=16, skip=4)
        self.assertNotIn("\x00", scr.all_text())
        self.assertNotIn("\x1b", scr.all_text())

    def test_8_color_pair_dedup(self):
        scr = FakeScreen(24, 80)
        with _CursesEnv(colors=8, pairs=64):
            u = U.UI(scr)
            for fg in T.GRADIENT + [T.RULE, T.DIM, T.MUTED, T.SOFT, T.LIGHT, T.GREEN, T.RED, T.YELLOW]:
                for bg in (None, T.SEL, T.BAR):
                    u.color(fg, bg)
            # Collapsed into basic 8 colours — far below 64 pair limit
            self.assertLess(len(u._pairs), 30)

    def test_edit_border_and_ctrl_c(self):
        scr = FakeScreen(24, 80, keys=["a", "KBINT"])
        with _CursesEnv():
            u = U.UI(scr)
            res = u.edit("Name", "hi", hint="hint text")
        self.assertIsNone(res)
        # Right border of edit box must remain intact (`│`), never overwritten by hint text
        box_rows = [scr.row_text(y) for y in range(24) if "│" in scr.row_text(y)]
        self.assertTrue(len(box_rows) >= 3)
        for row in box_rows:
            self.assertEqual(row.count("│"), 2)

    def test_card_bars_toast_and_scrollbar(self):
        scr = FakeScreen(30, 100)
        with _CursesEnv():
            u = U.UI(scr)
            u.header("Home", right_badge=("badge", T.ICE))
            u.card(3, 2, 8, 50, title="Status", badge="● ready")
            u.bar(5, 4, 20, 0.65, T.ICE, bg=T.SEL, gradient=True)
            u.heat_bar(6, 4, 20, 0.72)
            u.sparkline(6, 26, [0.2, 0.4, 0.6, 0.8, 0.5])
            u.wave_bar(7, 4, 24, tick=7, label="booting…")
            u.stacked_bar(8, 4, 20, [(12, T.GREEN), (2, T.RED), (2, T.DIM)])
            u.prism_rule(9, 4, 30)
            u.scrollbar(4, 51, 6, top=5, view=6, total=30)
            u.notify("saved!", T.GREEN, duration=5.0)
            u.footer("↑↓ move · q quit")
        txt = scr.all_text()
        self.assertIn("g·bench", txt)
        self.assertIn("badge", txt)
        self.assertIn("Status", txt)
        self.assertIn("● ready", txt)
        self.assertIn("saved!", txt)
        self.assertIn("┃", txt)

    def test_toast_expires(self):
        scr = FakeScreen(24, 80)
        with _CursesEnv():
            u = U.UI(scr)
            u.notify("brief", T.YELLOW, duration=0.01)
            time.sleep(0.02)
            u.footer("q quit")
        self.assertNotIn("brief", scr.all_text())

    def test_skip_heading_and_clamp_top(self):
        items = [(None, "H1", ""), ("a", "a", ""), (None, "H2", ""), ("b", "b", "")]
        self.assertEqual(tui._skip_heading(items, 0, 1), 1)
        self.assertEqual(tui._skip_heading(items, 2, 1), 3)
        self.assertEqual(tui._skip_heading(items, 2, -1), 1)
        self.assertEqual(tui._clamp_top(1, 1, 2, 4, items=items), 0)

    def test_screens_render_cleanly_at_80x24_and_60x18(self):
        for h, w in ((24, 80), (18, 60)):
            scr = FakeScreen(h, w, keys=["q", "q", "q"])
            with _CursesEnv(), mock.patch.object(tui, "Status") as st_cls:
                st = mock.MagicMock()
                st.mem_avail = 32 * 1024 ** 3
                st.mem_total = 64 * 1024 ** 3
                st.swap = 0
                st.mem_hist = [0.42, 0.48, 0.50]
                st.ep_ok = {}
                st.sbx = "ready"
                st.sbx_list = []
                st_cls.return_value = st
                app = tui.App(scr)
                app.s.results_dir = tempfile.mkdtemp()
                app.s.endpoints = [config.Endpoint(name="lfm3-1.7b", gguf="/tmp/fake.gguf")]
                app.s.active = "lfm3-1.7b"
                app.draw_home(0)
                home_txt = scr.all_text()
                self.assertIn("Workspace", home_txt)
                if (h, w) == (24, 80):
                    self.assertIn("Environment", home_txt)
                    self.assertIn("Metrics", home_txt)
                    self.assertIn("Settings", home_txt)
                # Also verify Endpoints, Run, Leaderboard, Sandbox, Settings open & close cleanly
                scr.keys = ["q"]
                app.screen_endpoints()
                scr.keys = ["q"]
                app.screen_run()
                scr.keys = ["q"]
                app.screen_leaderboard()
                scr.keys = ["q"]
                app.screen_sandbox()
                scr.keys = ["q"]
                app.screen_settings()
                scr.keys = ["j", "c", "\t", "j", "l", "l", "\n", "q", "b", "q", "q"]
                app.screen_metrics()

    def test_metrics_screen_split_and_cli(self):
        self.assertGreaterEqual(len(metrics.METRICS), 10)
        bms = metrics.benchmark_metrics()
        self.assertGreaterEqual(len(bms), 10)
        cb = metrics.benchmark_metric("causalbench")
        self.assertIsNotNone(cb)
        self.assertIn("F1", cb.metric_tag)
        self.assertEqual(metrics.benchmark_metric("decode tok/s (in512)").name, "personal_perf")
        self.assertEqual(metrics.benchmark_metric("golden_set").name, "golden_set")

        plain = metrics.format_plain()
        for needle in ("gScore", "golden_set", "decode tok/s", "TTFT", "compact (bits)", "1000.0 / TPOT_ms"):
            self.assertIn(needle, plain)
        ifeval_txt = metrics.format_plain("ifeval")
        self.assertIn("prompt", ifeval_txt.lower())
        self.assertIn("strict", ifeval_txt.lower())

        # Render split two-pane Metrics screen (both Core Formulas & By Benchmark pre-focused on causalbench)
        scr = FakeScreen(36, 120, keys=["\n", "q", "1", "j", "b", "q", "q"])
        with _CursesEnv(), mock.patch.object(tui, "Status") as st_cls:
            st = mock.MagicMock()
            st.mem_avail = 32 * 1024 ** 3
            st.mem_total = 64 * 1024 ** 3
            st.swap = 0
            st.mem_hist = [0.5]
            st.ep_ok = {}
            st.sbx = "ready"
            st.sbx_list = []
            st_cls.return_value = st
            app = tui.App(scr)
            app.s.results_dir = tempfile.mkdtemp()
            app.screen_metrics(initial_tab="bench", focus_name="causalbench")
        txt = scr.all_text()
        self.assertIn("Metrics & Formulas", txt)
        self.assertIn("By Benchmark", txt)

        # CLI `metrics` and `metrics <benchmark>` subcommands return 0
        with mock.patch("builtins.print") as p:
            self.assertEqual(cli.main(["metrics"]), 0)
            self.assertTrue(any("gScore" in str(call) for call in p.call_args_list))
        with mock.patch("builtins.print") as p:
            self.assertEqual(cli.main(["metrics", "causalbench"]), 0)
            self.assertTrue(any("F1" in str(call) for call in p.call_args_list))


if __name__ == "__main__":
    unittest.main()
