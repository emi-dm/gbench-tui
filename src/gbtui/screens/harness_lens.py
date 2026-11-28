"""Harness Lens screen: inspect & compare agent harnesses (inspired by harness-hub/harness-lens) + interactive Play."""
from __future__ import annotations

import curses
import contextlib

from .. import harness_lens, servers
from .. import theme as T
from ..ui import ESC, SIGINT, UI
from .common import ENTER, _clamp_top


class HarnessLensMixin:
    def screen_harness_lens(self):
        ui = self.ui
        specs = harness_lens.all_specs()
        sel, top = 0, 0
        compare_mode = False
        cmp_idx = 2 if len(specs) > 2 else 0  # default compare against Hermes

        while True:
            h, w = ui.size()
            self.scr.erase()
            ep = self.s.endpoint()
            ep_tag = f"target: {ep.name}" + (f" [{ep.effective_adapter}]" if ep and ep.effective_adapter else "") if ep else "no active endpoint"
            rbadge = ("◈ COMPARE A vs B", T.PINK) if compare_mode else (ep_tag[:34], T.CYAN if ep else T.YELLOW)
            ui.header("Harness Lens", "anatomy of coding-agent harnesses (prompts, tools, Code Mode, compaction) · inspired by harness-hub/harness-lens",
                      right_badge=rbadge)

            cur = specs[sel]
            if compare_mode:
                other = specs[cmp_idx]
                self._draw_harness_compare(cur, other, h, w)
                ui.footer("↑↓ left harness · ←→ right harness · p compare sys prompts · c detail · L play · q back")
                self.scr.refresh()
                k = ui.wait_key()
                if k is None or k == "RESIZE":
                    continue
                if k in ("q", ESC, SIGINT):
                    return
                if k in ("c", "\t"):
                    compare_mode = False
                elif k in (curses.KEY_UP, "k"):
                    sel = (sel - 1) % len(specs)
                elif k in (curses.KEY_DOWN, "j"):
                    sel = (sel + 1) % len(specs)
                elif k in (curses.KEY_LEFT, "h"):
                    cmp_idx = (cmp_idx - 1) % len(specs)
                elif k in (curses.KEY_RIGHT, "l"):
                    cmp_idx = (cmp_idx + 1) % len(specs)
                elif k == "p":
                    self._compare_system_prompts(cur, other)
                elif k == "L":
                    self.launch_harness_play(cur)
                continue

            split = w >= 88
            body_y = 3
            body_h = max(6, h - body_y - 2)
            if split:
                left_w = min(36, max(26, int(w * 0.30)))
                list_h = body_h
            else:
                left_w = max(12, w - 4)
                detail_h = min(11, max(5, body_h // 2))
                list_h = max(4, body_h - detail_h)
            view = max(1, list_h - 2)
            top = _clamp_top(top, sel, view, len(specs))

            ui.card(body_y, 2, list_h, left_w, title="Harnesses", color=T.RULE, title_color=T.VIOLET,
                    badge=f"{len(specs)} agents", badge_color=T.DIM, icon="⟐")
            for i in range(view):
                j = top + i
                if j >= len(specs):
                    break
                sp = specs[j]
                ry = body_y + 1 + i
                is_sel = j == sel
                col = T.pillar_color(j)
                if is_sel:
                    ui.sel_bar(ry, 3, left_w - 2, col)
                bg = T.SEL if is_sel else None
                inst = bool(harness_lens.installed_bin(sp))
                ui.put(ry, 4, "●" if inst else "○", ui.color(T.GREEN if inst else T.DIM, bg, inst))
                ui.put(ry, 6, sp.name[: left_w - 14], ui.color(T.LIGHT if is_sel else T.SOFT, bg, is_sel))
                tag = f"{len(sp.tools)}t"
                if left_w >= 28:
                    ui.put(ry, 2 + left_w - len(tag) - 2, tag, ui.color(T.CYAN if is_sel else T.DIM, bg))
            ui.scrollbar(body_y + 1, 2 + left_w - 1, view, top, view, len(specs))

            if split:
                dx, dy, dw, dh = 2 + left_w + 1, body_y, max(22, w - left_w - 5), body_h
            else:
                dx, dy, dw, dh = 2, body_y + list_h, max(12, w - 4), max(4, h - (body_y + list_h) - 2)
            self._draw_harness_detail(cur, dx, dy, dw, dh, T.pillar_color(sel))

            ui.footer("↑↓ select · p sys prompt · Enter full card · c compare A/B · L play · M matrix · q back")
            self.scr.refresh()
            k = ui.wait_key()
            if k is None or k == "RESIZE":
                continue
            if k in ("q", ESC, SIGINT):
                return
            if k in (curses.KEY_UP, "k"):
                sel = (sel - 1) % len(specs)
            elif k in (curses.KEY_DOWN, "j"):
                sel = (sel + 1) % len(specs)
            elif k in ("c", "\t"):
                compare_mode = True
                if cmp_idx == sel:
                    cmp_idx = (sel + 1) % len(specs)
            elif k == "p":
                self._show_system_prompt(cur)
            elif k in ENTER:
                self._show_harness_pager(cur)
            elif k == "L":
                self.launch_harness_play(cur)
            elif k == "M":
                self.screen_matrix()

    def _draw_harness_detail(self, sp: harness_lens.HarnessSpec, dx: int, dy: int, dw: int, dh: int, accent: int):
        ui = self.ui
        inst_bin = harness_lens.installed_bin(sp)
        bin_lbl = sp.cli_bins[0] if sp.cli_bins else sp.id
        badge = f"● CLI ready ({bin_lbl})" if inst_bin else "○ built-in REPL / Harbor"
        ui.card(dy, dx, dh, dw, title=f"{sp.name} · {sp.tagline}", color=accent, title_color=accent,
                badge=badge, badge_color=T.GREEN if inst_bin else T.MUTED)
        cy = dy + 1
        max_cy = dy + dh - 1
        inner_w = max(10, dw - 4)

        def row(lbl, val, col=T.SOFT, bold=False):
            nonlocal cy
            if cy >= max_cy:
                return
            ui.put(cy, dx + 2, f"{lbl:<13}", ui.color(T.DIM))
            ui.put(cy, dx + 15, val, ui.color(col, None, bold), maxw=max(0, inner_w - 13))
            cy += 1

        row("prompt size", f"~{sp.prompt_tokens_est:,} tok · min ctx {sp.min_ctx:,}", T.CYAN, bold=True)
        row("tool surface", f"{len(sp.tools)} tools · {sp.edit_strategy}", T.PALE)
        row("code mode", sp.code_mode, T.GREEN if sp.code_mode.lower().startswith("native") else T.YELLOW)
        row("compaction", sp.compaction, T.SOFT)
        row("sandbox", sp.sandbox_model, T.MUTED)
        row("harbor id", f"--agent {sp.id}  · repo: {sp.repo}", T.VIOLET)
        if cy < max_cy:
            cy += 1
        if cy < max_cy:
            ui.put(cy, dx + 2, "SYSTEM PROMPT ANATOMY", ui.color(T.PINK, None, True), maxw=inner_w)
            cy += 1
        for sec_name, sec_desc in sp.system_prompt_sections:
            if cy >= max_cy - 3:
                break
            ui.put(cy, dx + 2, f"▸ {sec_name[:20]:<21}", ui.color(T.LIGHT, None, True))
            ui.put(cy, dx + 25, sec_desc, ui.color(T.MUTED), maxw=max(0, dx + dw - 2 - (dx + 25)))
            cy += 1
        if cy + 1 < max_cy:
            cy += 1
            ui.put(cy, dx + 2, "TOOLS: ", ui.color(T.DIM))
            ui.put(cy, dx + 9, ", ".join(sp.tools), ui.color(T.CYAN), maxw=max(0, inner_w - 7))
            cy += 1
        if cy < max_cy:
            ui.put(cy, dx + 2, "STRENGTHS: ", ui.color(T.GREEN, None, True))
            ui.put(cy, dx + 13, sp.strengths, ui.color(T.SOFT), maxw=max(0, inner_w - 11))

    def _draw_harness_compare(self, a: harness_lens.HarnessSpec, b: harness_lens.HarnessSpec, h: int, w: int):
        ui = self.ui
        body_y = 3
        body_h = max(6, h - body_y - 2)
        half = (w - 5) // 2
        self._draw_harness_detail(a, 2, body_y, half, body_h, T.SKY)
        self._draw_harness_detail(b, 3 + half, body_y, w - 5 - half, body_h, T.PINK)

    def _show_harness_pager(self, sp: harness_lens.HarnessSpec):
        c = self.ui.color
        w = self.ui.size()[1]
        wrap_w = max(24, w - 10)
        lines = [
            [("harness   ", c(T.DIM)), (f"{sp.name} ({sp.id})", c(T.LIGHT, None, True)), (f"  — {sp.tagline}", c(T.PALE))],
            [("repo      ", c(T.DIM)), (sp.repo, c(T.CYAN))],
            [("prompt    ", c(T.DIM)), (f"~{sp.prompt_tokens_est:,} tokens · min ctx {sp.min_ctx:,}", c(T.PINK, None, True))],
            [("edit mode ", c(T.DIM)), (sp.edit_strategy, c(T.SOFT))],
            [("code mode ", c(T.DIM)), (sp.code_mode, c(T.GREEN if sp.code_mode.lower().startswith("native") else T.YELLOW))],
            [("compaction", c(T.DIM)), (" " + sp.compaction, c(T.SOFT))],
            [("sandbox   ", c(T.DIM)), (sp.sandbox_model, c(T.MUTED))],
            [("install   ", c(T.DIM)), (sp.install_hint, c(T.CYAN))],
            [("─" * max(10, w - 6), c(T.RULE_DIM))],
            [("", 0)],
            [("◆ ", c(T.VIOLET, None, True)), ("System Prompt Sections", c(T.LIGHT, None, True))],
        ]
        for sec_name, sec_desc in sp.system_prompt_sections:
            lines.append([(f"  • {sec_name:<22} ", c(T.CYAN, None, True)), (sec_desc, c(T.SOFT))])
        lines += [[("", 0)], [("◆ ", c(T.GREEN, None, True)), (f"Tools ({len(sp.tools)})", c(T.LIGHT, None, True))]]
        for part in UI.wrap(", ".join(sp.tools), wrap_w):
            lines.append([("  " + part, c(T.GREEN))])
        lines += [[("", 0)], [("◆ ", c(T.PINK, None, True)), ("Strengths", c(T.LIGHT, None, True))]]
        for part in UI.wrap(sp.strengths, wrap_w):
            lines.append([("  + " + part, c(T.GREEN))])
        lines += [[("", 0)], [("⚠ ", c(T.YELLOW, None, True)), ("Caveats", c(T.YELLOW, None, True))]]
        for part in UI.wrap(sp.caveats, wrap_w):
            lines.append([("  - " + part, c(T.MUTED))])
        prompt_txt, src = harness_lens.load_system_prompt(sp)
        if prompt_txt:
            lines += [
                [("", 0)],
                [("─" * max(10, w - 6), c(T.RULE_DIM))],
                [("◆ ", c(T.CYAN, None, True)), (f"System Prompt ({src})", c(T.LIGHT, None, True))],
                [("", 0)],
            ]
            lines += self._style_prompt_lines(prompt_txt, wrap_w)
        self.ui.pager(f"Harness Lens · {sp.name}", lines, sub=f"Harbor --agent {sp.id}")

    def _style_prompt_lines(self, prompt_txt: str, wrap_w: int) -> list:
        c = self.ui.color
        out = []
        for raw in prompt_txt.splitlines():
            st = raw.strip()
            if not st:
                out.append([("", 0)])
                continue
            if st.startswith("#"):
                out.append([(raw, c(T.PINK, None, True))])
            elif st.startswith("<") and st.endswith(">"):
                out.append([(raw, c(T.VIOLET, None, True))])
            elif st.startswith(("- ", "* ")):
                for i, part in enumerate(UI.wrap(raw, wrap_w)):
                    out.append([(part, c(T.CYAN if i == 0 and ":" in part[:26] else T.SOFT))])
            elif st.startswith(("<<<<<<<", "=======", ">>>>>>>")):
                out.append([(raw, c(T.YELLOW, None, True))])
            else:
                for part in UI.wrap(raw, wrap_w):
                    out.append([(part, c(T.SOFT))])
        return out

    def _show_system_prompt(self, sp: harness_lens.HarnessSpec):
        """Dedicated pager to read a harness's full system prompt (toggle live captured vs reference if available)."""
        w = self.ui.size()[1]
        wrap_w = max(24, w - 8)
        cap_txt, cap_src = harness_lens.captured_system_prompt(sp)
        state = {"prefer_captured": bool(cap_txt)}

        def build():
            txt, src = harness_lens.load_system_prompt(sp, prefer_captured=state["prefer_captured"])
            est_tok = max(1, len(txt) // 4)
            c = self.ui.color
            head = [
                [("harness ", c(T.DIM)), (f"{sp.name} ({sp.id})", c(T.LIGHT, None, True)),
                 (f"  · {len(txt.splitlines())} lines · {len(txt):,} chars · ~{est_tok:,} tok", c(T.CYAN))],
                [("source  ", c(T.DIM)), (src, c(T.GREEN if "captured" in src else T.PALE))],
                [("─" * max(10, w - 6), c(T.RULE_DIM))],
                [("", 0)],
            ]
            sub = f"{src} · ~{est_tok:,} tok"
            return f"System Prompt · {sp.name}", head + self._style_prompt_lines(txt, wrap_w), sub

        def toggle_src():
            if not cap_txt:
                return None
            state["prefer_captured"] = not state["prefer_captured"]
            t, lns, sb = build()
            return ("replace", t, lns, sb)

        title, lines, sub = build()
        hints = "↑↓ scroll · / search · v toggle captured/ref · ←→ sideways · q back" if cap_txt else \
                "↑↓ scroll · / search · ←→ sideways · q back"
        self.ui.pager(title, lines, sub=sub, hints=hints, keys={"v": toggle_src} if cap_txt else None)

    def _compare_system_prompts(self, a: harness_lens.HarnessSpec, b: harness_lens.HarnessSpec):
        """Side-by-side split view comparing two harnesses' system prompts."""
        ui = self.ui
        pa, src_a = harness_lens.load_system_prompt(a)
        pb, src_b = harness_lens.load_system_prompt(b)
        scroll = 0
        while True:
            h, w = ui.size()
            self.scr.erase()
            ui.header(f"System Prompts · {a.name} vs {b.name}",
                      f"~{max(1, len(pa) // 4):,} tok vs ~{max(1, len(pb) // 4):,} tok",
                      crumbs=["Harness Lens", "System Prompts"])
            py = 3
            ph = max(6, h - py - 2)
            half = (w - 5) // 2
            cw_a = max(10, half - 4)
            cw_b = max(10, (w - 5 - half) - 4)
            la = [part for raw in pa.splitlines() for part in (UI.wrap(raw, cw_a) or [""])]
            lb = [part for raw in pb.splitlines() for part in (UI.wrap(raw, cw_b) or [""])]
            max_lines = max(len(la), len(lb), 1)
            view = max(1, ph - 2)
            scroll = max(0, min(scroll, max(0, max_lines - view)))
            ui.card(py, 2, ph, half, title=f"{a.name} (~{max(1, len(pa) // 4)}t)", color=T.RULE, title_color=T.SKY,
                    badge="captured" if "captured" in src_a else "ref", badge_color=T.DIM)
            ui.card(py, 3 + half, ph, w - 5 - half, title=f"{b.name} (~{max(1, len(pb) // 4)}t)", color=T.RULE, title_color=T.PINK,
                    badge="captured" if "captured" in src_b else "ref", badge_color=T.DIM)
            for i in range(view):
                j = scroll + i
                if j < len(la):
                    col = T.PINK if la[j].lstrip().startswith("#") else (T.VIOLET if la[j].lstrip().startswith("<") else T.SOFT)
                    ui.put(py + 1 + i, 4, la[j], ui.color(col), maxw=cw_a)
                if j < len(lb):
                    col = T.PINK if lb[j].lstrip().startswith("#") else (T.VIOLET if lb[j].lstrip().startswith("<") else T.SOFT)
                    ui.put(py + 1 + i, 5 + half, lb[j], ui.color(col), maxw=cw_b)
            ui.scrollbar(py + 1, w - 3, view, scroll, view, max_lines)
            ui.footer("↑↓ / PgUp PgDn scroll · q back")
            self.scr.refresh()
            k = ui.wait_key()
            if k is None or k == "RESIZE":
                continue
            if k in ("q", ESC, SIGINT):
                return
            if k in (curses.KEY_UP, "k"):
                scroll = max(0, scroll - 1)
            elif k in (curses.KEY_DOWN, "j"):
                scroll = min(max(0, max_lines - view), scroll + 1)
            elif k == curses.KEY_PPAGE:
                scroll = max(0, scroll - view)
            elif k == curses.KEY_NPAGE:
                scroll = min(max(0, max_lines - view), scroll + view)

    def launch_harness_play(self, sp: harness_lens.HarnessSpec):
        """Suspend curses, ensure endpoint server is up, launch interactive harness session (Hermes/Pi/OpenCode/REPL)."""
        ui = self.ui
        if not self.s.endpoints:
            ui.message("Add an endpoint first in 1 · Endpoints.", title="Launch Harness", color=T.YELLOW)
            return
        ep_items = [(e, e.name, f"{e.effective_quant or '—'} · {e.effective_adapter or 'base'} · {e.base_url}") for e in self.s.endpoints]
        cur_idx = next((i for i, e in enumerate(self.s.endpoints) if e.name == self.s.active), 0)
        ep = ui.choose(f"Select model to play with {sp.name}", ep_items, sel=cur_idx)
        if ep is None:
            return
        if ep.gguf and not servers.probe(ep, timeout=1.2)["ok"]:
            why = ui.busy(f"starting llama-server for {ep.name}…", servers.start_server, ep, self.s.min_free_gb)
            if why:
                ui.message(why, title="llama-server", color=T.RED)
                return
        curses.def_prog_mode()
        curses.endwin()
        try:
            harness_lens.run_interactive_playground(sp, ep, self.s.min_free_gb)
        finally:
            with contextlib.suppress(curses.error):
                curses.reset_prog_mode()
                self.scr.refresh()
        self.st.probe_all()
        ui.notify(f"returned from {sp.name} session ({ep.name})", T.CYAN, 2.5)
