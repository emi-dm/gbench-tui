"""Shared helpers, background Status poller, Field spec, and reusable form / picker / live_task widgets."""
from __future__ import annotations

import curses
import math
import re
import threading
import time
from dataclasses import dataclass, field

from .. import catalog, config, leaderboard, results as R, runner, sandbox, servers, sysmem
from .. import theme as T
from ..ui import ESC, SIGINT, SPIN, UI, dur

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


def _adapter_color(adapter: str) -> int:
    low = (adapter or "").lower()
    if not low or low == "base":
        return T.DIM
    if "super" in low:
        return T.GREEN
    if "pi" in low:
        return T.SKY
    if "opencode" in low:
        return T.VIOLET
    if "hermes" in low:
        return T.YELLOW
    return T.PINK


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

    def stop(self):
        self._stop = True
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


class CommonMixin:
    """Shared chrome, caching, syntax highlighting, generic form, multi-picker, and live task runner."""

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

    def footer_status(self):
        out = []
        L = runner.snapshot()
        if L["active"]:
            d, t = L["progress"]
            prog = f" {d}/{t}" if t else ""
            step = f" [{L['matrix_pos']}]" if L.get("matrix_pos") else ""
            out.append((f"{SPIN[self.tick % len(SPIN)]} {L['current'] or L['phase']}{prog}{step}", T.PINK))
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
            "  1–9, 0    Jump directly to any screen (d Sandbox · s Settings)\n"
            "  m / h     Quick jump to Metrics (m) or Harness Lens (h)\n"
            "  ↑↓ / j k  Move selection    PgUp/PgDn  Page scroll\n"
            "  Enter     Open / Edit       Space      Toggle / Activate\n"
            "  t         Cycle color theme on Home (10 palettes)\n"
            "  /         Filter or search  q / Esc    Back / Quit\n\n"
            "Workflow\n"
            "  1 Endpoints     → servers, GGUF folder (F), discover (D), LoRA picker (l)\n"
            "  2 Run           → Golden Set, evals, perf, presets (P) or Harbor harness\n"
            "  3 Sweep         → Quantization Sweep or Adapter × Harness Matrix batch\n"
            "  4 Live run      → watch pipeline, batch step & streamed logs\n"
            "  5 Results       → failure taxonomy, token economy, notes (n), Compare A vs B (d)\n"
            "  6 Leaderboard   → rankings, Adapter × Harness heatmap (H), Pareto HTML (o), Report (R)\n"
            "  7 Harness Lens  → inspect Pi/OpenCode/Hermes, compare (c) & interactive Play (L)"
        )
        self.ui.message(txt, title="gbench-tui · shortcuts", color=T.VIOLET)

    def _draw_cmd_highlighted(self, y: int, x: int, cmd: str, width: int, max_lines: int) -> int:
        ui = self.ui
        lines = UI.wrap(cmd, max(20, width))[:max_lines]
        for i, ln in enumerate(lines):
            segs = []
            for tok in re.split(r"(\s+)", ln):
                if tok in ("gbench", "harbor"):
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

    def _log_segs(self, ln: str):
        c = self.ui.color
        if ln.startswith("✗") or "ERROR" in ln or "Traceback" in ln or ln.lstrip().startswith("FAIL "):
            return [(ln, c(T.RED))]
        if ln.startswith("✔"):
            return [(ln, c(T.GREEN, None, True))]
        if ln.startswith("━━━"):
            return [(ln, c(T.PINK, None, True))]
        if ln.startswith("$ "):
            segs = [("$ ", c(T.PINK, None, True))]
            for tok in re.split(r"(\s+)", ln[2:]):
                if tok in ("gbench", "harbor"):
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
                    if isinstance(v, list):
                        txt = f"({len(v)}) {', '.join(v)}" if v else "(all)"
                    else:
                        txt = str(v) if v else "(none — press Enter to pick)"
                    ui.put(ry, vx, txt, ui.color(T.PALE if v else T.DIM, bg, is_sel), maxw=vmax)
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
