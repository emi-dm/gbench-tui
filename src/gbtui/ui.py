"""Curses primitives: multi-theme prism colours, cards, pills, sparklines, animations, dialogs, toasts and pager. Stdlib only."""
from __future__ import annotations

import contextlib
import curses
import os
import re
import textwrap
import threading
import time
import unicodedata

from . import theme as T

os.environ.setdefault("ESCDELAY", "25")
ESC = "\x1b"
SIGINT = "SIGINT"
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
PULSE = "●◉●○"
SPARK = "▁▂▃▄▅▆▇█"

_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _clean(text: str) -> str:
    """Strip ANSI escape sequences, NUL bytes, and control characters so curses never crashes or misaligns."""
    s = str(text).replace("\t", "    ").replace("\r", "").replace("\n", " ")
    if "\x1b" in s:
        s = _ANSI_RE.sub("", s)
    return _CTRL_RE.sub("", s)


def _char_w(ch: str) -> int:
    cat = unicodedata.category(ch)
    if cat in ("Mn", "Me", "Cf"):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def disp_w(text: str) -> int:
    """Terminal display cell width of a cleaned string."""
    if text.isascii():
        return len(text)
    return sum(_char_w(ch) for ch in text)


def trunc_w(text: str, max_cells: int) -> tuple[str, int]:
    """Truncate cleaned string to fit within `max_cells` terminal columns. Returns (substring, cell_width)."""
    if max_cells <= 0 or not text:
        return "", 0
    if text.isascii():
        s = text[:max_cells]
        return s, len(s)
    out = []
    used = 0
    for ch in text:
        cw = _char_w(ch)
        if used + cw > max_cells:
            break
        out.append(ch)
        used += cw
    return "".join(out), used


def skip_w(text: str, cells: int) -> tuple[str, int]:
    """Skip leading `cells` display columns of `text`. Returns (remainder, remaining_cells_to_skip)."""
    if cells <= 0 or not text:
        return text, cells
    if text.isascii():
        if cells >= len(text):
            return "", cells - len(text)
        return text[cells:], 0
    skipped = 0
    for idx, ch in enumerate(text):
        cw = _char_w(ch)
        if skipped + cw > cells:
            return text[idx:], 0
        skipped += cw
        if skipped == cells:
            return text[idx + 1:], 0
    return "", cells - skipped


def _basic(c: int) -> int:
    """xterm-256 → the 8 basic colours (for terminals without 256 colours)."""
    if c in (T.GREEN, T.GREEN_BRIGHT, 29, 36, 43, 49, 79, 80, 84, 85, 86, 114, 120, 121):
        return curses.COLOR_GREEN
    if c in (T.YELLOW, T.GOLD, 215, 221, 222, 227, 228):
        return curses.COLOR_YELLOW
    if c in (T.RED, T.RED_SOFT, T.CORAL, 167, 168, 197, 203, 204, 209, 210):
        return curses.COLOR_RED
    if c in (T.VIOLET, T.PURPLE, T.PINK, T.PALE, 91, 99, 127, 135, 141, 163, 170, 171, 176, 177, 183, 206, 207, 211, 212, 213):
        return curses.COLOR_MAGENTA
    if c in (T.CYAN, T.SKY, T.ICE, 37, 39, 45, 51, 75, 81, 87, 111, 117, 122, 123, 159):
        return curses.COLOR_CYAN
    if c in (T.BLUE, T.DEEP, T.NAVY, T.RULE, 17, 24, 25, 26, 27, 33, 60, 61, 63, 69, 105):
        return curses.COLOR_BLUE
    if c >= 232:
        return curses.COLOR_BLACK if c < 244 else curses.COLOR_WHITE
    return curses.COLOR_WHITE


class UI:
    def __init__(self, scr):
        self.scr = scr
        with contextlib.suppress(curses.error):
            curses.curs_set(0)
        scr.keypad(True)
        curses.start_color()
        with contextlib.suppress(curses.error):
            curses.use_default_colors()
        self._pairs: dict = {}
        self._pairs_theme: str = T.current_theme()
        self.status_fn = None              # () -> list of (text, colour) for the right side of the footer
        self._toast: tuple | None = None   # (text, color, expires_at)

    # ---------- primitives ----------
    def size(self):
        return self.scr.getmaxyx()

    def color(self, fg=None, bg=None, bold=False, dim=False, auto_sel=True) -> int:
        cur_theme = T.current_theme()
        if self._pairs_theme != cur_theme:
            # Reset pair slots on theme switch so scrolling through all 17 themes live never overflows
            # macOS ncurses' 255-pair 8-bit A_COLOR table into A_STANDOUT (reverse video).
            self._pairs.clear()
            self._pairs_theme = cur_theme
        if auto_sel and bg == T.SEL and fg is not None:
            fg = T.on_sel(fg)
        use_256 = getattr(curses, "COLORS", 8) >= 256
        cfg = -1 if fg is None else (fg if use_256 else _basic(fg))
        cbg = -1 if bg is None else (bg if use_256 else _basic(bg))
        key = (cfg, cbg)
        if key not in self._pairs:
            n = len(self._pairs) + 1
            if n >= self._max_pairs():
                # Table full mid-frame: reuse the closest existing pair (same fg, else same bg) instead of
                # overflowing A_COLOR (which bleeds into other attributes and paints "shadows").
                alt = next((p for (f, b), p in self._pairs.items() if f == cfg and b == -1), None) \
                    or next((p for (f, b), p in self._pairs.items() if f == cfg), None)
                base = curses.color_pair(alt) if alt else 0
                return base | (curses.A_BOLD if bold else 0) | (curses.A_DIM if dim else 0)
            try:
                curses.init_pair(n, cfg, cbg)
            except curses.error:
                return (curses.A_BOLD if bold else 0) | (curses.A_DIM if dim else 0)
            self._pairs[key] = n
        a = curses.color_pair(self._pairs[key])
        return a | (curses.A_BOLD if bold else 0) | (curses.A_DIM if dim else 0)

    def put(self, y, x, text, attr=0, maxw=None):
        h, w = self.size()
        if y < 0 or y >= h or x >= w or x < 0:
            return 0
        room = w - x - (1 if y == h - 1 else 0)
        if maxw is not None:
            room = min(room, maxw)
        if room <= 0:
            return 0
        clean = _clean(text)
        clipped, width = trunc_w(clean, room)
        if not clipped:
            return 0
        try:
            self.scr.addstr(y, x, clipped, attr)
        except curses.error:
            pass
        return width

    def segs(self, y, x, segs, maxw=None, skip=0):
        """Draws [(text, attr), …] from column x, skipping the first `skip` display cells."""
        h, w = self.size()
        limit = (w - x) if maxw is None else min(maxw, w - x)
        used = 0
        for text, attr in segs:
            clean = _clean(text)
            if skip:
                clean, skip = skip_w(clean, skip)
                if not clean:
                    continue
            if used >= limit:
                break
            used += self.put(y, x + used, clean, attr, maxw=limit - used)
        return used

    def fill(self, y, x, width, attr=0):
        if width > 0:
            self.put(y, x, " " * width, attr)

    def hline(self, y, x=0, width=None, ch="─", attr=None):
        h, w = self.size()
        ww = (w - x) if width is None else width
        ww -= 1 if y == h - 1 else 0
        if ww > 0:
            self.put(y, x, ch * ww, self.color(T.RULE) if attr is None else attr)

    def prism_rule(self, y, x=0, width=None, ch="─", tick=0):
        """Draws a crisp multi-colour prism gradient rule across `width` columns."""
        h, w = self.size()
        ww = (w - x) if width is None else width
        ww -= 1 if y == h - 1 else 0
        if ww <= 0:
            return
        for j in range(ww):
            col = T.grad(j, ww) if tick == 0 else T.shimmer(j, ww, tick, period=70)
            self.put(y, x + j, ch, self.color(col, None, False, auto_sel=False))

    def pill(self, y, x, text, fg=None, bg=None, bold=True, maxw=None):
        """Draws a solid high-contrast badge ` text ` (auto-picks dark/light ink via WCAG luminance)."""
        bg = T.CYAN if bg is None else bg
        fg = T.contrast_fg(bg) if fg in (None, T.INK, T.LIGHT) else fg
        return self.put(y, x, f" {text} ", self.color(fg, bg, bold, auto_sel=False), maxw=maxw)

    def soft_pill(self, y, x, text, fg=None, bg=None, bold=True, maxw=None):
        """Draws a subtle tinted chip ` text ` on an elevated slate surface."""
        bg = T.BAR_ALT if bg is None else bg
        fg = T.readable_accent(T.ICE if fg is None else fg, on_bg=bg)
        return self.put(y, x, f" {text} ", self.color(fg, bg, bold, auto_sel=False), maxw=maxw)

    # ---------- chrome: header, footer, toast ----------
    def _max_pairs(self) -> int:
        """Usable pair ids: limited by COLOR_PAIRS *and* by the A_COLOR bit mask (0xff00 → 255 on macOS ncurses)."""
        mask = getattr(curses, "A_COLOR", 0xFF00) or 0xFF00
        mask_pairs = (mask >> 8) if mask & 0xFF == 0 else 255
        return max(8, min(getattr(curses, "COLOR_PAIRS", 64), mask_pairs, 32767))

    def recycle_pairs(self, threshold: float = 0.6) -> None:
        """Called at frame start (header): once the session has accumulated many pairs, start a fresh table.
        Safe because every screen fully redraws after `header()`, so no stale cell keeps an old pair id."""
        if len(self._pairs) >= int(self._max_pairs() * threshold):
            self._pairs.clear()

    def header(self, title, sub="", right_badge=None, crumbs=None, tick=0):
        self.recycle_pairs()
        h, w = self.size()
        bar = self.color(T.SOFT, T.BAR)
        self.fill(0, 0, w, bar)
        badge = " ◆ g·bench "
        bgrad = T.BRAND_GRAD or T.GRADIENT
        for i, ch in enumerate(badge):                   # Multi-stop theme prism badge
            bi = min(len(bgrad) - 1, int(i * len(bgrad) / max(1, len(badge))))
            bg_c = bgrad[bi]
            self.put(0, i, ch, self.color(T.contrast_fg(bg_c), bg_c, True, auto_sel=False))
        x = len(badge)
        self.put(0, x, "▌", self.color(T.PINK if T.ACTIVE != "deepmind" else T.CYAN, T.BAR, True, auto_sel=False))
        x += 2
        if crumbs:
            for idx, part in enumerate(crumbs):
                if idx:
                    x += self.put(0, x, " › ", self.color(T.DIM, T.BAR))
                last = idx == len(crumbs) - 1
                x += self.put(0, x, part, self.color(T.LIGHT if last else T.MUTED, T.BAR, last), maxw=max(0, w - x - 2))
        else:
            x += self.put(0, x, title, self.color(T.LIGHT, T.BAR, True), maxw=max(0, w - x - 2))

        tname = f"✦ {T.current_theme()}"
        brand_r = " · gbench "
        right_total = len(tname) + len(brand_r)
        if w - right_total > x + 4:
            rx = w - right_total
            rx += self.put(0, rx, tname, self.color(T.PALE, T.BAR, False))
            self.put(0, rx, brand_r, self.color(T.DIM, T.BAR))

        # Row 1: either subtitle + tinted right pill, or full-width subtle prism accent rule
        rb_w = (disp_w(_clean(right_badge[0])) + 5) if right_badge else 0
        if sub:
            self.put(1, 2, "▍ ", self.color(T.VIOLET, None, True))
            self.put(1, 4, sub, self.color(T.MUTED), maxw=max(0, w - 6 - rb_w))
        else:
            end_x = max(0, w - rb_w)
            accent_w = min(end_x, max(24, w // 2))
            self.prism_rule(1, 0, accent_w, "─", tick=tick)
            if end_x > accent_w:
                self.hline(1, accent_w, end_x - accent_w, "─", self.color(T.RULE_DIM))
        if right_badge and w - rb_w > 6:
            rtxt, rcol = right_badge
            rw = disp_w(_clean(rtxt)) + 2
            self.soft_pill(1, max(2, w - rw - 2), rtxt, fg=rcol, bg=T.BAR_ALT, bold=True)

    def notify(self, text, color=None, duration=2.6):
        """Non-blocking toast notification displayed above the footer until it expires."""
        self._toast = (str(text), T.CYAN if color is None else color, time.monotonic() + duration)

    def toast(self, text, color=None, duration=2.4):
        """Show a non-blocking toast message (also immediately draws it on row h - 2)."""
        self.notify(text, color=T.YELLOW if color is None else color, duration=duration)
        self._draw_toast()

    def toast_timeout_ms(self, default: int = -1) -> int:
        """Timeout (ms) so blocking screens auto-clear expired toasts cleanly."""
        if not self._toast:
            return default
        rem_ms = int((self._toast[2] - time.monotonic()) * 1000) + 60
        rem_ms = max(80, rem_ms)
        return rem_ms if default < 0 else min(default, rem_ms)

    def _draw_toast(self):
        if not self._toast:
            return False
        text, color, exp = self._toast
        if time.monotonic() >= exp:
            self._toast = None
            return False
        h, w = self.size()
        if h < 4:
            return False
        self.fill(h - 2, 0, w)
        msg = f" {_clean(text)} "
        mw = disp_w(msg)
        x = max(1, (w - mw - 2) // 2)
        self.put(h - 2, x, "▌", self.color(color, T.BAR_ALT, True, auto_sel=False))
        self.put(h - 2, x + 1, msg, self.color(color, T.BAR_ALT, True, auto_sel=False), maxw=max(0, w - x - 2))
        return True

    def keyhints(self, y, x, text, maxw, bg=None):
        bg = T.BAR if bg is None else bg
        for i, seg in enumerate(text.split(" · ")):
            if x >= maxw:
                return x
            if i:
                x += self.put(y, x, " · ", self.color(T.DIM, bg), maxw=maxw - x)
            key, _sp, rest = seg.partition(" ")
            if not rest:
                key, rest = "", seg
            if key:
                x += self.put(y, x, key, self.color(T.CYAN, bg, True), maxw=max(0, maxw - x)) + 1
            x += self.put(y, x, rest, self.color(T.MUTED, bg), maxw=max(0, maxw - x))
        return x

    def footer(self, hints, draw_rule=True):
        h, w = self.size()
        if h < 3:
            return
        if not self._draw_toast() and draw_rule:
            self.hline(h - 2, 0, w, "─", self.color(T.RULE_DIM))
        bar = self.color(T.SOFT, T.BAR)
        self.fill(h - 1, 0, w - 1, bar)
        right = self.status_fn() if self.status_fn else []
        kept = []
        rw = 0
        for item in reversed(right):
            iw = disp_w(_clean(item[0])) + (3 if kept else 0)
            if rw + iw <= max(0, w - 12):
                kept.append(item)
                rw += iw
        kept.reverse()
        self.keyhints(h - 1, 2, hints, max(2, w - rw - 4), bg=T.BAR)
        x = max(2, w - rw - 2)
        for idx, (t, c) in enumerate(kept):
            if idx:
                x += self.put(h - 1, x, " │ ", self.color(T.DIM, T.BAR))
            x += self.put(h - 1, x, t, self.color(c, T.BAR, True))

    # ---------- layout & data widgets ----------
    def section(self, y, x, label, width, color=None):
        color = T.VIOLET if color is None else color
        self.put(y, x, "◆ ", self.color(color, None, True))
        lw = self.put(y, x + 2, label, self.color(color, None, True), maxw=max(0, width - 3))
        rem = width - lw - 3
        if rem > 0:
            self.put(y, x + 3 + lw, "─" * rem, self.color(T.RULE))

    def card(self, y, x, hh, ww, title="", color=None, title_color=None, badge="", badge_color=None,
             fill=True, icon="◆"):
        """Rounded bordered card with coloured accent title and optional right-aligned top badge."""
        h, w = self.size()
        if hh < 2 or ww < 4 or y >= h or x >= w:
            return
        ww = min(ww, w - x)
        hh = min(hh, h - y)
        if ww < 4 or hh < 2:
            return
        color = T.RULE if color is None else color
        title_color = T.VIOLET if title_color is None else title_color
        badge_color = T.MUTED if badge_color is None else badge_color
        a = self.color(color)
        self.put(y, x, "╭" + "─" * (ww - 2) + "╮", a)
        t_used = 0
        if title:
            t_used = self.put(y, x + 2, f" {icon} ", self.color(title_color, None, True), maxw=max(0, ww - 4))
            t_used += self.put(y, x + 2 + t_used, f"{_clean(title)} ", self.color(T.LIGHT, None, True),
                               maxw=max(0, ww - 4 - t_used))
        if badge:
            b_str = f" {_clean(badge)} "
            bw = disp_w(b_str)
            bx = x + ww - bw - 2
            min_bx = x + (t_used + 3 if title else 2)
            if bx >= min_bx:
                self.put(y, bx, b_str, self.color(badge_color, None, True), maxw=bw)
        for i in range(1, hh - 1):
            self.put(y + i, x, "│", a)
            if fill:
                self.fill(y + i, x + 1, ww - 2)
            self.put(y + i, x + ww - 1, "│", a)
        self.put(y + hh - 1, x, "╰" + "─" * (ww - 2) + "╯", a)

    def scrollbar(self, y, x, height, top, view, total):
        """Draws a sleek vertical scrollbar when total > view."""
        if height <= 0 or total <= view or view <= 0:
            return
        thumb_h = max(1, int(round(height * view / total)))
        max_top = max(1, total - view)
        thumb_y = int(round((height - thumb_h) * min(max_top, max(0, top)) / max_top))
        for i in range(height):
            if thumb_y <= i < thumb_y + thumb_h:
                self.put(y + i, x, "┃", self.color(T.VIOLET))
            else:
                self.put(y + i, x, "│", self.color(T.RULE_DIM))

    def progress(self, y, x, width, frac, label="", bg=None, show_pct=False):
        """Multi-colour theme prism progress bar."""
        width = max(4, width)
        f = max(0.0, min(1.0, frac))
        full = width * f
        n = int(full)
        frac_idx = int((full - n) * 8)
        part = " ▏▎▍▌▋▊▉"[frac_idx]
        for j in range(width):
            if j < n:
                self.put(y, x + j, "█", self.color(T.grad(j, width), bg, auto_sel=False))
            elif j == n and part != " ":
                self.put(y, x + j, part, self.color(T.grad(j, width), bg, auto_sel=False))
            else:
                self.put(y, x + j, "░", self.color(T.RULE_DIM, bg))
        off = x + width + 1
        if show_pct:
            off += self.put(y, off, f"{int(round(f * 100)):3d}% ", self.color(T.LIGHT, bg, True))
        if label:
            self.put(y, off, label, self.color(T.SOFT, bg))

    def wave_bar(self, y, x, width, tick, label="", bg=None):
        """Smooth animated indeterminate prism scanning wave bar."""
        width = max(4, width)
        for j in range(width):
            ch, col = T.wave_cell(j, width, tick)
            self.put(y, x + j, ch, self.color(col, bg, auto_sel=False))
        if label:
            self.put(y, x + width + 2, label, self.color(T.MUTED, bg))

    def bar(self, y, x, width, frac, color, bg=None, gradient=False):
        """Horizontal sub-character bar (scores, metrics) with background preservation."""
        if width <= 0:
            return
        f = max(0.0, min(1.0, frac))
        full = width * f
        n = int(full)
        part = " ▏▎▍▌▋▊▉"[int((full - n) * 8)]
        if gradient:
            for j in range(width):
                if j < n:
                    self.put(y, x + j, "█", self.color(T.grad(j, width), bg, auto_sel=False))
                elif j == n and part != " ":
                    self.put(y, x + j, part, self.color(T.grad(j, width), bg, auto_sel=False))
                else:
                    self.put(y, x + j, "·", self.color(T.RULE_DIM, bg))
        else:
            bar_str = "█" * n + (part if n < width and part != " " else "")
            if bar_str:
                self.put(y, x, bar_str, self.color(color, bg, auto_sel=False))
            rest = width - len(bar_str)
            if rest > 0:
                self.put(y, x + len(bar_str), "·" * rest, self.color(T.RULE_DIM, bg))

    def heat_bar(self, y, x, width, frac, bg=None):
        """Utilization bar transitioning emerald → cyan → gold → coral along its filled span."""
        if width <= 0:
            return
        f = max(0.0, min(1.0, frac))
        full = width * f
        n = int(full)
        part = " ▏▎▍▌▋▊▉"[int((full - n) * 8)]
        for j in range(width):
            cell_frac = (j + 0.5) / max(1, width)
            col = T.mem_color(cell_frac)
            if j < n:
                self.put(y, x + j, "█", self.color(col, bg, auto_sel=False))
            elif j == n and part != " ":
                self.put(y, x + j, part, self.color(col, bg, auto_sel=False))
            else:
                self.put(y, x + j, "·", self.color(T.RULE_DIM, bg))

    def sparkline(self, y, x, values, color=None, bg=None):
        """Compact 1-row block sparkline `▁▂▃▄▅▆▇█` from normalized 0..1 floats."""
        if not values:
            return 0
        used = 0
        for v in values:
            f = max(0.0, min(1.0, float(v)))
            ch = SPARK[min(len(SPARK) - 1, int(f * (len(SPARK) - 1) + 0.5))]
            col = color if color is not None else T.mem_color(f)
            used += self.put(y, x + used, ch, self.color(col, bg, auto_sel=False))
        return used

    def stacked_bar(self, y, x, width, segments, bg=None):
        """Multi-segment horizontal bar. segments: [(weight, color), ...]."""
        total = sum(max(0.0, w_) for w_, _ in segments)
        if width <= 0 or total <= 0:
            self.put(y, x, "─" * max(0, width), self.color(T.RULE_DIM, bg))
            return
        pos = 0
        nz = [(wt, col) for wt, col in segments if wt > 0]
        for idx, (wt, col) in enumerate(nz):
            seg_w = (width - pos) if idx == len(nz) - 1 else int(round(width * wt / total))
            seg_w = max(1 if pos < width else 0, min(seg_w, width - pos))
            if seg_w > 0:
                self.put(y, x + pos, "█" * seg_w, self.color(col, bg, auto_sel=False))
                pos += seg_w
        if pos < width:
            self.put(y, x + pos, "─" * (width - pos), self.color(T.RULE_DIM, bg))

    def sel_bar(self, y, x, width, color=None):
        accent = T.PINK if color is None else color
        self.fill(y, x, width, self.color(T.LIGHT, T.SEL, auto_sel=False))
        self.put(y, x, "▌", self.color(accent, T.SEL, True, auto_sel=False))

    def shadow(self, y, x, hh, ww):
        """Subtle dark drop shadow along the right and bottom edge of a modal."""
        h, w = self.size()
        sh = self.color(T.INK, T.INK, auto_sel=False)
        for i in range(1, hh):
            if 0 <= y + i < h - 1 and x + ww < w:
                self.put(y + i, x + ww, " ", sh)
        if 0 <= y + hh < h - 1:
            for j in range(1, ww + 1):
                if x + j < w:
                    self.put(y + hh, x + j, " ", sh)

    def box(self, y, x, hh, ww, title="", color=None, shadow=True):
        h, w = self.size()
        if hh < 2 or ww < 4 or y >= h or x >= w:
            return
        ww = min(ww, w - x)
        hh = min(hh, h - y)
        if hh < 2 or ww < 4:
            return
        color = T.VIOLET if color is None else color
        if shadow:
            self.shadow(y, x, hh, ww)
        a = self.color(color)
        self.put(y, x, "╭" + "─" * (ww - 2) + "╮", a)
        if title:
            self.put(y, x + 2, f" ◆ {_clean(title)} ", self.color(T.LIGHT, None, True), maxw=max(0, ww - 4))
        for i in range(1, hh - 1):
            self.put(y + i, x, "│", a)
            self.fill(y + i, x + 1, ww - 2)
            self.put(y + i, x + ww - 1, "│", a)
        self.put(y + hh - 1, x, "╰" + "─" * (ww - 2) + "╯", a)

    def key(self, timeout=-1):
        self.scr.timeout(timeout)
        try:
            k = self.scr.get_wch()
        except curses.error:
            return None
        except KeyboardInterrupt:
            return SIGINT
        if k == curses.KEY_RESIZE:
            curses.update_lines_cols()
            return "RESIZE"
        return k

    def wait_key(self, timeout=-1):
        """Like key(), but wakes up early if an active toast is due to expire."""
        return self.key(self.toast_timeout_ms(timeout))

    # ---------- dialogs ----------
    @staticmethod
    def wrap(text, width):
        width = max(10, width)
        out = []
        for para in str(text).split("\n"):
            out += textwrap.wrap(para, width) or [""]
        return out

    def message(self, text, title="gbench-tui", color=None, wait=True):
        color = T.VIOLET if color is None else color
        while True:
            h, w = self.size()
            ww = min(w - 2, max(32, min(90, max((disp_w(_clean(line)) for line in str(text).split("\n")), default=20) + 8)))
            lines = self.wrap(text, max(12, ww - 6))[: max(1, h - 7)]
            hh = min(h - 2, len(lines) + 5)
            y, x = max(1, (h - hh) // 2), max(1, (w - ww) // 2)
            self.box(y, x, hh, ww, title, color)
            for i, line in enumerate(lines[: max(0, hh - 4)]):
                self.put(y + 2 + i, x + 3, line, self.color(T.SOFT if i > 0 else T.LIGHT), maxw=max(0, ww - 6))
            if wait:
                btn = " Enter ▸ ok "
                self.put(y + hh - 2, max(x + 2, x + ww - len(btn) - 2), btn,
                         self.color(T.INK, color, True, auto_sel=False), maxw=max(0, ww - 4))
                self.scr.refresh()
                k = self.key()
                if k == "RESIZE":
                    self.scr.erase()
                    continue
                if k in ("\n", "\r", " ", ESC, SIGINT, "q", curses.KEY_ENTER):
                    return
            else:
                self.scr.refresh()
                return

    def confirm(self, text, default=False, color=None) -> bool:
        color = T.YELLOW if color is None else color
        sel = 0 if default else 1
        while True:
            h, w = self.size()
            ww = min(w - 2, max(36, min(82, disp_w(_clean(text)) + 10)))
            lines = self.wrap(text, max(12, ww - 6))[: max(1, h - 8)]
            hh = min(h - 2, len(lines) + 5)
            y, x = max(1, (h - hh) // 2), max(1, (w - ww) // 2)
            self.box(y, x, hh, ww, "Confirm", color)
            for i, line in enumerate(lines[: max(0, hh - 4)]):
                self.put(y + 2 + i, x + 3, line, self.color(T.LIGHT), maxw=max(0, ww - 6))
            if ww >= 42:
                self.put(y + hh - 2, x + 3, "←→ choose · Enter · Esc", self.color(T.DIM), maxw=max(0, ww - 22))
            for j, lab in enumerate(("  Yes  ", "  No  ")):
                a = self.color(T.INK, color, True, auto_sel=False) if j == sel else self.color(T.MUTED, T.BAR_ALT)
                bx = max(x + 2, x + ww - 18 + j * 8)
                self.put(y + hh - 2, bx, lab, a, maxw=max(0, x + ww - 1 - bx))
            self.scr.refresh()
            k = self.key()
            if k == "RESIZE":
                self.scr.erase()
                continue
            if k in ("y", "Y"):
                return True
            if k in ("n", "N", ESC, SIGINT, "q"):
                return False
            if k in (curses.KEY_LEFT, curses.KEY_RIGHT, "\t", "h", "l"):
                sel = 1 - sel
            if k in ("\n", "\r", curses.KEY_ENTER):
                return sel == 0

    def edit(self, prompt, initial="", hint="", secret=False):
        """A one-line editor in a shadowed dialog. Returns the text, or None on Esc/Ctrl-C."""
        buf, pos = list(str(initial)), len(str(initial))
        with contextlib.suppress(curses.error):
            curses.curs_set(1)
        try:
            while True:
                h, w = self.size()
                ww = max(14, min(w - 2, 86))
                hh = min(h - 2, 7 if hint else 6)
                y, x = max(1, (h - hh) // 2), max(1, (w - ww) // 2)
                self.box(y, x, hh, ww, prompt, T.VIOLET)
                field_w = max(1, ww - 6)
                text = "".join(buf)
                shown = ("•" * len(text)) if secret else _clean(text)
                start = 0
                while start < pos and disp_w(shown[start:pos]) >= field_w:
                    start += 1
                vis_str, _ = trunc_w(shown[start:], field_w)
                self.put(y + 2, x + 2, "▌", self.color(T.PINK, T.SEL, True, auto_sel=False))
                self.fill(y + 2, x + 3, field_w, self.color(T.LIGHT, T.SEL, auto_sel=False))
                self.put(y + 2, x + 3, vis_str, self.color(T.LIGHT, T.SEL, True, auto_sel=False), maxw=field_w)
                if hint and hh >= 7:
                    self.put(y + 4, x + 3, hint, self.color(T.MUTED), maxw=field_w)
                hint_str = " Enter ok · ^U clear · Esc back "
                hx = max(x + 2, x + ww - len(hint_str) - 1)
                self.put(y + hh - 2, hx, hint_str, self.color(T.DIM), maxw=max(0, x + ww - 1 - hx))
                cursor_col = min(field_w - 1, disp_w(shown[start:pos]))
                with contextlib.suppress(curses.error):
                    self.scr.move(y + 2, x + 3 + cursor_col)
                self.scr.refresh()
                k = self.key()
                if k == "RESIZE":
                    self.scr.erase()
                    continue
                if k in ("\n", "\r", curses.KEY_ENTER):
                    return "".join(buf)
                if k in (ESC, SIGINT):
                    return None
                if k in (curses.KEY_BACKSPACE, "\x7f", "\b"):
                    if pos:
                        del buf[pos - 1]
                        pos -= 1
                elif k == curses.KEY_DC:
                    if pos < len(buf):
                        del buf[pos]
                elif k == curses.KEY_LEFT:
                    pos = max(0, pos - 1)
                elif k == curses.KEY_RIGHT:
                    pos = min(len(buf), pos + 1)
                elif k in (curses.KEY_HOME, "\x01"):
                    pos = 0
                elif k in (curses.KEY_END, "\x05"):
                    pos = len(buf)
                elif k == "\x15":
                    buf, pos = [], 0
                elif isinstance(k, str) and len(k) == 1 and k.isprintable():
                    buf.insert(pos, k)
                    pos += 1
        finally:
            with contextlib.suppress(curses.error):
                curses.curs_set(0)

    def choose(self, title, items, sel=0, hint="", on_move=None, on_redraw=None):
        """items: [(value, label, sublabel)] → value or None.

        Optional `on_move(value)` fires whenever selection changes (e.g. live theme preview) and `on_redraw()`
        repaints the background screen underneath the modal before drawing the dialog.
        """
        if not items:
            return None
        top = 0
        sel = max(0, min(sel, len(items) - 1))
        if on_move:
            on_move(items[sel][0])
        while True:
            if on_redraw:
                on_redraw()
            h, w = self.size()
            ww = min(w - 2, max(36, max(disp_w(_clean(i[1])) + disp_w(_clean(i[2] if len(i) > 2 else "")) for i in items) + 12))
            hh = max(5, min(h - 2, len(items) + 5))
            y, x = max(1, (h - hh) // 2), max(1, (w - ww) // 2)
            view = max(1, hh - 4)
            top = max(0, min(top, max(0, len(items) - view)))
            top = min(max(top, sel - view + 1), sel)
            self.box(y, x, hh, ww, title, T.VIOLET)
            for i in range(view):
                j = top + i
                if j >= len(items):
                    break
                it = items[j]
                if j == sel:
                    self.sel_bar(y + 1 + i, x + 1, ww - 3, T.PINK)
                bg = T.SEL if j == sel else None
                lw = self.put(y + 1 + i, x + 3, it[1], self.color(T.LIGHT if j == sel else T.SOFT, bg, j == sel), maxw=max(0, ww - 6))
                if len(it) > 2 and it[2]:
                    sub_x = x + 5 + lw
                    self.put(y + 1 + i, sub_x, it[2], self.color(T.MUTED, bg), maxw=max(0, x + ww - 2 - sub_x))
            self.scrollbar(y + 1, x + ww - 2, view, top, view, len(items))
            self.put(y + hh - 2, x + 3, hint or "↑↓ move · Enter choose · Esc back", self.color(T.DIM), maxw=max(0, ww - 6))
            self.scr.refresh()
            k = self.key()
            if k == "RESIZE":
                self.scr.erase()
                continue
            prev_sel = sel
            if k in (curses.KEY_UP, "k"):
                sel = (sel - 1) % len(items)
            elif k in (curses.KEY_DOWN, "j"):
                sel = (sel + 1) % len(items)
            elif k == curses.KEY_PPAGE:
                sel = max(0, sel - view)
            elif k == curses.KEY_NPAGE:
                sel = min(len(items) - 1, sel + view)
            elif k in ("\n", "\r", curses.KEY_ENTER):
                return items[sel][0]
            elif k in (ESC, SIGINT, "q"):
                return None
            if sel != prev_sel and on_move:
                on_move(items[sel][0])

    def busy(self, text, fn, *args, **kw):
        """Runs fn in a thread with an animated prism wave dialog and elapsed timer; returns its result."""
        box = {}

        def work():
            try:
                box["r"] = fn(*args, **kw)
            except Exception as e:                  # noqa: BLE001
                box["e"] = e
        t = threading.Thread(target=work, daemon=True)
        t0 = time.monotonic()
        t.start()
        i = 0
        while t.is_alive():
            if self.key(0) == "RESIZE":
                self.scr.erase()
            h, w = self.size()
            ww = max(18, min(w - 2, max(36, disp_w(_clean(text)) + 18)))
            hh = min(h - 2, 5)
            y, x = max(1, (h - hh) // 2), max(1, (w - ww) // 2)
            el = time.monotonic() - t0
            el_s = f"{el:4.1f}s" if el < 100 else dur(el)
            self.box(y, x, hh, ww, "Working", T.VIOLET)
            self.put(y + 1, x + 3, SPIN[i % len(SPIN)], self.color(T.PINK, None, True))
            self.put(y + 1, x + 5, text, self.color(T.LIGHT), maxw=max(0, ww - len(el_s) - 9))
            self.put(y + 1, max(x + 6, x + ww - len(el_s) - 3), el_s, self.color(T.DIM), maxw=len(el_s))
            if hh >= 5 and ww >= 14:
                self.wave_bar(y + 3, x + 3, max(4, ww - 6), i)
            self.scr.refresh()
            i += 1
            t.join(0.09)
        if "e" in box:
            raise box["e"]
        return box.get("r")

    # ---------- pager ----------
    def pager(self, title, lines, sub="", hints="↑↓ PgUp PgDn scroll · ←→ sideways · / search · g/G top/end · q back",
              follow=None, keys=None):
        """lines: list of str or [(text, attr)] segments. follow: callable returning fresh lines (live view)."""
        top, left, query = 0, 0, ""
        matches: list[int] = []
        cur_match = 0

        def plain(ln):
            if isinstance(ln, str):
                return ln
            return "".join(str(t) for t, _ in ln)

        def recompute_matches(reset_idx: bool = True):
            nonlocal matches, cur_match
            if not query:
                matches = []
                cur_match = 0
                return
            q = query.lower()
            matches = [idx for idx, ln in enumerate(lines) if q in plain(ln).lower()]
            if reset_idx or cur_match >= len(matches):
                cur_match = 0

        while True:
            if follow:
                lines = follow()
                if query:
                    recompute_matches(reset_idx=False)
            h, w = self.size()
            self.scr.erase()
            view = max(1, h - 6)
            max_top = max(0, len(lines) - view)
            top = max(0, min(top, max_top))
            max_line_w = max((disp_w(_clean(plain(ln))) for ln in lines[top: top + view]), default=0)
            left = max(0, min(left, max(0, max_line_w - max(10, w - 6))))
            pct = 100 if len(lines) <= view else min(100, int((top + view) * 100 / max(1, len(lines))))
            rbadge = None
            if query:
                rbadge = (f"/{query} ({len(matches)} matches)", T.YELLOW if matches else T.RED)
            elif len(lines) > view:
                rbadge = (f"line {top + 1}–{min(len(lines), top + view)}/{len(lines)} · {pct}%", T.MUTED)
            self.header(title, sub or f"{len(lines)} lines", right_badge=rbadge)
            for i in range(view):
                j = top + i
                if j >= len(lines):
                    break
                ln = lines[j]
                is_hit = query and (j in matches)
                if isinstance(ln, str):
                    ln = [(ln, self.color(T.YELLOW if is_hit else T.SOFT, T.BAR_ALT if is_hit else None, bool(is_hit)))]
                elif is_hit:
                    self.put(3 + i, 0, "▸", self.color(T.YELLOW, None, True))
                self.segs(3 + i, 2, ln, maxw=max(0, w - 4), skip=left)
            self.scrollbar(3, w - 1, view, top, view, len(lines))
            self.footer(hints)
            self.scr.refresh()
            k = self.wait_key(1000 if follow else -1)
            if k is None or k == "RESIZE":
                continue
            if query and matches and k == "n":
                cur_match = (cur_match + 1) % len(matches)
                top = max(0, matches[cur_match] - 2)
                continue
            if query and matches and k == "N":
                cur_match = (cur_match - 1) % len(matches)
                top = max(0, matches[cur_match] - 2)
                continue
            if keys and k in keys:
                r = keys[k]()
                if r == "quit":
                    return
                if isinstance(r, tuple) and r[0] == "replace":
                    title, lines, sub = r[1], r[2], r[3]
                    top, left = 0, 0
                    recompute_matches(reset_idx=True)
                continue
            if k in ("q", ESC, SIGINT):
                return
            if k == "/":
                q = self.edit("Search in view", query, hint="empty clears the search")
                if q is not None:
                    query = q.strip()
                    recompute_matches(reset_idx=True)
                    if matches:
                        top = max(0, matches[0] - 2)
            elif k == curses.KEY_LEFT:
                left = max(0, left - 8)
            elif k == curses.KEY_RIGHT:
                left += 8
            elif k in (curses.KEY_DOWN, "j"):
                top = min(max_top, top + 1)
            elif k in (curses.KEY_UP, "k"):
                top = max(0, top - 1)
            elif k in (curses.KEY_NPAGE, " "):
                top = min(max_top, top + max(1, view - 1))
            elif k == curses.KEY_PPAGE:
                top = max(0, top - max(1, view - 1))
            elif k in ("g", curses.KEY_HOME):
                top = 0
            elif k in ("G", curses.KEY_END):
                top = max_top

    def markdown(self, text, width):
        """Light Markdown → styled segment lines (headings, bullets, code blocks, tables)."""
        out, code = [], False
        for raw in text.splitlines():
            if raw.strip().startswith("```"):
                code = not code
                if code:
                    out.append([("╭─ code", self.color(T.DIM))])
                else:
                    out.append([("╰─", self.color(T.DIM))])
                continue
            if code:
                out.append([("│ ", self.color(T.RULE)), (raw, self.color(T.CYAN))])
                continue
            m = re.match(r"^(#{1,4})\s+(.*)", raw)
            if m:
                lvl = len(m.group(1))
                if out and out[-1] != [("", 0)]:
                    out.append([("", 0)])
                col = (T.LIGHT, T.PALE, T.CYAN, T.SOFT)[lvl - 1]
                bar = "▍ " if lvl <= 2 else "▪ "
                out.append([(bar, self.color(T.PINK if lvl == 1 else T.VIOLET if lvl == 2 else T.CYAN, None, True)),
                            (m.group(2), self.color(col, None, True))])
                continue
            if raw.startswith("|"):
                if re.match(r"^\|\s*[:\-| ]+\|$", raw):
                    out.append([(raw.replace("-", "─").replace("|", "┼"), self.color(T.RULE))])
                else:
                    out.append([(raw, self.color(T.SOFT))])
                continue
            line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", raw)
            bullet = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", line)
            ind, body, mark = ("", line, "")
            if bullet:
                ind, mark, body = bullet.group(1), bullet.group(2), bullet.group(3)
            wrapped = textwrap.wrap(body, max(20, width - len(ind) - 4)) or [""]
            for i, part in enumerate(wrapped):
                segs = []
                if mark:
                    segs.append((ind + ("• " if i == 0 else "  "), self.color(T.PINK, None, True)))
                for tok in re.split(r"(`[^`]+`|\*\*[^*]+\*\*)", part):
                    if tok.startswith("`") and tok.endswith("`") and len(tok) > 1:
                        segs.append((tok[1:-1], self.color(T.CYAN)))
                    elif tok.startswith("**") and tok.endswith("**") and len(tok) > 3:
                        segs.append((tok[2:-2], self.color(T.LIGHT, None, True)))
                    elif tok:
                        segs.append((tok, self.color(T.SOFT)))
                out.append(segs)
        return out


def ago(ts: float) -> str:
    d = max(0.0, time.time() - ts)
    if d < 90:
        return f"{int(d)}s"
    if d < 5400:
        return f"{int(d // 60)}m"
    if d < 86400 * 2:
        return f"{d / 3600:.1f}h"
    return f"{int(d // 86400)}d"


def dur(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"
