"""Curses primitives: colours, text, boxes, dialogs, one-line editor, pickers and a pager. Standard library only."""
from __future__ import annotations

import curses
import os
import re
import textwrap
import time

from . import theme as T

os.environ.setdefault("ESCDELAY", "25")
ESC = "\x1b"
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _basic(c: int) -> int:
    """xterm-256 → the 8 basic colours (for terminals without 256 colours)."""
    table = {T.BLUE: curses.COLOR_BLUE, T.DEEP: curses.COLOR_BLUE, T.CYAN: curses.COLOR_CYAN, T.VIOLET: curses.COLOR_CYAN,
             T.PURPLE: curses.COLOR_CYAN, T.PINK: curses.COLOR_WHITE, T.GREEN: curses.COLOR_GREEN,
             T.YELLOW: curses.COLOR_YELLOW, T.RED: curses.COLOR_RED}
    if c in table:
        return table[c]
    if c >= 232:
        return curses.COLOR_BLACK if c < 244 else curses.COLOR_WHITE
    return curses.COLOR_WHITE


class UI:
    def __init__(self, scr):
        self.scr = scr
        curses.curs_set(0)
        scr.keypad(True)
        curses.start_color()
        try:
            curses.use_default_colors()
        except curses.error:
            pass
        self._pairs: dict = {}
        self.status_fn = None        # () -> list of (text, colour) for the right side of the footer

    # ---------- primitives ----------
    def size(self):
        return self.scr.getmaxyx()

    def color(self, fg=None, bg=None, bold=False, dim=False) -> int:
        key = (fg, bg)
        if key not in self._pairs:
            n = len(self._pairs) + 1
            if n >= curses.COLOR_PAIRS:
                return curses.A_BOLD if bold else 0
            conv = (lambda c: -1 if c is None else c) if curses.COLORS >= 256 else (lambda c: -1 if c is None else _basic(c))
            try:
                curses.init_pair(n, conv(fg), conv(bg))
            except curses.error:
                return curses.A_BOLD if bold else 0
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
        text = str(text).replace("\t", "    ")
        text = text[:room]
        try:
            self.scr.addstr(y, x, text, attr)
        except curses.error:
            pass
        return len(text)

    def segs(self, y, x, segs, maxw=None, skip=0):
        """Draws [(text, attr), …] from column x, skipping the first `skip` characters (horizontal scroll)."""
        h, w = self.size()
        limit = (w - x) if maxw is None else min(maxw, w - x)
        used = 0
        for text, attr in segs:
            text = str(text)
            if skip:
                if skip >= len(text):
                    skip -= len(text)
                    continue
                text, skip = text[skip:], 0
            if used >= limit:
                break
            used += self.put(y, x + used, text, attr, maxw=limit - used)
        return used

    def fill(self, y, x, width, attr=0):
        self.put(y, x, " " * max(0, width), attr)

    def hline(self, y, x=0, width=None, ch="─", attr=None):
        h, w = self.size()
        self.put(y, x, ch * ((width or w - x) - (1 if y == h - 1 else 0)), self.color(T.RULE) if attr is None else attr)

    def header(self, title, sub=""):
        h, w = self.size()
        bar = self.color(T.SOFT, T.BAR)
        self.fill(0, 0, w, bar)
        x = 0
        for i, ch in enumerate(" g·bench "):            # DeepMind blues: deep → sky, white letters
            self.put(0, x + i, ch, self.color(T.LIGHT, T.GRADIENT[min(i, 4)], True))
        x += 9
        self.put(0, x, "▌", self.color(T.GRADIENT[4], T.BAR))
        self.put(0, x + 2, title, self.color(T.LIGHT, T.BAR, True))
        right = "tui · Google DeepMind gbench "
        if w - len(right) > x + len(title) + 6:
            self.put(0, w - len(right), right, self.color(T.MUTED, T.BAR))
        if sub:
            self.put(1, 2, sub, self.color(T.MUTED))

    def keyhints(self, y, x, text, maxw):
        for i, seg in enumerate(text.split(" · ")):
            if x >= maxw:
                return x
            if i:
                x += self.put(y, x, " · ", self.color(T.RULE), maxw=maxw - x)
            key, _sp, rest = seg.partition(" ")
            if not rest:
                key, rest = "", seg
            if key:
                x += self.put(y, x, key, self.color(T.BLUE, None, True), maxw=max(0, maxw - x)) + 1
            x += self.put(y, x, rest, self.color(T.MUTED), maxw=max(0, maxw - x))
        return x

    def footer(self, hints):
        h, w = self.size()
        self.fill(h - 1, 0, w - 1)
        right = self.status_fn() if self.status_fn else []
        rw = sum(len(t) for t, _ in right) + 2 * len(right)
        self.keyhints(h - 1, 1, hints, w - rw - 3)
        x = w - rw - 1
        for t, c in right:
            x += self.put(h - 1, x, t, self.color(c, None, True)) + 2

    def toast(self, text, color=T.YELLOW):
        h, w = self.size()
        self.fill(h - 2, 0, w - 1)
        self.put(h - 2, 2, text, self.color(color, None, True))

    def section(self, y, x, label, width, color=T.BLUE):
        self.put(y, x, "◆ ", self.color(color))
        self.put(y, x + 2, label, self.color(color, None, True))
        self.put(y, x + 3 + len(label), "╌" * max(0, width - len(label) - 4), self.color(T.RULE))

    def progress(self, y, x, width, frac, label=""):
        n = int(round(width * max(0.0, min(1.0, frac))))
        for j in range(width):
            if j < n:
                self.put(y, x + j, "█", self.color(T.grad(j, width)))
            else:
                self.put(y, x + j, "░", self.color(T.RULE))
        if label:
            self.put(y, x + width + 1, label, self.color(T.SOFT))

    def bar(self, y, x, width, frac, color):
        """A thin horizontal bar (scores)."""
        full = width * max(0.0, min(1.0, frac))
        n = int(full)
        part = " ▏▎▍▌▋▊▉"[int((full - n) * 8)]
        self.put(y, x, "█" * n + (part if n < width and part != " " else ""), self.color(color))
        rest = width - n - (1 if n < width and part != " " else 0)
        if rest > 0:
            self.put(y, x + width - rest, "·" * rest, self.color(T.RULE))

    def sel_bar(self, y, x, width, color=T.BLUE):
        self.fill(y, x, width, self.color(T.LIGHT, T.SEL))
        self.put(y, x, "▌", self.color(color, T.SEL))

    def box(self, y, x, hh, ww, title="", color=T.BLUE):
        a = self.color(color)
        self.put(y, x, "╭" + "─" * (ww - 2) + "╮", a)
        if title:
            self.put(y, x + 2, f" {title} ", self.color(color, None, True))
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
            return "q"
        if k == curses.KEY_RESIZE:
            curses.update_lines_cols()
            return "RESIZE"
        return k

    # ---------- dialogs ----------
    @staticmethod
    def wrap(text, width):
        out = []
        for para in str(text).split("\n"):
            out += textwrap.wrap(para, width) or [""]
        return out

    def message(self, text, title="gbench-tui", color=T.BLUE, wait=True):
        h, w = self.size()
        ww = min(w - 4, max(40, min(90, max(len(line) for line in str(text).split("\n")) + 6)))
        lines = self.wrap(text, ww - 4)[: h - 6]
        hh = len(lines) + 4
        y, x = max(1, (h - hh) // 2), max(1, (w - ww) // 2)
        self.box(y, x, hh, ww, title, color)
        for i, line in enumerate(lines):
            self.put(y + 1 + i, x + 2, line, self.color(T.LIGHT))
        if wait:
            self.put(y + hh - 2, x + ww - 14, " Enter ▸ ok ", self.color(color, None, True))
            self.scr.refresh()
            while self.key() not in ("\n", "\r", " ", ESC, "q", curses.KEY_ENTER):
                pass

    def confirm(self, text, default=False, color=T.YELLOW) -> bool:
        h, w = self.size()
        ww = min(w - 4, max(44, min(80, len(text) + 8)))
        lines = self.wrap(text, ww - 4)
        hh = len(lines) + 4
        y, x = max(1, (h - hh) // 2), max(1, (w - ww) // 2)
        sel = 0 if default else 1
        while True:
            self.box(y, x, hh, ww, "confirm", color)
            for i, line in enumerate(lines):
                self.put(y + 1 + i, x + 2, line, self.color(T.LIGHT))
            for j, lab in enumerate((" Yes ", " No ")):
                a = self.color(T.INK, color, True) if j == sel else self.color(T.MUTED)
                self.put(y + hh - 2, x + ww - 16 + j * 7, lab, a)
            self.scr.refresh()
            k = self.key()
            if k in ("y", "Y", "s", "S"):
                return True
            if k in ("n", "N", ESC, "q"):
                return False
            if k in (curses.KEY_LEFT, curses.KEY_RIGHT, "\t", "h", "l"):
                sel = 1 - sel
            if k in ("\n", "\r", curses.KEY_ENTER):
                return sel == 0

    def edit(self, prompt, initial="", hint="", secret=False):
        """A one-line editor in a dialog. Returns the text, or None on Esc."""
        h, w = self.size()
        ww = min(w - 4, 84)
        y, x = max(1, h // 2 - 3), max(1, (w - ww) // 2)
        buf, pos = list(str(initial)), len(str(initial))
        curses.curs_set(1)
        try:
            while True:
                self.box(y, x, 6 if hint else 5, ww, prompt, T.BLUE)
                field_w = ww - 4
                text = "".join(buf)
                shown = ("•" * len(text)) if secret else text
                start = max(0, pos - field_w + 1)
                self.fill(y + 2, x + 2, field_w, self.color(T.LIGHT, T.SEL))
                self.put(y + 2, x + 2, shown[start:start + field_w], self.color(T.LIGHT, T.SEL))
                if hint:
                    self.put(y + 3, x + 2, hint[: field_w], self.color(T.MUTED))
                self.put(y + (4 if hint else 3), x + ww - 30, " Enter ok · Esc cancel ", self.color(T.MUTED))
                try:
                    self.scr.move(y + 2, x + 2 + pos - start)
                except curses.error:
                    pass
                self.scr.refresh()
                k = self.key()
                if k in ("\n", "\r", curses.KEY_ENTER):
                    return "".join(buf)
                if k == ESC:
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
                elif isinstance(k, str) and k.isprintable():
                    buf.insert(pos, k)
                    pos += 1
        finally:
            curses.curs_set(0)

    def choose(self, title, items, sel=0, hint=""):
        """items: [(value, label, sublabel)] → value or None."""
        if not items:
            return None
        h, w = self.size()
        ww = min(w - 4, max(40, max(len(i[1]) + len(i[2] if len(i) > 2 else "") for i in items) + 10))
        hh = min(h - 4, len(items) + 4)
        y, x = max(1, (h - hh) // 2), max(1, (w - ww) // 2)
        top = 0
        sel = max(0, min(sel, len(items) - 1))
        while True:
            view = hh - 4
            top = min(max(top, sel - view + 1), sel)
            self.box(y, x, hh, ww, title, T.VIOLET)
            for i in range(view):
                j = top + i
                if j >= len(items):
                    break
                it = items[j]
                if j == sel:
                    self.sel_bar(y + 1 + i, x + 1, ww - 2, T.VIOLET)
                self.put(y + 1 + i, x + 3, it[1], self.color(T.LIGHT, T.SEL if j == sel else None, j == sel), maxw=ww - 5)
                if len(it) > 2 and it[2]:
                    self.put(y + 1 + i, x + 4 + len(it[1]), it[2], self.color(T.MUTED, T.SEL if j == sel else None),
                             maxw=max(0, ww - 6 - len(it[1])))
            self.put(y + hh - 2, x + 2, hint or "↑↓ move · Enter choose · Esc back", self.color(T.MUTED), maxw=ww - 4)
            self.scr.refresh()
            k = self.key()
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
            elif k in (ESC, "q"):
                return None

    def busy(self, text, fn, *args, **kw):
        """Runs fn in a thread with a spinner dialog; returns its result."""
        import threading
        box = {}

        def work():
            try:
                box["r"] = fn(*args, **kw)
            except Exception as e:                  # noqa: BLE001
                box["e"] = e
        t = threading.Thread(target=work, daemon=True)
        t.start()
        i = 0
        h, w = self.size()
        ww = min(w - 4, len(text) + 10)
        while t.is_alive():
            y, x = h // 2 - 1, max(1, (w - ww) // 2)
            self.box(y, x, 3, ww, "", T.BLUE)
            self.put(y + 1, x + 2, SPIN[i % len(SPIN)], self.color(T.CYAN, None, True))
            self.put(y + 1, x + 4, text, self.color(T.LIGHT), maxw=ww - 6)
            self.scr.refresh()
            i += 1
            t.join(0.1)
        if "e" in box:
            raise box["e"]
        return box.get("r")

    # ---------- pager ----------
    def pager(self, title, lines, sub="", hints="↑↓ PgUp PgDn scroll · ←→ sideways · g/G top/end · q back", follow=None,
              keys=None):
        """lines: list of str or [(text, attr)] segments. follow: callable returning fresh lines (live view)."""
        top, left = 0, 0
        while True:
            if follow:
                lines = follow()
            h, w = self.size()
            self.scr.erase()
            self.header(title, sub)
            view = h - 4
            top = max(0, min(top, len(lines) - view))
            for i in range(view):
                j = top + i
                if j >= len(lines):
                    break
                ln = lines[j]
                if isinstance(ln, str):
                    ln = [(ln, self.color(T.SOFT))]
                self.segs(2 + i, 1, ln, maxw=w - 2, skip=left)
            if len(lines) > view:
                pct = (top + view) / len(lines)
                self.put(h - 2, w - 12, f" {min(100, int(pct * 100)):3d}% ", self.color(T.MUTED))
            self.footer(hints)
            self.scr.refresh()
            k = self.key(1000 if follow else -1)
            if k is None or k == "RESIZE":
                continue
            if keys and k in keys:
                r = keys[k]()
                if r == "quit":
                    return
                continue
            if k in ("q", ESC):
                return
            if k == curses.KEY_LEFT:
                if left == 0:
                    return
                left = max(0, left - 8)
            elif k == curses.KEY_RIGHT:
                left += 8
            elif k in (curses.KEY_DOWN, "j"):
                top += 1
            elif k in (curses.KEY_UP, "k"):
                top = max(0, top - 1)
            elif k in (curses.KEY_NPAGE, " "):
                top += view - 1
            elif k == curses.KEY_PPAGE:
                top = max(0, top - view + 1)
            elif k in ("g", curses.KEY_HOME):
                top = 0
            elif k in ("G", curses.KEY_END):
                top = max(0, len(lines) - view)

    def markdown(self, text, width):
        """Light Markdown → segment lines (headings, bullets, code, tables kept as is)."""
        out, code = [], False
        for raw in text.splitlines():
            if raw.strip().startswith("```"):
                code = not code
                continue
            if code:
                out.append([("  " + raw, self.color(T.CYAN))])
                continue
            m = re.match(r"^(#{1,4})\s+(.*)", raw)
            if m:
                lvl = len(m.group(1))
                if out and out[-1] != [("", 0)]:
                    out.append([("", 0)])
                col = (T.BLUE, T.VIOLET, T.CYAN, T.SOFT)[lvl - 1]
                out.append([("▍" if lvl <= 2 else " ", self.color(col)), (m.group(2), self.color(col, None, True))])
                continue
            if raw.startswith("|"):
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
                    segs.append((ind + ("• " if i == 0 else "  "), self.color(T.BLUE)))
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
    d = time.time() - ts
    if d < 90:
        return f"{int(d)}s"
    if d < 5400:
        return f"{int(d // 60)}m"
    if d < 86400 * 2:
        return f"{d / 3600:.1f}h"
    return f"{int(d // 86400)}d"


def dur(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"
