"""gbench-tui: thin App entry point composing the modular `gbtui.screens` package."""
from __future__ import annotations

import contextlib
import curses
import locale
import os

from . import config, theme as T
from .screens import (
    CommonMixin,
    EndpointsMixin,
    Field,
    HarnessLensMixin,
    HomeMixin,
    LeaderboardMixin,
    ResultsMixin,
    RunMixin,
    Status,
    SuitesMixin,
    SystemMixin,
    _adapter_color,
    _clamp_top,
    _friendly_probe_err,
    _need_color,
    _quant_color,
    _short,
    _skip_heading,
)
from .ui import UI

__all__ = [
    "App",
    "Field",
    "Status",
    "run_tui",
    "_adapter_color",
    "_clamp_top",
    "_friendly_probe_err",
    "_need_color",
    "_quant_color",
    "_short",
    "_skip_heading",
]


class App(
    HomeMixin,
    EndpointsMixin,
    RunMixin,
    ResultsMixin,
    LeaderboardMixin,
    HarnessLensMixin,
    SuitesMixin,
    SystemMixin,
    CommonMixin,
):
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


def _utf8():
    with contextlib.suppress(locale.Error):
        locale.setlocale(locale.LC_ALL, "")


def run_tui():
    _utf8()
    os.environ.setdefault("TERM", "xterm-256color")

    def main(scr):
        App(scr).main()
    curses.wrapper(main)
