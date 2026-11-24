"""gbench-tui: the screens. Home · Endpoints · Run · Live run · Suites · Results · Leaderboard · Export · Sandbox · Settings."""
from __future__ import annotations

import contextlib
import curses
import locale
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__, catalog, config, export, leaderboard, results as R, runner, sandbox, servers, sysmem
from . import theme as T
from .ui import ESC, SPIN, UI, dur

ENTER = ("\n", "\r", curses.KEY_ENTER)


@dataclass
class Field:
    key: str
    label: str
    kind: str = "text"            # text | int | bool | choice | action | info | picker
    choices: list = field(default_factory=list)
    help: str = ""


class Status:
    """Host / endpoint / sandbox state, refreshed in a background thread so the screens never block on a probe."""

    def __init__(self, app):
        self.app = app
        self.mem_avail = 0
        self.mem_total = sysmem.total_bytes()
        self.swap = 0
        self.ep_ok: dict = {}
        self.sbx = "checking…"
        self.sbx_list: list = []
        self._stop = False
        self._kick = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def kick(self):
        self._kick.set()

    def _loop(self):
        n = 0
        while not self._stop:
            self.mem_avail = sysmem.available_bytes()
            self.swap = sysmem.swap_used_bytes()
            ep = self.app.s.endpoint()
            if ep:
                self.ep_ok[ep.name] = servers.probe(ep, timeout=1.5)
            if n % 6 == 0 or self._kick.is_set():
                self.sbx = sandbox.status() or "ready"
                self.sbx_list = [s for s in sandbox.sandboxes() if str(s.get("name", "")).startswith("gbtui-")]
            self._kick.clear()
            n += 1
            self._kick.wait(5)



def _short(bench: str) -> str:
    """A column header that fits: 'decode tok/s (in512)' -> 'tok/s·512', 'TTFT (in512)' -> 'TTFT·512'."""
    import re
    m = re.match(r"(.+?) \(in(\d+)\)$", bench)
    name, n = (m.group(1), m.group(2)) if m else (bench, "")
    name = {"decode tok/s": "tok/s", "output tok/s": "out t/s"}.get(name, name)
    return f"{name}·{n}" if n else name

class App:
    def __init__(self, scr):
        self.ui = UI(scr)
        self.scr = scr
        self.s = config.load()
        self.st = Status(self)
        self.ui.status_fn = self.footer_status
        self.marked: set = set()
        self.tick = 0

    # ------------------------------------------------------------------ chrome
    def footer_status(self):
        out = []
        L = runner.LIVE
        if L.active:
            d, t = L.progress
            prog = f" {d}/{t}" if t else ""
            out.append((f"{SPIN[self.tick % len(SPIN)]} {L.current or L.phase}{prog}", T.CYAN))
        elif L.phase in ("done", "failed", "cancelled"):
            out.append(({"done": "✔ run finished", "failed": "✗ run failed", "cancelled": "■ cancelled"}[L.phase],
                        T.GREEN if L.phase == "done" else T.RED))
        if self.st.mem_avail:
            free = self.st.mem_avail / sysmem.GB
            out.append((f"RAM {free:.0f}G free", T.GREEN if free > self.s.min_free_gb * 2 else T.YELLOW if free > self.s.min_free_gb else T.RED))
        return out

    def save(self):
        config.save(self.s)

    # ------------------------------------------------------------------ home
    MENU = [("1", "Endpoints", "models and OpenAI-compatible servers to evaluate; start a GGUF with llama-server"),
            ("2", "Run", "Golden Set, eval suites or device performance, inside a Docker Sandbox"),
            ("3", "Live run", "follow the run in progress"),
            ("4", "Suites", "gbench's 90 eval suites by pillar and the Golden Set, with their docs"),
            ("5", "Results", "every run: scores, samples, failures"),
            ("6", "Leaderboard", "best of every model, ranked by gScore"),
            ("7", "Export", "runs as a dataset (JSONL / Parquet + dataset card)"),
            ("8", "Sandbox", "Docker Sandboxes template, live sandboxes, memory caps"),
            ("9", "Settings", "memory guard, results folder, defaults")]

    def main(self):
        sel = 0
        actions = [self.screen_endpoints, self.screen_run, self.screen_live, self.screen_suites, self.screen_results,
                   self.screen_leaderboard, self.screen_export, self.screen_sandbox, self.screen_settings]
        while True:
            self.draw_home(sel)
            k = self.ui.key(500)
            self.tick += 1
            if k is None or k == "RESIZE":
                continue
            if k in (curses.KEY_UP, "k"):
                sel = (sel - 1) % len(self.MENU)
            elif k in (curses.KEY_DOWN, "j"):
                sel = (sel + 1) % len(self.MENU)
            elif isinstance(k, str) and k in "123456789":
                sel = int(k) - 1
                actions[sel]()
            elif k in ENTER:
                actions[sel]()
            elif k in ("q", ESC):
                if self.quit():
                    return

    def draw_home(self, sel):
        ui = self.ui
        h, w = ui.size()
        self.scr.erase()
        ui.header("home")
        y = 2
        wm = T.WORDMARK
        if w >= len(wm[0]) + 6 and h >= 28:
            x0 = 3
            for i, row in enumerate(wm):
                for j, ch in enumerate(row):
                    if ch == " ":
                        continue
                    shade = ch in "╗╔╝╚═║"
                    ui.put(y + i, x0 + j, ch, ui.color(T.grad(j, len(row)), None, not shade, shade))
            ui.put(y + len(wm), x0 + 1, "a terminal UI for Google DeepMind's gbench", ui.color(T.MUTED))
            ui.put(y + len(wm), x0 + 44, f"v{__version__}", ui.color(T.RULE))
            y += len(wm) + 2
        else:
            ui.put(y, 3, "g·bench tui", ui.color(T.BLUE, None, True))
            y += 2
        left_w = min(54, w - 4)
        for i, (num, name, desc) in enumerate(self.MENU):
            if i == sel:
                ui.sel_bar(y + i, 2, left_w)
            live = (num == "3" and runner.LIVE.active)
            ui.put(y + i, 4, num, ui.color(T.grad(i, len(self.MENU)), T.SEL if i == sel else None, True))
            ui.put(y + i, 7, name + (" ●" if live else ""), ui.color(T.LIGHT if i == sel else T.SOFT, T.SEL if i == sel else None, i == sel))
        for i, line in enumerate(UI.wrap(self.MENU[sel][2], left_w - 4)[:2]):
            ui.put(y + len(self.MENU) + 1 + i, 4, line, ui.color(T.MUTED))
        # status panel
        px = left_w + 6
        pw = w - px - 2
        if pw >= 34:
            self.draw_status(2 if h < 28 or w < len(wm[0]) + 6 else y, px, pw)
        elif h > y + len(self.MENU) + 12:
            self.draw_status(y + len(self.MENU) + 3, 3, w - 6)
        ui.footer("↑↓ move · Enter open · 1-9 jump · q quit")
        self.scr.refresh()

    def draw_status(self, y, x, w):
        ui = self.ui
        ui.section(y, x, "Status", w, T.VIOLET)
        y += 1
        ep = self.s.endpoint()

        def row(label, text, color):
            nonlocal y
            ui.put(y, x + 1, f"{label:<10}", ui.color(T.MUTED))
            ui.put(y, x + 12, text, ui.color(color), maxw=w - 13)
            y += 1
        if ep:
            pr = self.st.ep_ok.get(ep.name)
            state = ("● up" if pr and pr["ok"] else "○ down" if pr else "… probing")
            row("endpoint", f"{ep.name}  {state}", T.GREEN if pr and pr["ok"] else T.YELLOW)
            row("", ep.base_url, T.SOFT)
            if ep.gguf:
                pid = servers.server_pid(ep)
                row("llama", f"pid {pid} · {sysmem.fmt(sysmem.rss_bytes(pid))} RSS" if pid else "not started by gbench-tui",
                    T.SOFT if pid else T.MUTED)
        else:
            row("endpoint", "none: add one (1)", T.YELLOW)
        sb = self.st.sbx
        row("sandbox", "● " + sb if sb == "ready" else "⚠ " + sb, T.GREEN if sb == "ready" else T.YELLOW)
        if self.st.sbx_list:
            row("", f"{len(self.st.sbx_list)} gbtui sandbox(es) alive", T.CYAN)
        tot = self.st.mem_total or 1
        av = self.st.mem_avail
        row("memory", f"{sysmem.fmt(av)} free of {sysmem.fmt(tot)}" + (f" · swap {sysmem.fmt(self.st.swap)}" if self.st.swap else ""),
            T.SOFT)
        ui.bar(y, x + 12, max(10, min(30, w - 14)), 1 - av / tot, T.VIOLET)
        y += 2
        L = runner.LIVE
        if L.active or L.phase != "idle":
            ui.section(y, x, "Run", w, T.CYAN)
            y += 1
            row("run", L.run_id[:w - 14], T.SOFT)
            d, t = L.progress
            row("phase", f"{L.phase}" + (f" · {L.current}" if L.current else ""), T.CYAN)
            if t:
                ui.progress(y, x + 12, max(10, min(30, w - 22)), d / t, f"{d}/{t}")
                y += 1
        else:
            runs = R.list_runs(self.s.results_path)[:3] if self.tick % 20 == 0 or not hasattr(self, "_recent") else self._recent
            self._recent = runs
            if runs:
                ui.section(y, x, "Recent runs", w, T.CYAN)
                y += 1
                for r in runs:
                    ui.put(y, x + 1, f"{r.when[5:]} {r.model[:16]:<16} {r.tier:<6}", ui.color(T.SOFT), maxw=w - 2)
                    ui.put(y, x + 37, r.headline, ui.color(T.CYAN), maxw=max(0, w - 38))
                    y += 1

    def quit(self) -> bool:
        if runner.LIVE.active:
            if not self.ui.confirm("A run is in progress. Quit and cancel it? (its sandbox is removed)"):
                return False
            runner.cancel()
            self.ui.busy("cancelling the run and removing its sandbox…", lambda: [time.sleep(0.5) for _ in range(40)
                                                                                    if runner.LIVE.active])
        started = servers.started_servers(self.s.endpoints)
        if started and self.s.stop_server_on_exit:
            for ep in started:
                servers.stop_server(ep)
        self.st._stop = True
        return True

    # ------------------------------------------------------------------ endpoints
    def screen_endpoints(self):
        ui, sel = self.ui, 0
        while True:
            eps = self.s.endpoints
            h, w = ui.size()
            self.scr.erase()
            ui.header("Endpoints", "OpenAI-compatible /v1 servers gbench reaches from its sandbox (--remote-endpoint)")
            y = 3
            ui.put(y, 4, f"{'':2}{'name':<20}{'model (--models)':<24}{'tokenizer':<26}url", ui.color(T.MUTED))
            y += 1
            for i, e in enumerate(eps):
                active = e.name == self.s.active
                if i == sel:
                    ui.sel_bar(y + i, 2, w - 4)
                bg = T.SEL if i == sel else None
                pr = self.st.ep_ok.get(e.name)
                dot = "●" if pr and pr["ok"] else "○"
                ui.put(y + i, 3, dot, ui.color(T.GREEN if pr and pr["ok"] else T.MUTED, bg, True))
                ui.put(y + i, 4, ("★ " if active else "  "), ui.color(T.YELLOW, bg, True))
                ui.put(y + i, 6, f"{e.name[:19]:<20}", ui.color(T.LIGHT, bg, active))
                ui.put(y + i, 26, f"{(e.model or '?')[:23]:<24}", ui.color(T.SOFT, bg))
                ui.put(y + i, 50, f"{(e.tokenizer or '—')[:25]:<26}", ui.color(T.SOFT, bg))
                ui.put(y + i, 76, e.base_url, ui.color(T.CYAN, bg), maxw=max(0, w - 84))
                if e.gguf:
                    pid = servers.server_pid(e)
                    ui.put(y + i, w - 8, "gguf▸" if not pid else "gguf●", ui.color(T.GREEN if pid else T.MUTED, bg))
            if eps:
                e = eps[min(sel, len(eps) - 1)]
                dy = y + len(eps) + 2
                ui.section(dy, 2, e.name, w - 4, T.VIOLET)
                info = [("url", e.base_url), ("model", e.model or "(the only model the endpoint serves)"),
                        ("tokenizer", e.tokenizer or "(none: gbench needs one for golden chat-template cases and perf)"),
                        ("api key", f"${e.api_key_env}" if e.api_key_env else "none"),
                        ("gguf", f"{e.gguf}  · ctx {e.ctx} · {e.parallel} slots · ~{sysmem.fmt(servers.estimate_bytes(e))}"
                         if e.gguf else "— (a server you run yourself)")]
                pr = self.st.ep_ok.get(e.name)
                if pr:
                    info.append(("probe", ("serves " + ", ".join(pr["models"][:5])) if pr["ok"] else pr["error"]))
                for i, (k, v) in enumerate(info):
                    ui.put(dy + 1 + i, 4, f"{k:<10}", ui.color(T.MUTED))
                    ui.put(dy + 1 + i, 15, v, ui.color(T.SOFT), maxw=w - 17)
            ui.footer("Enter edit · Space use · a add · g add GGUF · s start/stop server · t test · d delete · q back")
            self.scr.refresh()
            k = ui.key(1000)
            self.tick += 1
            if k is None or k == "RESIZE":
                continue
            if k in ("q", ESC):
                return
            if k in (curses.KEY_UP, "k") and eps:
                sel = (sel - 1) % len(eps)
            elif k in (curses.KEY_DOWN, "j") and eps:
                sel = (sel + 1) % len(eps)
            elif k == "a":
                self.edit_endpoint(None)
                sel = len(self.s.endpoints) - 1
            elif k == "g":
                self.add_gguf()
                sel = len(self.s.endpoints) - 1
            elif not eps:
                continue
            elif k in ENTER:
                self.edit_endpoint(eps[sel])
            elif k == " ":
                self.s.active = eps[sel].name
                self.save()
                self.st.kick()
            elif k == "d":
                if ui.confirm(f"Delete endpoint {eps[sel].name}? (its results stay)"):
                    servers.stop_server(eps[sel])
                    del eps[sel]
                    sel = max(0, sel - 1)
                    if self.s.active not in [e.name for e in eps]:
                        self.s.active = eps[0].name if eps else ""
                    self.save()
            elif k == "s":
                e = eps[sel]
                if not e.gguf:
                    ui.message("This endpoint has no GGUF: start its server yourself (vLLM, Ollama, LM Studio…).")
                elif servers.server_pid(e):
                    servers.stop_server(e)
                    ui.toast(f"llama-server for {e.name} stopped", T.GREEN)
                else:
                    why = ui.busy(f"starting llama-server for {e.name}…", servers.start_server, e, self.s.min_free_gb)
                    ui.message(why or f"llama-server ready on {e.base_url}", color=T.RED if why else T.GREEN)
                self.st.kick()
            elif k == "t":
                self.test_endpoint(eps[sel])

    def test_endpoint(self, e):
        ui = self.ui
        pr = ui.busy(f"GET {e.base_url}/models…", servers.probe, e)
        if not pr["ok"]:
            ui.message(f"✗ {e.base_url} not reachable: {pr['error']}" + ("\n\nStart its llama-server with s." if e.gguf else ""),
                       color=T.RED)
            return
        r = ui.busy("one chat completion…", servers.chat_once, e)
        txt = (f"✔ serves: {', '.join(pr['models'])} ({pr['latency_ms']} ms)\n\n"
               + (f"✔ chat: “{r['text'][:200]}” in {r['seconds']} s" if r["ok"] else f"✗ chat: {r['text']}"))
        if e.model and e.model not in pr["models"]:
            txt += f"\n\n⚠ '{e.model}' is not in the served ids: gbench matches by name, or set the model to one of them."
        ui.message(txt, color=T.GREEN if r["ok"] else T.YELLOW)

    def add_gguf(self):
        ui = self.ui
        found = sorted({str(p) for d in (Path.home(), Path.home() / "Downloads", Path.home() / "models")
                        if d.is_dir() for p in d.glob("*.gguf")})
        items = [(p, Path(p).name, f"{os.path.getsize(p) / 1e9:.1f} GB") for p in found] + [("", "other path…", "")]
        path = ui.choose("GGUF file", items)
        if path is None:
            return
        if path == "":
            path = ui.edit("path to the .gguf", str(Path.home()))
            if not path:
                return
        stem = Path(path).stem
        base = stem.split("-Q")[0].split("-F16")[0].split("-BF16")[0]
        name = base.lower()
        port = 8080 + len([e for e in self.s.endpoints if e.gguf])
        tok = ui.edit("tokenizer (HF id, for gbench's --tokenizer)", f"LiquidAI/{base}" if base.startswith("LFM") else "",
                      hint="the original model's repo, e.g. LiquidAI/LFM3-1.7B; empty = none")
        ep = config.Endpoint(name=name, base_url=f"http://127.0.0.1:{port}/v1", model=name, tokenizer=tok or "", gguf=path)
        if any(e.name == ep.name for e in self.s.endpoints):
            ep.name += f"-{port}"
        self.s.endpoints.append(ep)
        self.s.active = self.s.active or ep.name
        self.save()

    def edit_endpoint(self, e):
        new = e is None
        e = e or config.Endpoint(name=f"endpoint-{len(self.s.endpoints) + 1}")
        fields = [Field("name", "Name"), Field("base_url", "Base URL (/v1)", help="what gbench gets as --remote-endpoint"),
                  Field("model", "Model id (--models)", help="the id the endpoint serves; empty = its only model"),
                  Field("tokenizer", "Tokenizer (--tokenizer)", help="HF id of the original model, e.g. LiquidAI/LFM3-1.7B"),
                  Field("api_key_env", "API key env var", help="name of the environment variable with the key (never the key)"),
                  Field("gguf", "GGUF file (optional)", help="gbench-tui starts llama-server for it on the URL's port"),
                  Field("ctx", "llama-server ctx", "int", help="total context; small = less memory"),
                  Field("parallel", "llama-server slots", "int", help="concurrent requests (each gets ctx/slots)"),
                  Field("extra_args", "llama-server extra args"), Field("notes", "Notes")]
        vals = {f.key: getattr(e, f.key) for f in fields}
        if self.form(("New endpoint" if new else f"Endpoint · {e.name}"), fields, vals, done_label="Save"):
            old = e.name
            for k, v in vals.items():
                setattr(e, k, v)
            if new:
                self.s.endpoints.append(e)
            if self.s.active in ("", old):
                self.s.active = e.name
            self.save()
            self.st.kick()

    # ------------------------------------------------------------------ generic form
    def form(self, title, fields, vals, done_label="Done", sub="", on_change=None, extra=None) -> bool:
        ui, sel = self.ui, 0
        make = fields if callable(fields) else (lambda: fields)
        while True:
            fields = make() + [Field("__done", f"▶ {done_label}", "action")]
            sel = min(sel, len(fields) - 1)
            while fields[sel].kind == "info":          # info rows are not selectable
                sel = (sel + 1) % len(fields)
            h, w = ui.size()
            self.scr.erase()
            ui.header(title, sub)
            y = 3
            lw = max(len(f.label) for f in fields if f.kind != "info") + 4
            for i, f in enumerate(fields):
                if f.kind == "info":
                    ui.put(y + i, 4, f.label, ui.color(T.VIOLET, None, True), maxw=w - 6)
                    continue
                if i == sel:
                    ui.sel_bar(y + i, 2, w - 4)
                bg = T.SEL if i == sel else None
                if f.kind == "action":
                    ui.put(y + i, 4, f.label, ui.color(T.CYAN if f.key == "__done" else T.BLUE, bg, True))
                    continue
                ui.put(y + i, 4, f.label, ui.color(T.SOFT, bg))
                v = vals.get(f.key)
                if f.kind == "bool":
                    txt, col = ("● yes" if v else "○ no"), (T.GREEN if v else T.MUTED)
                elif f.kind == "choice":
                    lab = next((c[1] for c in f.choices if c[0] == v), str(v))
                    txt, col = f"‹ {lab} ›", T.CYAN
                elif f.kind == "picker":
                    txt, col = (", ".join(v) if v else "(all)") if isinstance(v, list) else str(v), T.CYAN
                else:
                    txt, col = (str(v) if v not in (None, "") else "—"), T.LIGHT
                ui.put(y + i, 4 + lw, txt, ui.color(col, bg, i == sel), maxw=w - lw - 8)
            f = fields[sel]
            if f.help:
                ui.put(h - 3, 2, "▎", ui.color(T.VIOLET))
                ui.put(h - 3, 4, f.help, ui.color(T.MUTED), maxw=w - 6)
            if extra:
                extra(y + len(fields) + 1)
            ui.footer("↑↓ move · Enter edit/toggle · ←→ change choice · Esc cancel")
            self.scr.refresh()
            k = ui.key(1000)
            self.tick += 1
            if k is None or k == "RESIZE":
                continue
            if k == ESC or k == "q":
                return False
            if k in (curses.KEY_UP, "k"):
                sel = (sel - 1) % len(fields)
                while fields[sel].kind == "info":
                    sel = (sel - 1) % len(fields)
                continue
            if k in (curses.KEY_DOWN, "j", "\t"):
                sel = (sel + 1) % len(fields)
                while fields[sel].kind == "info":
                    sel = (sel + 1) % len(fields)
                continue
            changed = False
            if f.kind == "choice" and k in (curses.KEY_LEFT, curses.KEY_RIGHT, " "):
                keys = [c[0] for c in f.choices]
                i = keys.index(vals[f.key]) if vals[f.key] in keys else 0
                vals[f.key] = keys[(i + (-1 if k == curses.KEY_LEFT else 1)) % len(keys)]
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
                    v = ui.choose(f.label, [(c[0], c[1], c[2] if len(c) > 2 else "") for c in f.choices],
                                  sel=[c[0] for c in f.choices].index(vals[f.key]) if vals[f.key] in [c[0] for c in f.choices] else 0)
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
                                v = int(v)
                            except ValueError:
                                ui.toast("a whole number, please", T.RED)
                                continue
                        vals[f.key] = v.strip() if isinstance(v, str) else v
                        changed = True
            if changed and on_change:
                on_change(vals)

    # ------------------------------------------------------------------ run
    def screen_run(self):
        ui = self.ui
        if runner.LIVE.active:
            return self.screen_live()
        if not self.s.endpoints:
            ui.message("Add an endpoint first (1 · Endpoints).")
            return
        d = self.s.run
        vals = {"endpoint": self.s.active or self.s.endpoints[0].name, "tier": d.tier, "evals": list(d.evals),
                "golden_tasks": list(d.golden_tasks), "eval_limit": d.eval_limit, "max_output_tokens": d.max_output_tokens,
                "thinking": d.thinking, "temperature": d.temperature, "concurrency": d.concurrency,
                "suite_timeout": d.suite_timeout, "perf_mode": d.perf_mode,
                "memory": self.s.sandbox_memory_gb}
        tiers = [(k, k, v) for k, v in runner.TIERS.items()]

        def fields_for(v):
            fl = [Field("endpoint", "Endpoint", "choice", [(e.name, e.name, e.base_url) for e in self.s.endpoints],
                        help="the server gbench measures (--remote-endpoint, --models, --tokenizer)"),
                  Field("tier", "What to run", "choice", tiers, help=runner.TIERS.get(v["tier"], ""))]
            if v["tier"] == "golden":
                fl.append(Field("golden_tasks", "Golden Set tasks", "picker", [self.pick_golden],
                                help="empty = all 16 (vision / audio cases are N/A for a text-only model)"))
            if v["tier"] in ("evals", "preflight"):
                fl.append(Field("evals", "Eval suites", "picker", [self.pick_suites],
                                help="★ = runs on a laptop with no extra setup; see 4 · Suites for each one's needs"))
            if v["tier"] == "evals":
                fl += [Field("eval_limit", "Samples per suite", "int", help="--eval-limit: a stratified subset (0 = all)"),
                       Field("concurrency", "Concurrent requests", "int", help="--eval-concurrency (≤ llama-server slots)"),
                       Field("suite_timeout", "Suite timeout (s)", "int", help="--suite-timeout: TIMEOUT instead of hanging")]
            if v["tier"] in ("evals", "preflight", "golden"):
                fl += [Field("max_output_tokens", "Max output tokens", "int", help="--max-output-tokens: truncation counts as a failure"),
                       Field("thinking", "Thinking / reasoning", "bool", help="--eval-thinking"),
                       Field("temperature", "Temperature", help="empty = gbench's default (0.0, 1.0 with thinking)")]
            if v["tier"] == "perf":
                fl.append(Field("perf_mode", "Mode", "choice", [("personal", "personal", "single stream: decode tok/s, TTFT"),
                                                               ("quick", "quick", "serving preset quick, text only")]))
            fl += [Field("memory", "Sandbox RAM (GB)", "int", help="the microVM's cap; gbench + its datasets fit in 4–6 GB"),
                   Field("cmd", "", "info")]
            return fl

        def preview(y):
            ep = self.s.endpoint(vals["endpoint"])
            spec = self.spec_from(vals, ep)
            h, w = ui.size()
            ui.section(y, 2, "gbench command (inside the sandbox)", w - 4, T.VIOLET)
            for i, line in enumerate(UI.wrap(runner.command_preview(spec), w - 8)[:4]):
                ui.put(y + 1 + i, 4, line, ui.color(T.CYAN))
            up = (self.st.ep_ok.get(ep.name) or {}).get("ok")
            need = vals["memory"] * sysmem.GB + (0 if not ep.gguf or up or servers.server_pid(ep) else servers.estimate_bytes(ep))
            why = sysmem.check(need, self.s.min_free_gb)
            ui.put(y + 6, 4, ("✔ memory: " + sysmem.fmt(need) + " needed, " + sysmem.fmt(self.st.mem_avail) + " free")
                   if not why else "⚠ " + why, ui.color(T.GREEN if not why else T.RED), maxw=w - 6)
            if vals["tier"] in ("evals", "preflight"):
                heavy = [f"{n} ({', '.join(catalog.suite(n)['needs'])})" for n in vals["evals"]
                         if catalog.suite(n) and catalog.suite(n)["needs"]]
                if heavy:
                    ui.put(y + 7, 4, "⚠ needs more than an endpoint: " + "; ".join(heavy) + " — try a Preflight first",
                           ui.color(T.YELLOW), maxw=w - 6)
            if not self.st.sbx == "ready":
                ui.put(y + 8, 4, "⚠ sandbox: " + self.st.sbx, ui.color(T.YELLOW), maxw=w - 6)

        if not self.form("Run gbench", lambda: fields_for(vals), vals, done_label="Start run", extra=preview,
                         sub="every run gets its own Docker Sandbox, removed when it ends"):
            return
        ep = self.s.endpoint(vals["endpoint"])
        self.s.active = ep.name
        self.s.sandbox_memory_gb = max(2, int(vals["memory"]))
        for k in ("tier", "evals", "golden_tasks", "eval_limit", "max_output_tokens", "thinking", "temperature",
                  "concurrency", "suite_timeout", "perf_mode"):
            setattr(self.s.run, k, vals[k])
        self.save()
        why = runner.start(self.spec_from(vals, ep), self.s)
        if why:
            ui.message(why, color=T.RED)
            return
        self.screen_live()

    @staticmethod
    def spec_from(v, ep):
        return runner.RunSpec(endpoint=ep, tier=v["tier"], evals=list(v["evals"]), golden_tasks=list(v["golden_tasks"]),
                              eval_limit=int(v["eval_limit"] or 0), max_output_tokens=int(v["max_output_tokens"] or 4096),
                              thinking=bool(v["thinking"]), temperature=str(v["temperature"] or ""),
                              concurrency=int(v["concurrency"] or 1), suite_timeout=int(v["suite_timeout"] or 0),
                              perf_mode=v["perf_mode"])

    def pick_golden(self, current):
        gold = catalog.load()["golden"]
        items = [(g["id"], g.get("category", ""), g.get("description", "")) for g in gold]
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
            h, w = ui.size()
            self.scr.erase()
            ui.header(title, f"{len(chosen)} chosen" + (f" · filter: {flt}" if flt else ""))
            view = h - 5
            sel = max(0, min(sel, len(vis) - 1))
            while vis and vis[sel][0] is None and sel < len(vis) - 1:
                sel += 1
            top = min(max(top, sel - view + 1), sel)
            for i in range(view):
                j = top + i
                if j >= len(vis):
                    break
                v, lab, note = vis[j]
                y = 3 + i
                if v is None:
                    ui.put(y, 3, lab, ui.color(T.VIOLET, None, True))
                    continue
                on = v in chosen
                if j == sel:
                    ui.sel_bar(y, 2, w - 4)
                bg = T.SEL if j == sel else None
                ui.put(y, 4, "◉" if on else "○", ui.color(T.CYAN if on else T.MUTED, bg, True))
                ui.put(y, 6, lab[:28], ui.color(T.LIGHT if on else T.SOFT, bg, on))
                ui.put(y, 35, note, ui.color(T.MUTED, bg), maxw=w - 37)
            ui.footer("Space toggle · Enter done · / filter · a all/none visible · i info · Esc cancel")
            self.scr.refresh()
            k = ui.key()
            if k in (ESC,):
                return None
            if k in ENTER:
                return [it[0] for it in items if it[0] in chosen]
            if k in (curses.KEY_UP, "k"):
                sel -= 1
                while sel > 0 and vis[sel][0] is None:
                    sel -= 1
            elif k in (curses.KEY_DOWN, "j"):
                sel = min(len(vis) - 1, sel + 1)
            elif k == curses.KEY_NPAGE:
                sel = min(len(vis) - 1, sel + view)
            elif k == curses.KEY_PPAGE:
                sel = max(0, sel - view)
            elif k == " " and vis and vis[sel][0] is not None:
                chosen ^= {vis[sel][0]}
                sel = min(len(vis) - 1, sel + 1)
            elif k == "a":
                vals = {it[0] for it in vis if it[0] is not None}
                chosen = (chosen - vals) if vals <= chosen else (chosen | vals)
            elif k == "/":
                f = ui.edit("filter", flt)
                flt = (f or "").lower().strip()
                sel = top = 0
            elif k == "i" and vis and vis[sel][0] is not None:
                self.show_doc(vis[sel][0])

    # ------------------------------------------------------------------ live
    def screen_live(self):
        ui = self.ui
        L = runner.LIVE
        if L.phase == "idle":
            ui.message("No run in progress. Start one in 2 · Run.")
            return
        follow = True
        top = 0
        while True:
            h, w = ui.size()
            self.scr.erase()
            sub = (f"{L.spec.endpoint.name} · {L.spec.tier}" if L.spec else "")
            ui.header("Live run", sub)
            y = 3
            el = (L.finished or time.time()) - L.started if L.started else 0
            col = {"done": T.GREEN, "failed": T.RED, "cancelled": T.YELLOW}.get(L.phase, T.CYAN)
            ui.put(y, 3, (SPIN[self.tick % len(SPIN)] + " " if L.active else "") + L.phase.upper(), ui.color(col, None, True))
            ui.put(y, 18, f"{L.run_id}", ui.color(T.SOFT))
            ui.put(y, w - 14, f"⏱ {dur(el)}", ui.color(T.MUTED))
            y += 1
            if L.sandbox:
                ui.put(y, 3, f"sandbox {L.sandbox}" + (" (removed)" if not L.active else ""), ui.color(T.MUTED))
            y += 1
            d, t = L.progress
            if t:
                ui.progress(y, 3, max(10, w - 30), d / t, f"{d}/{t} {L.current}"[:24])
            elif L.current:
                ui.put(y, 3, L.current, ui.color(T.CYAN))
            y += 2
            ui.section(y, 2, "gbench output", w - 4, T.VIOLET)
            y += 1
            view = h - y - 2
            lines = L.lines
            if follow:
                top = max(0, len(lines) - view)
            for i in range(view):
                j = top + i
                if j >= len(lines):
                    break
                ln = lines[j]
                c = (T.RED if ln.startswith("✗") or "ERROR" in ln or "Traceback" in ln else
                     T.GREEN if ln.startswith("✔") else T.CYAN if ln.startswith("$") or ln.startswith("▸") else
                     T.YELLOW if "WARNING" in ln or ln.startswith("⚠") else T.SOFT)
                ui.put(y + i, 3, ln, ui.color(c), maxw=w - 5)
            hint = ("c cancel · b back (keeps running) · ↑↓ scroll · f follow" if L.active
                    else "r results · b back · ↑↓ scroll")
            ui.footer(hint)
            self.scr.refresh()
            k = ui.key(700)
            self.tick += 1
            if k is None or k == "RESIZE":
                continue
            if k in ("b", "q", ESC):
                return
            if k == "c" and L.active:
                if ui.confirm("Cancel the run? gbench stops and its sandbox is removed."):
                    runner.cancel()
            elif k in (curses.KEY_UP, "k"):
                follow, top = False, max(0, top - 1)
            elif k in (curses.KEY_DOWN, "j"):
                top += 1
            elif k == curses.KEY_PPAGE:
                follow, top = False, max(0, top - view)
            elif k == curses.KEY_NPAGE:
                top += view
            elif k in ("f", "G"):
                follow = True
            elif k == "r" and not L.active and L.result_dir:
                run = R.load_run(Path(L.result_dir))
                if run:
                    self.run_detail(run)

    # ------------------------------------------------------------------ suites
    def screen_suites(self):
        cat = catalog.load()
        items = [(None, f"Golden Set · {len(cat['golden'])} deterministic capability checks (--golden-only)", "")]
        items += [("golden:" + g["id"], "  " + g["id"], f" {g.get('category', '')} · {g.get('description', '')}")
                  for g in cat["golden"]]
        for p in cat["pillars"] + ["Other"]:
            group = [s for s in cat["suites"] if s["pillar"] == p]
            if not group:
                continue
            items.append((None, f"{p} · {len(group)} suites", ""))
            for s in sorted(group, key=lambda s: s["name"]):
                items.append((s["name"], ("★ " if s["light"] else "  ") + s["name"],
                              (" [" + ", ".join(s["needs"]) + "] " if s["needs"] else " ") + s["title"] + " · " + s["description"]))
        ui, sel, top = self.ui, 1, 0
        while True:
            h, w = ui.size()
            self.scr.erase()
            ui.header("Suites", f"from {cat['source']} · ★ runs on a laptop with no extra setup · Enter: the suite's guide")
            view = h - 5
            top = min(max(top, sel - view + 1), sel)
            for i in range(view):
                j = top + i
                if j >= len(items):
                    break
                v, lab, note = items[j]
                y = 3 + i
                if v is None:
                    ui.put(y, 2, lab, ui.color(T.grad(j, len(items)), None, True))
                    continue
                if j == sel:
                    ui.sel_bar(y, 2, w - 4)
                bg = T.SEL if j == sel else None
                ui.put(y, 4, lab[:29], ui.color(T.LIGHT if j == sel else T.SOFT, bg, j == sel))
                ni = note.find("]")
                nx = 34
                if note.startswith(" [") and ni > 0:
                    ui.put(y, nx, note[: ni + 1], ui.color(T.YELLOW, bg), maxw=w - nx - 2)
                    ui.put(y, nx + ni + 1, note[ni + 1:], ui.color(T.MUTED, bg), maxw=max(0, w - nx - 2 - ni - 1))
                else:
                    ui.put(y, nx, note, ui.color(T.MUTED, bg), maxw=w - nx - 2)
            ui.footer("↑↓ move · Enter guide · Space add to the next run · q back")
            self.scr.refresh()
            k = ui.key()
            if k in ("q", ESC):
                return
            if k in (curses.KEY_UP, "k"):
                sel = max(1, sel - 1)
                if items[sel][0] is None:
                    sel = max(1, sel - 1)
            elif k in (curses.KEY_DOWN, "j"):
                sel = min(len(items) - 1, sel + 1)
                if items[sel][0] is None:
                    sel = min(len(items) - 1, sel + 1)
            elif k == curses.KEY_NPAGE:
                sel = min(len(items) - 1, sel + view)
            elif k == curses.KEY_PPAGE:
                sel = max(1, sel - view)
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
                    ev.remove(v) if v in ev else ev.append(v)
                    self.s.run.tier = "evals"
                    self.save()
                    ui.toast(f"next run's suites: {', '.join(ev) or '—'}", T.CYAN)
                    ui.key(900)

    def show_doc(self, name):
        s = catalog.suite(name) or {}
        w = self.ui.size()[1]
        head = [[("needs ", self.ui.color(T.MUTED)), (", ".join(s.get("needs") or ["nothing beyond the endpoint"]),
                                                     self.ui.color(T.YELLOW if s.get("needs") else T.GREEN))],
                [("options ", self.ui.color(T.MUTED)), (s.get("options", ""), self.ui.color(T.CYAN))], [("", 0)]]
        self.ui.pager(f"Suite · {name}", head + self.ui.markdown(catalog.doc(name), w - 4), sub=s.get("title", ""))

    def show_golden(self, tid):
        import json
        d = catalog.golden_task(tid) or {}
        w = self.ui.size()[1]
        md = f"# {tid}\n\n**category** {d.get('category', '')}\n\n{d.get('description', '')}\n\n## Conversation\n\n"
        for m in (d.get("conversation") or {}).get("messages", []):
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(p.get("text", f"[{p.get('type')}]") for p in c if isinstance(p, dict))
            md += f"- **{m.get('role')}**: {c}\n"
        md += "\n## Golden truth\n\n```\n" + json.dumps(d.get("golden_truth", {}), indent=2)[:3000] + "\n```\n"
        self.ui.pager(f"Golden Set · {tid}", self.ui.markdown(md, w - 4))

    # ------------------------------------------------------------------ results
    def screen_results(self):
        ui, sel = self.ui, 0
        runs = ui.busy("reading results…", R.list_runs, self.s.results_path)
        while True:
            h, w = ui.size()
            self.scr.erase()
            ui.header("Results", f"{len(runs)} runs in {self.s.results_path}")
            if not runs:
                ui.put(4, 4, "No runs yet: start one in 2 · Run.", ui.color(T.MUTED))
            y = 3
            ui.put(y, 4, f"  {'date':<17}{'model':<18}{'tier':<8}{'status':<10}{'time':<8}headline", ui.color(T.MUTED))
            view = h - 6
            top = max(0, sel - view + 1)
            for i, r in enumerate(runs[top:top + view]):
                j = top + i
                yy = y + 1 + i
                if j == sel:
                    ui.sel_bar(yy, 2, w - 4)
                bg = T.SEL if j == sel else None
                mark = "◉ " if r.id in self.marked else "  "
                ui.put(yy, 4, mark, ui.color(T.CYAN, bg, True))
                scol = {"ok": T.GREEN, "failed": T.RED, "cancelled": T.YELLOW}.get(r.status, T.MUTED)
                ui.put(yy, 6, f"{r.when:<17}{r.model[:17]:<18}{r.tier:<8}", ui.color(T.SOFT, bg))
                ui.put(yy, 49, f"{r.status:<10}", ui.color(scol, bg))
                ui.put(yy, 59, f"{dur(r.duration) if r.duration else '–':<8}", ui.color(T.MUTED, bg))
                ui.put(yy, 67, r.headline, ui.color(T.CYAN, bg), maxw=max(0, w - 70))
            ui.footer("Enter open · m mark · x export marked · l log · D delete · q back")
            self.scr.refresh()
            k = ui.key()
            if k in ("q", ESC):
                return
            if not runs:
                continue
            if k in (curses.KEY_UP, "k"):
                sel = (sel - 1) % len(runs)
            elif k in (curses.KEY_DOWN, "j"):
                sel = (sel + 1) % len(runs)
            elif k in ENTER:
                self.run_detail(runs[sel])
            elif k == "m":
                self.marked ^= {runs[sel].id}
            elif k == "l":
                self.show_log(runs[sel])
            elif k == "x":
                chosen = [r for r in runs if r.id in self.marked] or [runs[sel]]
                self.do_export(chosen)
            elif k == "D":
                if ui.confirm(f"Delete run {runs[sel].id} and its files?"):
                    shutil.rmtree(runs[sel].path, ignore_errors=True)
                    del runs[sel]
                    sel = max(0, sel - 1)
                    leaderboard.write(self.s)

    def show_log(self, run):
        p = run.path / "gbench.log"
        try:
            text = p.read_text(errors="replace")
        except OSError:
            text = "(no gbench.log in this run)"
        self.ui.pager(f"Log · {run.id}", text.splitlines(), hints="↑↓ scroll · G end · q back")

    def run_detail(self, run):
        ui, sel = self.ui, 0
        while True:
            h, w = ui.size()
            self.scr.erase()
            ui.header(f"Run · {run.id}", f"{run.model} · {run.tier} · {run.status} · {run.when}"
                      + (f" · {dur(run.duration)}" if run.duration else ""))
            y = 3
            spec = run.meta.get("spec") or {}
            info = [("endpoint", f"{run.endpoint.get('base_url', '?')}  model {run.endpoint.get('model', '?')}  "
                                 f"tokenizer {run.endpoint.get('tokenizer') or '—'}"),
                    ("settings", f"limit {spec.get('eval_limit')} · max tokens {spec.get('max_output_tokens')} · thinking "
                                 f"{'on' if spec.get('thinking') else 'off'} · temp {spec.get('temperature') or 'default'}"),
                    ("sandbox", f"{(run.meta.get('sandbox') or {}).get('template', '?')} · "
                                f"{(run.meta.get('sandbox') or {}).get('memory_gb', '?')} GB · gbench {run.meta.get('gbench_commit', '?')}")]
            if run.meta.get("error"):
                info.append(("error", run.meta["error"]))
            for k_, v in info:
                ui.put(y, 4, f"{k_:<10}", ui.color(T.MUTED))
                ui.put(y, 15, v, ui.color(T.RED if k_ == "error" else T.SOFT), maxw=w - 17)
                y += 1
            y += 1
            ui.section(y, 2, "Scores", w - 4, T.VIOLET)
            y += 1
            sc = run.scores
            for i, s in enumerate(sc):
                yy = y + i
                if yy >= h - 3:
                    break
                if i == sel:
                    ui.sel_bar(yy, 2, w - 4)
                bg = T.SEL if i == sel else None
                ui.put(yy, 4, f"{s.kind:<7}", ui.color(T.MUTED, bg))
                ui.put(yy, 12, f"{s.name[:26]:<27}", ui.color(T.LIGHT, bg, True))
                if s.value is None:
                    ui.put(yy, 40, f"{s.status or 'no score'}", ui.color(T.RED, bg))
                elif s.unit == "%":
                    ui.bar(yy, 40, 20, s.value / 100, T.score_color(s.value))
                    ui.put(yy, 61, f"{s.value:5.1f}%", ui.color(T.score_color(s.value), bg, True))
                    ui.put(yy, 69, f"{s.correct if s.correct is not None else '?'}/{s.n}", ui.color(T.SOFT, bg))
                else:
                    ui.put(yy, 40, f"{s.value:.2f} {s.unit}", ui.color(T.CYAN, bg, True))
                ui.put(yy, 80, s.detail or ("" if s.comparable is None else ("comparable" if s.comparable else "not leaderboard-comparable")),
                       ui.color(T.MUTED, bg), maxw=max(0, w - 82))
            if not sc:
                ui.put(y, 4, "No scores in this run (see its log: l).", ui.color(T.MUTED))
            ui.footer("Enter samples · f failures only · l log · x export · q back")
            self.scr.refresh()
            k = ui.key()
            if k in ("q", ESC):
                return
            if k in (curses.KEY_UP, "k") and sc:
                sel = (sel - 1) % len(sc)
            elif k in (curses.KEY_DOWN, "j") and sc:
                sel = (sel + 1) % len(sc)
            elif k in ENTER and sc:
                self.samples(run, sc[sel], False)
            elif k == "f" and sc:
                self.samples(run, sc[sel], True)
            elif k == "l":
                self.show_log(run)
            elif k == "x":
                self.do_export([run])

    def samples(self, run, score, failures):
        ui = self.ui
        its = R.items(run, score)
        if failures:
            its = [i for i in its if i["correct"] is False]
        if not its:
            ui.message("No per-sample records for this score" + (" (or no failures)." if failures else "."))
            return
        sel = 0
        while True:
            h, w = ui.size()
            self.scr.erase()
            ok = sum(1 for i in its if i["correct"])
            na = sum(1 for i in its if i["correct"] is None)
            ui.header(f"Samples · {score.name}", f"{len(its)} samples · {ok} correct · {len(its) - ok - na} not"
                      + (" · failures only" if failures else ""))
            view = h - 5
            top = max(0, sel - view + 1)
            for n, it in enumerate(its[top:top + view]):
                j = top + n
                y = 3 + n
                if j == sel:
                    ui.sel_bar(y, 2, w - 4)
                bg = T.SEL if j == sel else None
                mark = "–" if it["correct"] is None else ("✔" if it["correct"] else "✗")
                ui.put(y, 4, mark, ui.color(T.MUTED if it["correct"] is None else (T.GREEN if it["correct"] else T.RED), bg, True))
                sid = str(it["id"])
                sid = sid[7:19] if sid.startswith("sha256:") else sid[:22]
                ui.put(y, 6, f"{sid:<23}", ui.color(T.LIGHT, bg))
                ui.put(y, 30, f"{str(it.get('category') or it.get('status') or '')[:12]:<13}", ui.color(T.VIOLET, bg))
                # the question (last user turn) and, after it, gbench's verdict detail
                q = (it["prompt"] or "").rsplit("[user]", 1)[-1].strip().replace("\n", " ")
                det = (it["detail"] or "").replace("\n", " ")
                det = "" if det in ("ok", it.get("status")) else det
                ui.put(y, 44, q, ui.color(T.SOFT, bg), maxw=max(0, w - 46 - (min(len(det), 40) + 3 if det else 0)))
                if det:
                    ui.put(y, max(44, w - 2 - min(len(det), 40)), det[:40], ui.color(T.MUTED, bg))
            ui.footer("Enter open · ↑↓ move · q back")
            self.scr.refresh()
            k = ui.key()
            if k in ("q", ESC):
                return
            if k in (curses.KEY_UP, "k"):
                sel = (sel - 1) % len(its)
            elif k in (curses.KEY_DOWN, "j"):
                sel = (sel + 1) % len(its)
            elif k == curses.KEY_NPAGE:
                sel = min(len(its) - 1, sel + view)
            elif k == curses.KEY_PPAGE:
                sel = max(0, sel - view)
            elif k in ENTER:
                self.sample_view(its[sel])

    def sample_view(self, it):
        import json
        ui = self.ui
        w = ui.size()[1]
        c = ui.color
        lines = [[("status  ", c(T.MUTED)), ("✔ correct" if it["correct"] else ("– " if it["correct"] is None else "✗ ") + f"{it['status']}",
                                             c(T.GREEN if it["correct"] else (T.MUTED if it["correct"] is None else T.RED), None, True))]]
        if it.get("detail"):
            lines += [[("detail  " if n == 0 else "        ", c(T.MUTED)), (ln, c(T.YELLOW))]
                      for n, ln in enumerate(UI.wrap(it["detail"], w - 14))]
        if it.get("finish_reason") or it.get("completion_tokens"):
            lines.append([("tokens  ", c(T.MUTED)), (f"{it.get('completion_tokens')} · finish {it.get('finish_reason')}", c(T.SOFT))])
        lines.append([("", 0)])

        def block(title, text, col):
            if not text:
                return
            lines.append([("▍", c(col)), (title, c(col, None, True))])
            for ln in str(text).splitlines() or [""]:
                for part in UI.wrap(ln, w - 6) or [""]:
                    lines.append([("  " + part, c(T.SOFT))])
            lines.append([("", 0)])
        if it.get("messages"):
            for m in it["messages"] if isinstance(it["messages"], list) else []:
                cont = m.get("content") if isinstance(m, dict) else m
                if isinstance(cont, list):
                    cont = " ".join(p.get("text", f"[{p.get('type')}]") for p in cont if isinstance(p, dict))
                block(f"{m.get('role', '?') if isinstance(m, dict) else '?'}", cont, T.BLUE)
        else:
            block("prompt", it.get("prompt") or it.get("title"), T.BLUE)
        block("reasoning", it.get("reasoning"), T.MUTED)
        block("model response", it.get("response"), T.CYAN)
        if it.get("tool_calls"):
            block("tool calls", json.dumps(it["tool_calls"], indent=2, default=str), T.VIOLET)
        gold = it.get("gold")
        block("expected", gold if isinstance(gold, str) else json.dumps(gold, indent=2, default=str), T.GREEN)
        ui.pager(f"Sample · {it['suite']} · {it['id']}", lines)

    # ------------------------------------------------------------------ leaderboard
    def screen_leaderboard(self):
        ui = self.ui
        comparable = False
        left = 0
        sel = 0
        lb = ui.busy("building the leaderboard…", lambda: leaderboard.build(R.list_runs(self.s.results_path), comparable))
        while True:
            h, w = ui.size()
            self.scr.erase()
            ui.header("Leaderboard", f"{len(lb['models'])} models · gScore = mean of % scores · updated {lb['updated']}"
                      + (" · leaderboard-comparable only" if comparable else ""))
            benches = lb["benches"] + lb["perf"]
            y = 3
            x0 = 4
            ui.put(y, x0, f"{'#':<3}{'model':<22}{'gScore':<18}", ui.color(T.MUTED))
            cw = 12
            cols_x = x0 + 43
            vis = benches[left:]
            for j, b in enumerate(vis):
                if cols_x + j * cw + cw > w - 1:
                    break
                ui.put(y, cols_x + j * cw, _short(b)[: cw - 1], ui.color(T.grad(j + left, max(2, len(benches))), None, True))
            for i, r in enumerate(lb["models"]):
                yy = y + 1 + i * 2
                if yy >= h - 3:
                    break
                if i == sel:
                    ui.sel_bar(yy, 2, w - 4, T.VIOLET)
                bg = T.SEL if i == sel else None
                medal = {0: T.YELLOW, 1: T.SOFT, 2: T.PINK}.get(i, T.MUTED)
                ui.put(yy, x0, f"{i + 1:<3}", ui.color(medal, bg, True))
                ui.put(yy, x0 + 3, f"{r['model'][:21]:<22}", ui.color(T.LIGHT, bg, True))
                g = r["gscore"]
                if g is not None:
                    ui.bar(yy, x0 + 25, 10, g / 100, T.score_color(g))
                    ui.put(yy, x0 + 36, f"{g:5.1f}", ui.color(T.score_color(g), bg, True))
                for j, b in enumerate(vis):
                    cx = cols_x + j * cw
                    if cx + cw > w - 1:
                        break
                    s = r["scores"].get(b)
                    if s is None:
                        ui.put(yy, cx, "   –", ui.color(T.RULE, bg))
                    elif s.unit == "%":
                        ui.put(yy, cx, f"{s.value:5.1f}", ui.color(T.score_color(s.value), bg, True))
                        ui.put(yy, cx + 6, f"{s.n}" if s.n else "", ui.color(T.MUTED, bg))
                    else:
                        ui.put(yy, cx, f"{s.value:.1f}{'' if s.unit == 'tok/s' else s.unit}", ui.color(T.CYAN, bg, True))
                ui.put(yy + 1, x0 + 3, f"{r['when']} · {r['runs']} run(s) · {r['n_bench']} benchmarks", ui.color(T.MUTED))
            if not lb["models"]:
                ui.put(5, 4, "No results yet. Run something (2 · Run): the leaderboard updates itself after every run.",
                       ui.color(T.MUTED))
            p = leaderboard.paths(self.s)
            ui.put(h - 3, 2, f"files: {p['md']} · .html · .json", ui.color(T.MUTED), maxw=w - 4)
            ui.footer("←→ more columns · c comparable only · o open HTML · w write files · q back")
            self.scr.refresh()
            k = ui.key()
            if k in ("q", ESC):
                return
            if k == curses.KEY_RIGHT:
                left = min(max(0, len(benches) - 1), left + 1)
            elif k == curses.KEY_LEFT:
                left = max(0, left - 1)
            elif k in (curses.KEY_UP, "k") and lb["models"]:
                sel = (sel - 1) % len(lb["models"])
            elif k in (curses.KEY_DOWN, "j") and lb["models"]:
                sel = (sel + 1) % len(lb["models"])
            elif k == "c":
                comparable = not comparable
                lb = leaderboard.build(R.list_runs(self.s.results_path), comparable)
            elif k == "w":
                leaderboard.write(self.s, comparable)
                ui.toast("leaderboard.md / .html / .json written", T.GREEN)
                ui.key(900)
            elif k == "o":
                leaderboard.write(self.s, comparable)
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                with contextlib.suppress(OSError):
                    subprocess.Popen([opener, str(p["html"])], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
        vals = {"out": out_default, "fmt": "both", "failures": False, "runs": f"{len(runs)} run(s): " +
                ", ".join(r.id for r in runs[:3]) + ("…" if len(runs) > 3 else "")}
        fields = [Field("runs", "Runs", "info"), Field("out", "Output folder", help="a dataset folder: README.md + data/"),
                  Field("fmt", "Format", "choice", [("jsonl", "JSONL", ""), ("parquet", "Parquet", "pyarrow, else in a sandbox"),
                                                    ("both", "JSONL + Parquet", "")]),
                  Field("failures", "Failures only", "bool", help="only the samples the model got wrong")]
        fields[0].label = vals["runs"]
        if not self.form("Export as a dataset", fields, vals, done_label="Export",
                         sub="samples (prompt, response, gold, correct) + scores, with a Hugging Face dataset card"):
            return
        log: list = []
        out = Path(vals["out"]).expanduser()
        try:
            ui.busy("exporting…", export.export, runs, out, vals["fmt"], vals["failures"], log.append)
        except Exception as e:                       # noqa: BLE001
            log.append(f"✗ {e}")
        log += ["", f"Upload it with:  huggingface-cli upload <you>/<name> {out} --repo-type dataset"]
        ui.pager("Export", log)

    # ------------------------------------------------------------------ sandbox
    def screen_sandbox(self):
        ui = self.ui
        log: list = []
        while True:
            h, w = ui.size()
            self.scr.erase()
            ui.header("Sandbox", "gbench always runs inside Docker Sandboxes (sbx): a microVM with its own RAM cap and network policy")
            y = 3
            st = self.st.sbx
            rows = [("sbx", shutil.which("sbx") or "not installed"),
                    ("status", st), ("template", sandbox.TEMPLATE + ("  ✔ built" if st == "ready" else "")),
                    ("gbench", f"{sandbox.GBENCH_REPO} @ {sandbox.gbench_commit() or sandbox.GBENCH_REF}"),
                    ("run RAM", f"{self.s.sandbox_memory_gb} GB · {self.s.cpus()} CPUs per run sandbox (one run at a time)"),
                    ("network", "deny-all + the endpoint + Hugging Face / GitHub raw (datasets)"),
                    ("host ↔", f"localhost endpoints are reached as {sandbox.host_alias()}"),
                    ("cache", str(sandbox.work_dir()) + "  (the sandbox workspace: HF datasets, in-flight results)")]
            for k_, v in rows:
                ui.put(y, 4, f"{k_:<10}", ui.color(T.MUTED))
                ui.put(y, 15, v, ui.color(T.GREEN if v.endswith("built") or v == "ready" else T.SOFT), maxw=w - 17)
                y += 1
            y += 1
            ui.section(y, 2, "Live gbtui sandboxes", w - 4, T.VIOLET)
            y += 1
            for s in self.st.sbx_list or [{"name": "(none)"}]:
                ui.put(y, 4, f"{s.get('name')}  {s.get('status', '')}  {s.get('memory', '')}", ui.color(T.SOFT))
                y += 1
            y += 1
            if log:
                ui.section(y, 2, "Log", w - 4, T.CYAN)
                for i, ln in enumerate(log[-(h - y - 4):]):
                    ui.put(y + 1 + i, 4, ln, ui.color(T.RED if ln.startswith("✗") else T.GREEN if ln.startswith("✔") else T.SOFT),
                           maxw=w - 6)
            ui.footer("b build template · c refresh catalog · p prune dead sandboxes · s sbx login help · q back")
            self.scr.refresh()
            k = ui.key(1500)
            if k in ("q", ESC):
                return
            if k == "b":
                self.live_task("Building the sandbox template", lambda say: sandbox.setup(say, cpus=self.s.cpus()))
                catalog.reload()
                self.st.kick()
            elif k == "c":
                self.live_task("Refreshing gbench's catalog", sandbox.refresh_catalog)
                catalog.reload()
            elif k == "p":
                gone = ui.busy("pruning…", sandbox.prune)
                log.append(f"✔ pruned {len(gone)} sandbox(es)")
                self.st.kick()
            elif k == "s":
                ui.message("Docker Sandboxes needs one sign-in on this machine. Quit gbench-tui and run:\n\n    sbx login\n\n"
                           "Then build the template here (b). Every run then creates its own sandbox from it.")

    def live_task(self, title, fn):
        """Runs fn(say) in a thread and shows its lines live."""
        ui = self.ui
        lines: list = []
        done = {"v": None, "end": False}

        def work():
            try:
                done["v"] = fn(lines.append)
            except Exception as e:                   # noqa: BLE001
                lines.append(f"✗ {e}")
            done["end"] = True
        threading.Thread(target=work, daemon=True).start()
        while True:
            h, w = ui.size()
            self.scr.erase()
            ui.header(title, "running…" if not done["end"] else ("finished ✔" if done["v"] else "finished ✗"))
            view = h - 5
            for i, ln in enumerate(lines[-view:]):
                ui.put(3 + i, 3, ln, ui.color(T.RED if ln.startswith("✗") else T.GREEN if ln.startswith("✔") else T.SOFT),
                       maxw=w - 5)
            if not done["end"]:
                ui.put(h - 2, 3, SPIN[self.tick % len(SPIN)] + " working (this can take several minutes)", ui.color(T.CYAN))
            ui.footer("q back" if done["end"] else "please wait…")
            self.scr.refresh()
            k = ui.key(300)
            self.tick += 1
            if done["end"] and k in ("q", ESC) + ENTER:
                return done["v"]

    # ------------------------------------------------------------------ settings
    def screen_settings(self):
        s = self.s
        vals = {"sandbox_memory_gb": s.sandbox_memory_gb, "sandbox_cpus": s.sandbox_cpus, "min_free_gb": s.min_free_gb,
                "results_dir": s.results_dir or str(s.results_path), "stop_server_on_exit": s.stop_server_on_exit}
        fields = [Field("sandbox_memory_gb", "Sandbox RAM (GB)", "int", help="cap of each run's microVM"),
                  Field("sandbox_cpus", "Sandbox CPUs (0 = auto)", "int", help=f"auto = {max(2, min(6, (os.cpu_count() or 4) // 2))}"),
                  Field("min_free_gb", "Keep free for the system (GB)", "text",
                        help="no sandbox or llama-server starts if the host would drop below this"),
                  Field("results_dir", "Results folder"),
                  Field("stop_server_on_exit", "Stop llama-server on exit", "bool")]
        if self.form("Settings", fields, vals, done_label="Save"):
            s.sandbox_memory_gb = max(2, int(vals["sandbox_memory_gb"]))
            s.sandbox_cpus = max(0, int(vals["sandbox_cpus"]))
            try:
                s.min_free_gb = float(vals["min_free_gb"])
            except ValueError:
                pass
            s.results_dir = vals["results_dir"]
            s.stop_server_on_exit = vals["stop_server_on_exit"]
            self.save()


def _utf8():
    with contextlib.suppress(locale.Error):
        locale.setlocale(locale.LC_ALL, "")


def run_tui():
    _utf8()
    os.environ.setdefault("TERM", "xterm-256color")

    def main(scr):
        App(scr).main()
    curses.wrapper(main)
