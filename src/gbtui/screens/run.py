"""Run, Quantization Sweep, Adapter × Harness Matrix, and Live Run screens."""
from __future__ import annotations

import curses
import time
from pathlib import Path

from .. import catalog, config, harbor_runner, results as R, runner, servers, sysmem
from .. import theme as T
from ..ui import ESC, SIGINT, SPIN, UI, dur
from .common import ENTER, Field, _adapter_color, _clamp_top


class RunMixin:
    def screen_run(self):
        ui = self.ui
        if runner.snapshot()["active"]:
            return self.screen_live()
        if not self.s.endpoints:
            ui.message("Add an endpoint first (1 · Endpoints).")
            return
        d = self.s.run
        active_ep = self.s.endpoint() or self.s.endpoints[0]
        presets = self.s.all_presets()
        vals = {
            "preset": "",
            "endpoint": active_ep.name, "tier": d.tier, "evals": list(d.evals),
            "golden_tasks": list(d.golden_tasks), "eval_limit": d.eval_limit, "max_output_tokens": d.max_output_tokens,
            "thinking": d.thinking, "temperature": d.temperature, "concurrency": d.concurrency,
            "suite_timeout": d.suite_timeout, "perf_mode": d.perf_mode,
            "harness": d.harness, "tool_mode": getattr(d, "tool_mode", "normal") or "normal",
            "harness_dataset": d.harness_dataset,
            "harness_tasks": " ".join(d.harness_tasks), "harness_limit": d.harness_limit,
            "memory": self.s.sandbox_memory_gb,
        }
        tiers = [(k, k, v) for k, v in runner.TIERS.items()]
        preset_choices = [("", "Custom (manual)", "configure tier and options below")] + [
            (k, p["label"], p.get("desc", "")) for k, p in presets.items()
        ]
        tool_modes = [
            ("normal", "normal", "standard individual tool calls (read/edit/bash/...)"),
            ("code_mode", "code_mode", "batch via code execution (e.g. Hermes execute_code)"),
        ]

        def apply_preset(v):
            pkey = v.get("preset")
            if pkey and pkey in presets:
                p = presets[pkey]
                steps = p.get("steps") or []
                if steps:
                    s0 = steps[0]
                    v["tier"] = s0.get("tier", v["tier"])
                    if "evals" in s0:
                        v["evals"] = list(s0["evals"])
                    if "eval_limit" in s0:
                        v["eval_limit"] = s0["eval_limit"]
                    if "perf_mode" in s0:
                        v["perf_mode"] = s0["perf_mode"]
                    if "harness" in s0:
                        v["harness"] = s0["harness"]
                    if "tool_mode" in s0:
                        v["tool_mode"] = s0["tool_mode"]

        def fields_for(v):
            ep_choices = [
                (e.name, e.name, f"{e.effective_quant or '—'} · {e.effective_adapter or 'base'} · {e.base_url}")
                for e in self.s.endpoints
            ]
            fl = [
                Field("preset", "Preset profile", "choice", preset_choices,
                      help="multi-step profiles (Smoke, Core ★, Coding & Math, Quant Shootout, Harness Smoke)"),
                Field("endpoint", "Endpoint", "choice", ep_choices,
                      help="the server gbench / Harbor measures (configure LoRAs in 1 · Endpoints)"),
            ]
            if not v.get("preset"):
                fl.append(Field("tier", "What to run", "choice", tiers, help=runner.TIERS.get(v["tier"], "")))
                if v["tier"] == "golden":
                    fl.append(Field("golden_tasks", "Golden Set tasks", "picker", [self.pick_golden],
                                    help="empty = all 16 (vision / audio cases are N/A for a text-only model)"))
                if v["tier"] in ("evals", "preflight"):
                    fl.append(Field("evals", "Eval suites", "picker", [self.pick_suites],
                                    help="★ = runs on a laptop with no extra setup; see 8 · Suites for each one's needs"))
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
                if v["tier"] == "harness":
                    fl += [
                        Field("harness", "Agent harness", "choice", harbor_runner.HARNESSES,
                              help="unmodified coding agent executed by Harbor inside task containers"),
                        Field("tool_mode", "Tool-call mode", "choice", tool_modes,
                              help="normal tool calls vs code_mode (compare tool call counts & token economy)"),
                        Field("harness_dataset", "Harbor dataset", help="e.g. terminal-bench@2.0, swe-bench-verified, or local path"),
                        Field("harness_tasks", "Task filter (glob)", help="optional space-separated --include-task-name patterns"),
                        Field("harness_limit", "Tasks limit (--n-tasks)", "int", help="manageable local subset (0 = all)", min_val=0),
                        Field("concurrency", "Concurrent trials", "int", help="1 recommended for local llama-server", min_val=1),
                        Field("max_output_tokens", "Max output tokens", "int", help="passed to agent endpoint config", min_val=256),
                    ]
            fl += [Field("memory", "Sandbox RAM (GB)", "int", help="container cap; gbench + datasets fit in 4–6 GB", min_val=2),
                   Field("sep", "", "info")]
            return fl

        def preview(y: int, max_h: int):
            ep = self.s.endpoint(vals["endpoint"]) or self.s.endpoints[0]
            h, w = ui.size()
            ch = min(max_h, 9)
            pkey = vals.get("preset")
            if pkey and pkey in presets:
                specs = runner.specs_for_preset(ep, pkey, self.s)
                est_s = sum((runner.estimate_duration_seconds(sp, self._runs_cache)[0] or 0.0) for sp in specs)
                title_c = f"Preset · {presets[pkey]['label']} ({len(specs)} steps)"
                badge_txt = f"~{dur(est_s)}" if est_s else f"{len(specs)} steps"
                ui.card(y, 2, ch, w - 4, title=title_c, color=T.RULE, title_color=T.PINK,
                        badge=badge_txt, badge_color=T.CYAN)
                step_txt = "  →  ".join(
                    f"{sp.tier}" + (f"({','.join(sp.evals[:2])}·n={sp.eval_limit})" if sp.tier == "evals" else
                                    f"({sp.harness})" if sp.tier == "harness" else "")
                    for sp in specs
                )
                ui.put(y + 1, 4, step_txt, ui.color(T.SOFT), maxw=max(10, w - 8))
                if ch >= 5:
                    ui.put(y + 2, 4, presets[pkey].get("desc", ""), ui.color(T.MUTED), maxw=max(10, w - 8))
                cy = y + 4
            else:
                spec = self.spec_from(vals, ep)
                est_s, _est_src = runner.estimate_duration_seconds(spec, self._runs_cache)
                tcol = T.tier_color(vals["tier"])
                title_c = "Harbor Command & Preflight" if vals["tier"] == "harness" else "Sandbox Command & Preflight"
                est_b = f" · ~{dur(est_s)}" if est_s else ""
                ui.card(y, 2, ch, w - 4, title=title_c, color=T.RULE, title_color=tcol,
                        badge=f"{vals['tier'].upper()} · {vals['memory']}GB{est_b}", badge_color=tcol)
                used = self._draw_cmd_highlighted(y + 1, 4, runner.command_preview(spec), max(10, w - 8), max(1, min(3, ch - 4)))
                cy = y + 1 + used + 1

            up = (self.st.ep_ok.get(ep.name) or {}).get("ok")
            need = max(2, int(vals["memory"])) * sysmem.GB + (0 if not ep.gguf or up or servers.server_pid(ep) else servers.estimate_bytes(ep))
            why = sysmem.check(need, self.s.min_free_gb)
            if cy < y + ch - 1:
                cx = 4
                cx += ui.soft_pill(cy, cx, f"{'✔' if not why else '✗'} RAM {sysmem.fmt(need)}",
                                   fg=T.GREEN if not why else T.RED, bg=T.BAR_ALT) + 1
                if ep.lora:
                    cx += ui.soft_pill(cy, cx, f"LoRA {ep.effective_adapter or 'on'}", fg=_adapter_color(ep.effective_adapter), bg=T.BAR_ALT) + 1
                if up:
                    cx += ui.soft_pill(cy, cx, f"● {ep.name} online", fg=T.GREEN, bg=T.BAR_ALT) + 1
                elif ep.gguf:
                    cx += ui.soft_pill(cy, cx, "▸ llama auto-start", fg=T.CYAN, bg=T.BAR_ALT) + 1
                else:
                    cx += ui.soft_pill(cy, cx, "⚠ endpoint offline", fg=T.YELLOW, bg=T.BAR_ALT) + 1
                if vals["tier"] == "harness" and not pkey:
                    hb_ok = harbor_runner.installed()
                    if cx + 16 < w - 4:
                        ui.soft_pill(cy, cx, "✔ harbor ready" if hb_ok else "⚠ uv tool install harbor",
                                     fg=T.GREEN if hb_ok else T.YELLOW, bg=T.BAR_ALT, maxw=max(0, w - cx - 4))
                else:
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
            if cy < y + ch - 1 and vals["tier"] == "harness" and ep.ctx_warning:
                ui.put(cy, 4, "⚠ " + ep.ctx_warning, ui.color(T.YELLOW), maxw=max(0, w - 8))
            elif cy < y + ch - 1 and vals["tier"] in ("evals", "preflight") and not pkey:
                heavy = [f"{n} ({', '.join(catalog.suite(n)['needs'])})" for n in vals["evals"]
                         if catalog.suite(n) and catalog.suite(n)["needs"]]
                if heavy:
                    ui.put(cy, 4, "⚠ extra needs: " + "; ".join(heavy) + " — try Preflight first",
                           ui.color(T.YELLOW), maxw=max(0, w - 8))

        if not self.form("Run benchmark", lambda: fields_for(vals), vals, done_label="Start run", on_change=apply_preset,
                         extra=preview, extra_h=9,
                         sub="gbench runs inside hardened Docker containers · presets & Harbor supported"):
            return
        ep = self.s.endpoint(vals["endpoint"]) or self.s.endpoints[0]
        self.s.active = ep.name
        self.s.sandbox_memory_gb = max(2, int(vals["memory"]))
        for k in ("tier", "evals", "golden_tasks", "eval_limit", "max_output_tokens", "thinking", "temperature",
                  "concurrency", "suite_timeout", "perf_mode", "harness", "tool_mode", "harness_dataset", "harness_limit"):
            setattr(self.s.run, k, vals[k])
        self.s.run.harness_tasks = str(vals.get("harness_tasks") or "").split()
        self.save()
        if vals.get("preset") and vals["preset"] in presets:
            specs = runner.specs_for_preset(ep, vals["preset"], self.s)
            why = runner.start_batch(specs, self.s)
        else:
            why = runner.start(self.spec_from(vals, ep), self.s)
        if why:
            ui.message(why, title="Cannot start run", color=T.RED)
            return
        self.screen_live()

    @staticmethod
    def spec_from(v, ep):
        tier = v["tier"]
        limit = max(0, int(v.get("harness_limit" if tier == "harness" else "eval_limit") or 0))
        tasks_raw = v.get("harness_tasks") or []
        tasks = tasks_raw.split() if isinstance(tasks_raw, str) else list(tasks_raw)
        return runner.RunSpec(
            endpoint=ep, tier=tier, evals=list(v["evals"]), golden_tasks=list(v["golden_tasks"]),
            eval_limit=limit, max_output_tokens=max(1, int(v["max_output_tokens"] or 4096)),
            thinking=bool(v["thinking"]), temperature=str(v["temperature"] or ""),
            concurrency=max(1, int(v["concurrency"] or 1)), suite_timeout=max(0, int(v["suite_timeout"] or 0)),
            perf_mode=v["perf_mode"],
            harness=str(v.get("harness") or "pi"),
            harness_dataset=str(v.get("harness_dataset") or "terminal-bench@2.0"),
            harness_tasks=tasks,
            tool_mode=str(v.get("tool_mode") or "normal"),
        )

    def screen_sweep(self):
        """Quantization Sweep & Adapter × Harness Matrix launcher."""
        ui = self.ui
        if runner.snapshot()["active"]:
            return self.screen_live()
        mode = ui.choose("Batch Runner · Choose Mode", [
            ("sweep", "⚡ Quantization Sweep (overnight queue)", "auto-cycle start llama-server → preset → stop llama-server across endpoints"),
            ("matrix", "◈ Adapter × Harness Matrix (Harbor)", "cross base + LoRAs × (pi, opencode, hermes) and normal vs code_mode"),
        ])
        if mode is None:
            return
        if mode == "matrix":
            return self.screen_matrix()
        if not self.s.endpoints:
            ui.message("Add endpoints (or press F in 1 · Endpoints to import a .gguf folder) first.")
            return
        ep_items = [
            (e.name, e.name, f"{e.effective_quant or '—'} · {e.effective_adapter or 'base'} · {Path(e.gguf).name if e.gguf else e.base_url}")
            for e in self.s.endpoints
        ]
        chosen_eps = self.multi_pick("Quant Sweep · Select endpoints (Space toggles)", ep_items, {e.name for e in self.s.endpoints})
        if not chosen_eps:
            return
        presets = self.s.all_presets()
        p_items = [(k, p["label"], p.get("desc", "")) for k, p in presets.items()]
        preset_key = ui.choose("Quant Sweep · Select evaluation preset", p_items)
        if not preset_key:
            return
        eps = [self.s.endpoint(n) for n in chosen_eps if self.s.endpoint(n)]
        specs = runner.build_sweep_specs(eps, preset_key, self.s, stop_between=True)
        est_total = sum((runner.estimate_duration_seconds(sp, self._runs_cache)[0] or 0.0) for sp in specs)
        est_s = f" (est. ~{dur(est_total)})" if est_total else ""
        if not ui.confirm(f"Launch Quant Sweep: {len(eps)} endpoint(s) × {presets[preset_key]['label']} = {len(specs)} run(s){est_s}?"):
            return
        why = runner.start_batch(specs, self.s)
        if why:
            ui.message(why, title="Cannot start sweep", color=T.RED)
            return
        self.screen_live()

    def screen_matrix(self):
        """Configure and launch an Adapter × Harness evaluation matrix via Harbor (supporting normal / code_mode / both)."""
        ui = self.ui
        if runner.snapshot()["active"]:
            return self.screen_live()
        if not self.s.endpoints:
            ui.message("Add endpoints (base + LoRAs) first in 1 · Endpoints.")
            return
        ep_items = [(e.name, e.name, f"{e.effective_quant or '—'} · {e.effective_adapter or 'base'}") for e in self.s.endpoints]
        chosen_eps = self.multi_pick("Select endpoints for Adapter × Harness matrix (Space toggles)", ep_items,
                                     {e.name for e in self.s.endpoints})
        if not chosen_eps:
            return
        h_items = [(h[0], h[1], h[2]) for h in harbor_runner.HARNESSES]
        chosen_h = self.multi_pick("Select harnesses to cross (Space toggles)", h_items, {"pi", "opencode", "hermes"})
        if not chosen_h:
            return
        tm = ui.choose("Tool-call mode to evaluate", [
            ("normal", "normal only", "standard per-tool calls"),
            ("code_mode", "code_mode only", "batch via code execution (Hermes execute_code / scripts)"),
            ("both", "normal + code_mode (A/B)", "run both modes to compare tool_calls & token economy"),
        ])
        if tm is None:
            return
        ds = ui.edit("Harbor dataset", self.s.run.harness_dataset or "terminal-bench@2.0",
                     hint="e.g. terminal-bench@2.0, swe-bench-verified, or local task folder")
        if not ds:
            return
        lim_s = ui.edit("Tasks per cell (--n-tasks)", str(self.s.run.harness_limit or 2), hint="0 = all tasks")
        if lim_s is None:
            return
        try:
            lim = max(0, int(lim_s))
        except ValueError:
            lim = 2
        modes = ["normal", "code_mode"] if tm == "both" else [tm]
        specs = []
        for ep_name in chosen_eps:
            ep = self.s.endpoint(ep_name)
            if not ep:
                continue
            for h in chosen_h:
                for m in modes:
                    specs.append(runner.RunSpec(
                        endpoint=ep, tier="harness", eval_limit=lim,
                        max_output_tokens=self.s.run.max_output_tokens, concurrency=1,
                        harness=h, harness_dataset=ds, harness_tasks=list(self.s.run.harness_tasks),
                        tool_mode=m,
                    ))
        if not specs:
            return
        if not ui.confirm(f"Launch {len(specs)} Harbor run(s) ({len(chosen_eps)} endpoints × {len(chosen_h)} harnesses × {len(modes)} mode(s)) on {ds}?"):
            return
        why = runner.start_batch(specs, self.s)
        if why:
            ui.message(why, title="Cannot start matrix", color=T.RED)
            return
        self.screen_live()

    def pick_golden(self, current):
        cat = catalog.load()
        items = [(g["id"], g["id"], f"{g.get('category', '')} · {g.get('description', '')}") for g in cat["golden"]]
        return self.multi_pick("Golden Set tasks (empty = all 16)", items, set(current))

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

    def screen_live(self):
        ui = self.ui
        if runner.snapshot()["phase"] == "idle":
            ui.message("No run in progress. Start one in 2 · Run or 3 · Sweep.")
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
            step_bit = f" · step {L['matrix_pos']}" if L.get("matrix_pos") else ""
            sub = (f"{spec.endpoint.name} · {spec.tier}" if spec else "") + step_bit
            el = (L["finished"] or time.time()) - L["started"] if L["started"] else 0
            ui.header("Live run", sub, right_badge=(f"⏱ {dur(el)}", T.PINK if L["active"] else T.MUTED))

            top_h = 6
            pcol = T.VIOLET if L["active"] else T.status_color(L["phase"])
            badge_p = (f"STEP {L['matrix_pos']} · {L['phase'].upper()}" if L.get("matrix_pos") else L["phase"].upper())
            ui.card(3, 2, top_h, w - 4, title=L["run_id"] or "Benchmark", color=pcol, title_color=T.PINK if L["active"] else pcol,
                    badge=badge_p, badge_color=T.status_color(L["phase"]))
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
                    lbl = L["current"] or {"server": "starting llama-server…", "sandbox": "starting container…",
                                           "running": "running benchmark…", "collecting": "saving results…"}.get(L["phase"], "")
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
            lines = [part for ln in raw_lines for part in UI.wrap(ln, max(20, w - 8))] if wrap_lines else raw_lines
            view = max(1, oh - 2)
            max_top = max(0, len(lines) - view)
            top = max_top if follow else max(0, min(top, max_top))
            max_lw = max((len(ln) for ln in lines[top: top + view]), default=0)
            left = max(0, min(left, max(0, max_lw - max(10, w - 10))))
            obadge = (f"● FOLLOW · {len(raw_lines)} lines" if follow else f"PAUSED {top + 1}/{max(1, len(lines))}")
            ui.card(oy, 2, oh, w - 4, title="Benchmark Output", color=T.RULE, title_color=T.CYAN,
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
                if ui.confirm("Cancel the run? The benchmark stops and its container is removed."):
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
