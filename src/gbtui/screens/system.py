"""System screens: Export (HF dataset & Markdown/HTML report), Docker Sandbox, and Settings (10 themes + notifications)."""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from .. import catalog, export, harbor_runner, installer, metrics, report, results as R, sandbox, sysmem
from .. import theme as T
from ..ui import ESC, SIGINT
from .common import Field


class SystemMixin:
    def screen_export(self):
        runs = self.ui.busy("reading results…", R.list_runs, self.s.results_path)
        if not runs:
            self.ui.message("No runs to export yet.")
            return
        mode = self.ui.choose("Export & Share", [
            ("distill", "Knowledge Distillation / LoRA SFT (Unsloth & HF TRL)", "standardized ChatML `messages` or ShareGPT traces ready for LoRA training"),
            ("dataset", "Evaluation Dataset (gbench raw JSONL / Parquet)", "tabular per-sample prompts, verdicts, gold answers & scores"),
            ("report", "Comparative Report (Markdown + HTML)", "ready for GitHub, Reddit or HF Model Card (quant table, failures, regressions)"),
        ])
        if mode is None:
            return
        if mode == "report":
            md_p, html_p = self.ui.busy("generating comparative report…", report.write_report, self.s)
            md_txt = md_p.read_text(encoding="utf-8")
            self.ui.pager("Comparative Report", self.ui.markdown(md_txt, max(24, self.ui.size()[1] - 6)),
                          sub=f"written to {md_p} & {html_p}")
            return
        chosen = [r for r in runs if r.id in self.marked]
        self.do_export(chosen or runs, default_schema="chatml" if mode == "distill" else "gbench")

    def do_export(self, runs, default_schema: str = "chatml"):
        ui = self.ui
        is_distill = default_schema in ("chatml", "sharegpt")
        prefix = "distill" if is_distill else "gbench"
        out_default = str(Path.home() / "gbench-datasets" / f"{prefix}-{time.strftime('%Y%m%d-%H%M')}")
        vals = {
            "out": out_default,
            "schema": default_schema,
            "split_filter": "passed" if is_distill else "all",
            "reasoning_mode": "think_tags",
            "fmt": "jsonl" if is_distill else "both",
            "runs": f"{len(runs)} run(s): " + ", ".join(r.id for r in runs[:3]) + ("…" if len(runs) > 3 else ""),
        }
        fields = [
            Field("runs", vals["runs"], "info"),
            Field("out", "Output folder", help="dataset folder: README.md + data/train.jsonl (or samples.jsonl)"),
            Field("schema", "Dataset schema", "choice", [
                ("chatml", "ChatML / OpenAI messages (Unsloth & HF TRL)", "recommended for Unsloth Studio, TRL SFTTrainer, MLX-LM & Axolotl"),
                ("sharegpt", "ShareGPT (conversations)", "compatible with unsloth.standardize_sharegpt() & LLaMA-Factory"),
                ("gbench", "gbench Raw Evaluation", "tabular samples.jsonl + scores.jsonl with every benchmark column"),
            ], help="standard format for LoRA distillation or raw evaluation analysis"),
            Field("split_filter", "Sample filter", "choice", [
                ("passed", "Passed only (rejection sampling SFT)", "only correct teacher traces — best for LoRA distillation"),
                ("all", "All samples", "both passed and failed traces"),
                ("failed", "Failures only", "only samples the model got wrong"),
            ], help="for knowledge distillation, 'passed' keeps only verified correct solutions"),
            Field("reasoning_mode", "Reasoning format", "choice", [
                ("think_tags", "<think>...</think> tags", "Unsloth Studio / DeepSeek-R1 / Qwen3 / LFM3 style"),
                ("reasoning_content", "reasoning_content key", "separate HF TRL / OpenAI-compatible field"),
                ("strip", "Strip reasoning", "final answer only (distill thinking -> non-thinking model)"),
            ], help="how teacher chain-of-thought reasoning is encoded in assistant turns"),
            Field("fmt", "File format", "choice", [
                ("jsonl", "JSONL", "drag-and-drop into Unsloth Studio / Desktop"),
                ("parquet", "Parquet", "pyarrow, else in a sandbox"),
                ("both", "JSONL + Parquet", "both formats"),
            ]),
        ]
        while True:
            if not self.form("Export dataset / distillation traces", fields, vals, done_label="Export",
                             sub="exports verified multi-turn traces & tool calls ready for Unsloth Studio / HF TRL LoRA training"):
                return
            if not str(vals["out"] or "").strip():
                ui.message("Output folder cannot be empty.", color=T.RED)
                continue
            break
        log: list = []
        out = Path(vals["out"]).expanduser()
        try:
            ui.busy(
                "exporting dataset…",
                lambda: export.export(
                    runs, out, vals["fmt"], False, log.append,
                    schema=vals["schema"], split_filter=vals["split_filter"],
                    reasoning_mode=vals["reasoning_mode"],
                ),
            )
        except Exception as e:                       # noqa: BLE001
            log.append(f"✗ {e}")
        if vals["schema"] in ("chatml", "sharegpt"):
            log += [
                "",
                f"Unsloth Studio / Desktop: drag & drop  {out / 'data' / 'train.jsonl'}",
                f"Upload to HF Hub:         huggingface-cli upload <you>/<name> {out} --repo-type dataset",
            ]
        else:
            log += ["", f"Upload with:  huggingface-cli upload <you>/<name> {out} --repo-type dataset"]
        ui.pager("Export", [self._log_segs(ln) for ln in log], sub=str(out))

    def screen_sandbox(self):
        ui = self.ui
        log: list = []
        while True:
            h, w = ui.size()
            self.scr.erase()
            st = self.st.sbx
            btxt = "● READY" if st == "ready" else ("CHECKING" if st == "checking…" else "⚠ SETUP NEEDED")
            bcol = T.GREEN if st == "ready" else (T.CYAN if st == "checking…" else T.YELLOW)
            ui.header("Sandbox", "gbench always runs inside a hardened Docker container (read-only rootfs, cap-drop=ALL, RAM/CPU cap)",
                      right_badge=(btxt, bcol))
            st_map = installer.inspect_status()
            h_chips = [
                f"{'●' if st_map[cid]['installed'] else '○'} {cid}"
                for cid in ("harbor", "pi", "opencode", "hermes", "aider")
            ]
            rows = [
                ("docker", shutil.which("docker") or "not installed", T.SOFT if shutil.which("docker") else T.RED),
                ("status", st, T.GREEN if st == "ready" else (T.CYAN if st == "checking…" else T.YELLOW)),
                ("image", sandbox.IMAGE + ("  ✔ built" if st == "ready" else ""), T.GREEN if st == "ready" else T.SOFT),
                ("agents", "  ".join(h_chips), T.CYAN),
                ("llama.cpp", shutil.which("llama-server") or "not installed (optional — only for local .gguf files)",
                 T.GREEN if shutil.which("llama-server") else T.MUTED),
                ("run cap", f"{self.s.sandbox_memory_gb} GB RAM · {self.s.cpus()} CPUs · {sandbox.PIDS_LIMIT} PIDs (one run at a time)", T.CYAN),
                ("hardening", "--read-only · --cap-drop=ALL · no-new-privileges · non-root · no $HOME / docker.sock", T.SOFT),
                ("cache", str(sandbox.work_dir()) + f"  → {sandbox.CONTAINER_WORK} (HF datasets, in-flight runs)", T.DIM),
            ]
            env_h = len(rows) + 2
            ui.card(3, 2, env_h, w - 4, title="Modular Setup & Docker Sandbox", color=T.RULE, title_color=T.VIOLET)
            for idx, (k_, v, col_) in enumerate(rows):
                ui.put(4 + idx, 4, f"{k_:<10}", ui.color(T.DIM))
                ui.put(4 + idx, 15, v, ui.color(col_), maxw=max(0, w - 19))

            ly = 3 + env_h + 1
            live_list = self.st.sbx_list
            live_h = min(max(3, len(live_list) + 2), max(3, h - ly - (7 if log else 3)))
            if ly + live_h <= h - 2:
                ui.card(ly, 2, live_h, w - 4, title="Live gbtui Containers", color=T.RULE, title_color=T.CYAN,
                        badge=f"{len(live_list)} alive", badge_color=T.GREEN if live_list else T.DIM)
                if not live_list:
                    ui.put(ly + 1, 4, "(none — containers are created on demand and removed after each run)",
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

            ui.footer("i modular setup (LLMs/Agents/Harnesses) · b build image · c refresh catalog · p prune · s help · q back")
            self.scr.refresh()
            k = ui.key(1000)
            self.tick += 1
            if k is None or k == "RESIZE":
                continue
            if k in ("q", ESC, SIGINT):
                return
            if k == "i":
                self._modular_setup_wizard()
            elif k == "b":
                self.live_task("Building the Docker image", lambda say: sandbox.setup(say, cpus=self.s.cpus()))
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
                gone = ui.busy("pruning dead containers…", sandbox.prune)
                msg = f"✔ pruned {len(gone)} container(s)"
                log.append(msg)
                ui.notify(msg, T.GREEN)
                self.st.kick()
            elif k == "s":
                ui.message("gbench-tui uses standard Docker (Docker Desktop, OrbStack, Colima or Linux Docker Engine) —\n"
                           "no account or sign-in is required.\n\n"
                           "Start your Docker runtime, then press b here (or run `gbench-tui setup`) to build the\n"
                           "image once. Every run starts a fresh hardened container (--read-only, --cap-drop=ALL,\n"
                           "no-new-privileges, RAM/CPU/PID limits) and removes it when finished.",
                           title="Docker Sandbox help", color=T.VIOLET)

    def _modular_setup_wizard(self):
        """Interactive modular installer: LLMs only, Agents & Harnesses, Full, or custom component selection."""
        ui = self.ui
        choice = ui.choose("Modular Setup · What do you want to install?", [
            ("llm", "LLMs only (gbench Docker + catalog)", "Golden Set, ~90 academic eval suites & perf"),
            ("agents", "Agentic evals & harnesses (Harbor + Pi/OpenCode/Hermes)", "Terminal-Bench / SWE-bench + interactive Play"),
            ("full", "Full suite (LLMs + Agents + Harnesses)", "installs everything; only llama.cpp is optional for local GGUFs"),
            ("custom", "Custom selection…", "pick individual components and harnesses"),
        ])
        if choice is None:
            return
        if choice == "custom":
            st = installer.inspect_status()
            items = [
                (c.id, f"{'✔ ' if st[c.id]['installed'] else '○ '}{c.label}", f"[{c.group}] {c.desc}")
                for c in installer.COMPONENTS
            ]
            default_sel = {c.id for c in installer.COMPONENTS if not st[c.id]["installed"] and c.group in ("llm", "agents")}
            comps = self.multi_pick("Select components to install / update (Space toggles)", items, default_sel)
            if not comps:
                return
        else:
            comps = list(installer.PROFILES[choice]["components"])
        self.live_task(f"Modular Setup ({choice})", lambda say: installer.run_setup(comps, on_line=say, cpus=self.s.cpus()))
        self._cat_count = None
        self.st.kick()

    def screen_settings(self):
        s = self.s
        prev_theme = T.current_theme()
        theme_choices = [(name, T.THEMES[name]["label"], T.THEMES[name]["desc"]) for name in T.THEME_ORDER]
        vals = {
            "theme": getattr(s, "theme", "aurora"),
            "sandbox_memory_gb": s.sandbox_memory_gb, "sandbox_cpus": s.sandbox_cpus, "min_free_gb": s.min_free_gb,
            "results_dir": s.results_dir or str(s.results_path),
            "stop_server_on_exit": s.stop_server_on_exit,
            "notify_on_finish": getattr(s, "notify_on_finish", True),
        }
        fields = [
            Field("theme", f"Color theme ({len(T.THEME_ORDER)})", "choice", theme_choices,
                  help="live preview with ←→ · also press t on Home"),
            Field("sandbox_memory_gb", "Sandbox RAM (GB)", "int", help="RAM cap of each run's Docker container", min_val=2),
            Field("sandbox_cpus", "Sandbox CPUs (0 = auto)", "int",
                  help=f"auto = {max(2, min(6, (os.cpu_count() or 4) // 2))}", min_val=0),
            Field("min_free_gb", "Keep free for system (GB)", "float",
                  help="no sandbox or llama-server starts if host memory would drop below this", min_val=0.5),
            Field("results_dir", "Results folder", help="where runs and leaderboards are stored"),
            Field("stop_server_on_exit", "Stop llama-server on exit", "bool", help="stop servers started by gbench-tui on quit"),
            Field("notify_on_finish", "Notify when run finishes", "bool", help="terminal bell + macOS/Linux desktop notification"),
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
            s.notify_on_finish = vals["notify_on_finish"]
            self.save()
            self.ui.notify(f"✔ settings saved (theme: {T.theme_info()['label']})", T.GREEN)
        else:
            T.set_theme(prev_theme)
