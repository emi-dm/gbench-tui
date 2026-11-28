"""Command line: `gbench-tui` opens the TUI; subcommands script the same things.

    gbench-tui                       the TUI
    gbench-tui setup                 build the hardened Docker image (gbench + its eval extras) and the catalog
    gbench-tui endpoints             list endpoints (probe each)
    gbench-tui discover [--add]      auto-discover running local servers (Ollama, LM Studio, llama-server, vLLM)
    gbench-tui import-gguf DIR       inspect/import a folder of .gguf files via binary GGUF headers
    gbench-tui loras                 browse / download harness-hub GGUF LoRA adapters
    gbench-tui serve [NAME] [--stop] start / stop llama-server for an endpoint's GGUF (+ optional LoRA)
    gbench-tui run [--preset ...] [--tier golden|evals|perf|preflight|harness] [--tool-mode normal|code_mode] …
    gbench-tui sweep --endpoints E1 E2 ... [--folder DIR] --preset quant-shootout
    gbench-tui matrix --endpoints E1 E2 ... --harnesses pi opencode hermes [--tool-mode normal|code_mode|both]
    gbench-tui results               list runs (with notes/tags/archived)
    gbench-tui compare RUN_A RUN_B   sample-by-sample diff, regressions, improvements & token economy
    gbench-tui leaderboard           print (and write) the leaderboard + Adapter × Harness matrix + Pareto HTML
    gbench-tui report [--out DIR]    generate comparative Markdown + HTML report for GitHub/Reddit/HF cards
    gbench-tui lens [HARNESS]        inspect / compare agent harnesses (Harness Lens)
    gbench-tui play [ENDPOINT] --harness hermes|pi|opencode   launch interactive harness playground
    gbench-tui export OUT [--format jsonl|parquet|both] [--failures] [--runs ID …]
    gbench-tui prune                 remove containers left by dead processes
"""
from __future__ import annotations

import argparse
import re
import signal
import time
from pathlib import Path

from . import (
    __version__,
    compare as CMP,
    config,
    export,
    gguf_meta,
    harbor_runner,
    harness_hub,
    harness_lens,
    installer,
    leaderboard,
    metrics,
    report,
    results as R,
    runner,
    sandbox,
    servers,
)


def _p(*a):
    print(*a, flush=True)


def _stream_live(stop_endpoints: list | None = None) -> int:
    signal.signal(signal.SIGTERM, lambda *_: runner.cancel())
    shown = 0

    def flush():
        nonlocal shown
        snap = runner.snapshot()
        start_idx = max(0, shown - snap["dropped"])
        lines = snap["lines"]
        for ln in lines[start_idx:]:
            _p(ln)
        shown = snap["dropped"] + len(lines)
        return snap

    try:
        while True:
            snap = flush()
            if not snap["active"]:
                break
            time.sleep(0.4)
    except KeyboardInterrupt:
        _p("■ cancelling…")
        runner.cancel()
        runner.wait(timeout=30)
    runner.wait(timeout=30)
    snap = flush()
    if snap["result_dir"]:
        run = R.load_run(Path(snap["result_dir"]))
        if run:
            _p(f"\nresults: {run.path}")
            for sc in run.scores:
                v = "—" if sc.value is None else (f"{sc.value:.1f}%" if sc.unit == "%" else f"{sc.value:.2f} {sc.unit}")
                _p(f"  {sc.kind:<7} {sc.name:<28} {v:>12}  n={sc.n} {sc.status}")
    for ep in (stop_endpoints or []):
        if ep.gguf:
            servers.stop_server(ep)
    return 0 if snap["phase"] == "done" else 1


def cmd_run(a, s) -> int:
    ep = s.endpoint(a.endpoint)
    if not ep:
        _p(f"✗ no endpoint {a.endpoint or s.active!r} (gbench-tui endpoints)")
        return 2
    if a.memory is not None:
        if a.memory < 2:
            _p("✗ --memory must be at least 2 GB")
            return 2
        s.sandbox_memory_gb = a.memory
    if a.preset:
        presets = s.all_presets()
        if a.preset not in presets:
            _p(f"✗ unknown preset {a.preset!r} (available: {', '.join(presets)})")
            return 2
        specs = runner.specs_for_preset(ep, a.preset, s)
        _p(f"Preset {a.preset!r} ({presets[a.preset]['label']}): {len(specs)} step(s) on {ep.name}")
        why = runner.start_batch(specs, s)
        if why:
            _p("✗ " + why)
            return 2
        return _stream_live([ep] if a.stop_server else [])

    d = s.run
    tier = a.tier or d.tier
    default_limit = d.harness_limit if tier == "harness" else d.eval_limit
    spec = runner.RunSpec(
        endpoint=ep,
        tier=tier,
        evals=a.evals or d.evals,
        golden_tasks=a.golden_tasks or d.golden_tasks,
        eval_limit=default_limit if a.limit is None else max(0, a.limit),
        max_output_tokens=d.max_output_tokens if a.max_output_tokens is None else max(1, a.max_output_tokens),
        thinking=d.thinking if a.thinking is None else a.thinking,
        temperature=d.temperature if a.temperature is None else a.temperature,
        concurrency=d.concurrency if a.concurrency is None else max(1, a.concurrency),
        suite_timeout=d.suite_timeout,
        perf_mode=a.perf_mode or d.perf_mode,
        harness=a.harness or d.harness,
        harness_dataset=a.dataset or d.harness_dataset,
        harness_tasks=a.tasks or d.harness_tasks,
        tool_mode=a.tool_mode or getattr(d, "tool_mode", "normal") or "normal",
    )
    _p(runner.command_preview(spec))
    why = runner.start(spec, s)
    if why:
        _p("✗ " + why)
        return 2
    return _stream_live([ep] if a.stop_server else [])


def cmd_sweep(a, s) -> int:
    presets = s.all_presets()
    if a.preset not in presets:
        _p(f"✗ unknown preset {a.preset!r} (available: {', '.join(presets)})")
        return 2
    eps: list[config.Endpoint] = []
    if a.folder:
        folder = Path(a.folder).expanduser()
        if not folder.is_dir():
            _p(f"✗ not a directory: {folder}")
            return 2
        bases, _loras = gguf_meta.scan_folder(folder, recursive=True)
        for g in bases:
            base, fn_q = config.split_model_quant(g.path.stem)
            q = g.quant or fn_q
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", (g.name or base).lower()).strip(".-") or "gguf"
            name = f"{slug}-{q.lower()}" if q and q.lower() not in slug else slug
            eps.append(config.Endpoint(
                name=name, base_url="http://127.0.0.1:8080/v1", model=name,
                quant=q, gguf=str(g.path), ctx=min(65536, max(8192, g.context_length or 16384)),
            ))
    if a.endpoints:
        for n in a.endpoints:
            ep = s.endpoint(n)
            if not ep:
                _p(f"✗ unknown endpoint: {n}")
                return 2
            eps.append(ep)
    if not eps:
        _p("✗ specify --endpoints or --folder")
        return 2
    specs = runner.build_sweep_specs(eps, a.preset, s, stop_between=True)
    _p(f"Quant Sweep ({a.preset}): {len(eps)} endpoint(s) × {len(presets[a.preset]['steps'])} step(s) = {len(specs)} run(s)")
    why = runner.start_batch(specs, s)
    if why:
        _p("✗ " + why)
        return 2
    rc = _stream_live(eps if a.stop_server else [])
    lb = leaderboard.write(s)
    _p("\n" + leaderboard.to_markdown(lb))
    return rc


def cmd_matrix(a, s) -> int:
    eps = [s.endpoint(n) for n in a.endpoints]
    missing = [n for n, ep in zip(a.endpoints, eps) if ep is None]
    if missing:
        _p(f"✗ unknown endpoint(s): {', '.join(missing)}")
        return 2
    harnesses = [config.normalize_harness(h) for h in a.harnesses]
    modes = ["normal", "code_mode"] if a.tool_mode == "both" else [a.tool_mode]
    d = s.run
    specs = []
    for ep in eps:
        assert ep is not None
        for h in harnesses:
            for m in modes:
                specs.append(runner.RunSpec(
                    endpoint=ep,
                    tier="harness",
                    eval_limit=d.harness_limit if a.limit is None else max(0, a.limit),
                    max_output_tokens=d.max_output_tokens if a.max_output_tokens is None else max(1, a.max_output_tokens),
                    concurrency=1 if a.concurrency is None else max(1, a.concurrency),
                    harness=h,
                    harness_dataset=a.dataset or d.harness_dataset,
                    harness_tasks=a.tasks or [],
                    tool_mode=m,
                ))
    _p(f"Adapter × Harness matrix: {len(eps)} endpoint(s) × {len(harnesses)} harness(es) × {len(modes)} mode(s) = {len(specs)} run(s)")
    why = runner.start_batch(specs, s)
    if why:
        _p("✗ " + why)
        return 2
    rc = _stream_live(eps if a.stop_server else [])
    lb = leaderboard.write(s)
    _p("\n" + leaderboard.to_markdown(lb))
    return rc


def cmd_discover(a, s) -> int:
    found = servers.discover_local()
    if not found:
        _p("no running local servers found (checked Ollama :11434, LM Studio :1234, llama-server :8080/:8081, vLLM :8000)")
        return 0
    added = 0
    for f in found:
        models_s = ", ".join(f["models"][:3]) or "(none)"
        _p(f"● {f['provider']:<14} {f['base_url']:<28} {f['latency_ms']:>4}ms  models: {models_s}")
        if a.add:
            model_id = f["models"][0] if f["models"] else ""
            slug_src = model_id or f["provider"].lower().replace(" ", "-")
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", slug_src.lower()).strip(".-") or "local-server"
            if any(e.name == slug for e in s.endpoints):
                continue
            s.endpoints.append(config.Endpoint(
                name=slug, base_url=f["base_url"], model=model_id,
                quant=config.detect_quant(model_id), notes=f"auto-discovered {f['provider']}",
            ))
            s.active = s.active or slug
            added += 1
    if a.add and added:
        config.save(s)
        _p(f"✔ added {added} endpoint(s) to config")
    return 0


def cmd_import_gguf(a, s) -> int:
    folder = Path(a.dir).expanduser()
    if not folder.is_dir():
        _p(f"✗ not a directory: {folder}")
        return 2
    bases, loras = gguf_meta.scan_folder(folder, recursive=a.recursive)
    infos = [*bases, *loras]
    if not infos:
        _p(f"no valid .gguf files found in {folder}")
        return 0
    added = 0
    for g in infos:
        kind = "LoRA " if g.is_lora else "MODEL"
        _p(f"[{kind}] {g.path.name:<46} {g.summary}")
        if a.add and not g.is_lora:
            base, fn_q = config.split_model_quant(g.path.stem)
            q = g.quant or fn_q
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", (g.name or base).lower()).strip(".-") or "gguf"
            name = f"{slug}-{q.lower()}" if q and q.lower() not in slug else slug
            if any(e.name == name for e in s.endpoints):
                continue
            s.endpoints.append(config.Endpoint(
                name=name, base_url="http://127.0.0.1:8080/v1", model=name,
                quant=q, gguf=str(g.path),
                ctx=min(65536, max(8192, g.context_length or 16384)),
                notes=g.summary,
            ))
            s.active = s.active or name
            added += 1
    if a.add and added:
        config.save(s)
        _p(f"✔ imported {added} GGUF endpoint(s)")
    return 0


def cmd_compare(a, s) -> int:
    runs = R.list_runs(s.results_path, include_archived=True)

    def resolve(q: str):
        return next((r for r in runs if r.id == q or q in r.id), None)

    ra, rb = resolve(a.run_a), resolve(a.run_b)
    if not ra or not rb:
        _p(f"✗ could not find runs: A={a.run_a!r} ({'ok' if ra else 'missing'}), B={a.run_b!r} ({'ok' if rb else 'missing'})")
        return 2
    diff = CMP.compare_runs(ra, rb)
    _p("Compare A vs B")
    _p(f"  A: {ra.id} ({ra.model_label})")
    _p(f"  B: {rb.id} ({rb.model_label})")
    _p(f"  Samples: {len(diff.samples)} total · ▼ {diff.regressions} regressions · ▲ {diff.improvements} improvements · = {diff.both_pass} both pass · {diff.both_fail} both fail")
    _p("\nScores:")
    for sc in diff.score_deltas:
        av = f"{sc['a']:.1f}" if sc["a"] is not None else "—"
        bv = f"{sc['b']:.1f}" if sc["b"] is not None else "—"
        dv = f"{sc['delta']:+.1f}" if sc["delta"] is not None else "—"
        _p(f"  {sc['name']:<26} A={av:>8}  B={bv:>8}  Δ={dv} {sc['unit']}")
    regs = [x for x in diff.samples if x.verdict == "regression"]
    if regs:
        _p(f"\nRegressions ({len(regs)}):")
        for x in regs[:15]:
            dt = f" ({x.token_delta:+d} tok)" if x.token_delta is not None else ""
            pr = str((x.a or x.b or {}).get("prompt") or "")[:90].replace("\n", " ")
            _p(f"  ▼ {x.suite}/{x.sample_id}{dt}: {pr}")
    imps = [x for x in diff.samples if x.verdict == "improvement"]
    if imps:
        _p(f"\nImprovements ({len(imps)}):")
        for x in imps[:15]:
            dt = f" ({x.token_delta:+d} tok)" if x.token_delta is not None else ""
            pr = str((x.a or x.b or {}).get("prompt") or "")[:90].replace("\n", " ")
            _p(f"  ▲ {x.suite}/{x.sample_id}{dt}: {pr}")
    return 0


def cmd_report(a, s) -> int:
    out_dir = Path(a.out).expanduser() if a.out else None
    md_p, html_p = report.write_report(s, out_dir=out_dir)
    _p(md_p.read_text(encoding="utf-8"))
    _p(f"written: {md_p} & {html_p}")
    return 0


def cmd_lens(a, _s) -> int:
    if a.compare:
        sa = harness_lens.get_spec(a.compare[0])
        sb = harness_lens.get_spec(a.compare[1])
        if not sa or not sb:
            avail = ", ".join(h.id for h in harness_lens.HARNESS_SPECS)
            _p(f"✗ unknown harness in --compare (available: {avail})")
            return 2
        _p(f"Harness Lens · {sa.name} vs {sb.name}")
        rows = [
            ("Prompt size", f"~{sa.prompt_tokens_est:,} tok", f"~{sb.prompt_tokens_est:,} tok"),
            ("Min context", f"{sa.min_ctx:,}", f"{sb.min_ctx:,}"),
            ("Tool count", f"{len(sa.tools)} tools", f"{len(sb.tools)} tools"),
            ("Edit strategy", sa.edit_strategy[:36], sb.edit_strategy[:36]),
            ("Code Mode", sa.code_mode[:36], sb.code_mode[:36]),
            ("Compaction", sa.compaction[:36], sb.compaction[:36]),
            ("Sandbox", sa.sandbox_model[:36], sb.sandbox_model[:36]),
        ]
        for k, va, vb in rows:
            _p(f"  {k:<15} | {va:<36} | {vb}")
        return 0
    _p(harness_lens.format_plain(a.harness or "", show_prompt=bool(getattr(a, "prompt", False))))
    return 0


def cmd_play(a, s) -> int:
    ep = s.endpoint(a.endpoint)
    if not ep:
        _p(f"✗ no endpoint {a.endpoint or s.active!r} (gbench-tui endpoints)")
        return 2
    sp = harness_lens.get_spec(a.harness)
    if not sp:
        avail = ", ".join(h.id for h in harness_lens.HARNESS_SPECS)
        _p(f"✗ unknown harness {a.harness!r} (available: {avail})")
        return 2
    return harness_lens.run_interactive_playground(sp, ep, s.min_free_gb)


def cmd_loras(a, s) -> int:
    if a.download:
        try:
            adapters = harness_hub.fetch_gguf_adapters(author=a.author, base_filter=a.base or "", harness_filter=a.harness or "")
        except Exception as e:
            _p(f"✗ could not query {a.author}: {e}")
            return 2
        target = next((x for x in adapters if x["repo_id"] == a.download or x["filename"] == a.download
                       or a.download.lower() in x["repo_id"].lower()), None)
        if not target:
            _p(f"✗ no matching GGUF LoRA found for {a.download!r}")
            return 2
        try:
            dest = harness_hub.download_lora(target["repo_id"], target["filename"], on_line=_p)
        except Exception as e:
            _p(f"✗ download failed: {e}")
            return 2
        if a.attach:
            base_ep = s.endpoint(a.attach)
            if not base_ep:
                _p(f"✗ base endpoint {a.attach!r} not found")
                return 2
            ad = target["adapter_name"] or config.detect_adapter(str(dest), target_harness=target["harness"])
            new_name = f"{base_ep.name}-{ad}" if ad else f"{base_ep.name}-lora"
            existing = s.endpoint(new_name)
            ctx = max(base_ep.ctx, 65536)
            ep = existing or config.Endpoint(
                name=new_name, base_url=base_ep.base_url, model=new_name, tokenizer=base_ep.tokenizer,
                quant=base_ep.effective_quant, gguf=base_ep.gguf, ctx=ctx, parallel=base_ep.parallel,
            )
            ep.lora = str(dest)
            ep.adapter_name = ad
            ep.target_harness = target["harness"]
            ep.ctx = max(ep.ctx, 65536)
            if not existing:
                s.endpoints.append(ep)
            config.save(s)
            _p(f"✔ endpoint {ep.name} configured (lora={dest.name}, harness={ep.target_harness}, ctx={ep.ctx})")
        return 0

    try:
        adapters = harness_hub.fetch_gguf_adapters(author=a.author, base_filter=a.base or "", harness_filter=a.harness or "")
    except Exception as e:
        _p(f"✗ could not fetch {a.author} adapters: {e}")
        return 2
    if not adapters:
        _p("no matching GGUF LoRA adapters found")
        return 0
    for x in adapters:
        flag = "✔ cached" if x["cached_path"] else "↓ remote"
        _p(f"{x['repo_id']:<42} {x['harness'] or '-':<14} {x['base_model']:<18} {x['filename']:<42} {flag}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gbt", description="A terminal UI for Google DeepMind's gbench & Harbor (aliases: gbt, gbtui, gbench-tui)")
    ap.add_argument("--version", action="version", version=f"gbench-tui {__version__}")
    sub = ap.add_subparsers(dest="cmd")
    su = sub.add_parser("setup", help="modular setup: install Docker gbench (llm), Harbor + harnesses (agents), or full")
    su.add_argument("--profile", default="llm", choices=list(installer.PROFILES),
                    help="llm (gbench Docker), agents (Harbor + Pi/OpenCode/Hermes), or full (both)")
    su.add_argument("--harnesses", nargs="+", choices=["pi", "opencode", "hermes", "aider", "claude-code"],
                    help="override which harness CLIs to install")
    su.add_argument("--check", action="store_true", help="only print status of installed vs missing components")
    sub.add_parser("catalog", help="refresh gbench's suite catalog (from inside an offline container)")
    sub.add_parser("endpoints", help="list endpoints")

    ds = sub.add_parser("discover", help="auto-discover running local servers (Ollama, LM Studio, llama-server, vLLM)")
    ds.add_argument("--add", action="store_true", help="add discovered servers to config")

    ig = sub.add_parser("import-gguf", help="scan a folder of .gguf files reading binary GGUF headers")
    ig.add_argument("dir", help="folder containing .gguf files")
    ig.add_argument("--recursive", action="store_true", help="scan subdirectories recursively")
    ig.add_argument("--add", action="store_true", help="add discovered base models to config")

    lr = sub.add_parser("loras", help="browse and download harness-hub GGUF LoRA adapters")
    lr.add_argument("--author", default=harness_hub.DEFAULT_AUTHOR)
    lr.add_argument("--base", help="filter by base model (e.g. LFM3-1.7B)")
    lr.add_argument("--harness", help="filter by harness (e.g. pi, opencode, hermes)")
    lr.add_argument("--download", help="download matching repo id or filename to ~/.cache/gbench-tui/loras/")
    lr.add_argument("--attach", help="create/update a LoRA endpoint cloned from this base endpoint name")

    sv = sub.add_parser("serve", help="start / stop llama-server for an endpoint's GGUF (+ optional LoRA)")
    sv.add_argument("name", nargs="?")
    sv.add_argument("--stop", action="store_true")

    r = sub.add_parser("run", help="run gbench (in Docker) or Harbor (agent harness)")
    r.add_argument("--endpoint")
    r.add_argument("--preset", help="evaluation preset profile (smoke, core, coding, quant-shootout, harness-smoke)")
    r.add_argument("--tier", choices=list(runner.TIERS))
    r.add_argument("--evals", nargs="+")
    r.add_argument("--golden-tasks", nargs="+")
    r.add_argument("--harness", help="Harbor agent harness (pi, opencode, hermes, claude-code, openhands, aider)")
    r.add_argument("--tool-mode", choices=["normal", "code_mode"], help="tool-call mode for Harbor runs")
    r.add_argument("--dataset", help="Harbor dataset@version or local task path (e.g. terminal-bench@2.0)")
    r.add_argument("--tasks", nargs="+", help="Harbor --include-task-name patterns")
    r.add_argument("--limit", type=int)
    r.add_argument("--max-output-tokens", type=int)
    r.add_argument("--concurrency", type=int)
    r.add_argument("--temperature")
    r.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None)
    r.add_argument("--perf-mode", choices=["personal", "quick"])
    r.add_argument("--memory", type=int, help="container RAM in GB")
    r.add_argument("--stop-server", action="store_true", help="stop llama-server when the run ends")

    sw = sub.add_parser("sweep", help="run a Quantization Sweep across multiple endpoints or a .gguf folder")
    sw.add_argument("--endpoints", nargs="+", help="endpoint names to sweep")
    sw.add_argument("--folder", help="folder of .gguf files to sweep automatically")
    sw.add_argument("--preset", default="quant-shootout", help="preset profile to run per endpoint")
    sw.add_argument("--stop-server", action="store_true", help="stop llama-server when sweep finishes")

    mx = sub.add_parser("matrix", help="run an Adapter × Harness evaluation matrix via Harbor")
    mx.add_argument("--endpoints", nargs="+", required=True, help="endpoint names (e.g. base, pi-lora, opencode-lora)")
    mx.add_argument("--harnesses", nargs="+", default=["pi", "opencode"], help="Harbor harnesses to cross")
    mx.add_argument("--tool-mode", default="normal", choices=["normal", "code_mode", "both"],
                    help="tool-call mode (normal, code_mode, or both for A/B)")
    mx.add_argument("--dataset", default="terminal-bench@2.0", help="Harbor dataset@version or local path")
    mx.add_argument("--tasks", nargs="+", help="Harbor --include-task-name patterns")
    mx.add_argument("--limit", type=int, help="maximum tasks per cell (--n-tasks)")
    mx.add_argument("--max-output-tokens", type=int)
    mx.add_argument("--concurrency", type=int, help="concurrent Harbor trials (default 1 for local llama-server)")
    mx.add_argument("--stop-server", action="store_true", help="stop llama-server when matrix finishes")

    sub.add_parser("results", help="list runs")

    cp = sub.add_parser("compare", help="compare two runs sample-by-sample (regressions, improvements, token economy)")
    cp.add_argument("run_a", help="Run A id (or substring)")
    cp.add_argument("run_b", help="Run B id (or substring)")

    lb = sub.add_parser("leaderboard", help="print and write the leaderboard")
    lb.add_argument("--comparable", action="store_true")

    rp = sub.add_parser("report", help="generate a comparative Markdown + HTML report for GitHub/Reddit/HF")
    rp.add_argument("--out", help="output directory (default: alongside results)")
    rp.add_argument("--comparable", action="store_true")

    ln = sub.add_parser("lens", help="inspect or compare agent harnesses (Harness Lens)")
    ln.add_argument("harness", nargs="?", help="harness id (pi, opencode, hermes, claude-code, openhands, aider)")
    ln.add_argument("-p", "--prompt", action="store_true", help="print the harness's full system prompt")
    ln.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two harnesses side-by-side")

    pl = sub.add_parser("play", help="launch an interactive agent harness (Hermes/Pi/OpenCode) against an endpoint")
    pl.add_argument("endpoint", nargs="?", help="endpoint name (default: active)")
    pl.add_argument("--harness", default="hermes", help="harness to launch (hermes, pi, opencode, aider, claude-code)")

    ex = sub.add_parser("export", help="export runs as an evaluation dataset or LoRA distillation corpus")
    ex.add_argument("out")
    ex.add_argument("--schema", default="gbench", choices=list(export.SCHEMAS),
                    help="chatml (Unsloth Studio & HF TRL messages), sharegpt (conversations), or gbench (raw eval)")
    ex.add_argument("--filter", dest="split_filter", default="all", choices=list(export.FILTER_MODES),
                    help="passed (rejection sampling SFT), all, or failed")
    ex.add_argument("--reasoning", default="think_tags", choices=list(export.REASONING_MODES),
                    help="think_tags (<think>...</think>), reasoning_content, or strip")
    ex.add_argument("--format", default="jsonl", choices=["jsonl", "parquet", "both"])
    ex.add_argument("--failures", action="store_true", help="alias for --filter failed")
    ex.add_argument("--runs", nargs="+")

    dt = sub.add_parser("distill", help="export verified traces for LoRA knowledge distillation (Unsloth Studio / HF TRL)")
    dt.add_argument("out", help="output dataset directory (writes data/train.jsonl + Unsloth/TRL README.md)")
    dt.add_argument("--schema", default="chatml", choices=["chatml", "sharegpt"],
                    help="chatml (OpenAI/TRL messages, default) or sharegpt (conversations)")
    dt.add_argument("--filter", dest="split_filter", default="passed", choices=list(export.FILTER_MODES),
                    help="passed (default: rejection sampling SFT), all, or failed")
    dt.add_argument("--reasoning", default="think_tags", choices=list(export.REASONING_MODES),
                    help="think_tags (default), reasoning_content, or strip")
    dt.add_argument("--format", default="jsonl", choices=["jsonl", "parquet", "both"])
    dt.add_argument("--runs", nargs="+", help="optional run IDs (or substrings) to include")

    sub.add_parser("prune", help="remove containers left by dead processes")
    mt = sub.add_parser("metrics", help="show every metric or a specific benchmark's scoring formula")
    mt.add_argument("target", nargs="?", help="optional benchmark name (e.g. gsm8k, ifeval, causalbench)")

    a = ap.parse_args(argv)
    if a.cmd == "metrics":
        _p(metrics.format_plain(a.target).rstrip())
        return 0
    if a.cmd == "lens":
        return cmd_lens(a, None)
    s = config.load()

    if a.cmd is None:
        from .tui import run_tui
        run_tui()
        return 0
    if a.cmd == "setup":
        if a.check:
            _p(installer.format_status_plain())
            return 0
        comps = list(installer.PROFILES[a.profile]["components"])
        if a.harnesses:
            comps = [c for c in comps if c in ("gbench_docker", "harbor")] + list(a.harnesses)
            if "harbor" not in comps:
                comps.insert(0, "harbor")
        return 0 if installer.run_setup(comps, on_line=_p, cpus=s.cpus()) else 2
    if a.cmd == "catalog":
        return 0 if sandbox.refresh_catalog(_p) else 2
    if a.cmd == "discover":
        return cmd_discover(a, s)
    if a.cmd == "import-gguf":
        return cmd_import_gguf(a, s)
    if a.cmd == "loras":
        return cmd_loras(a, s)
    if a.cmd == "endpoints":
        for e in s.endpoints:
            pr = servers.probe(e, timeout=2)
            ad = e.effective_adapter or "base"
            _p(f"{'★' if e.name == s.active else ' '} {e.name:<24} {e.base_url:<28} quant={e.effective_quant or '-':<7} "
               f"adapter={ad:<14} ctx={e.ctx:<6} "
               f"{'● ' + ','.join(pr['models'][:2]) if pr['ok'] else '○ ' + pr['error']}")
            if e.ctx_warning:
                _p(f"    ⚠ {e.ctx_warning}")
        return 0
    if a.cmd == "serve":
        e = s.endpoint(a.name)
        if not e:
            _p("✗ no such endpoint")
            return 2
        if a.stop:
            _p("stopped" if servers.stop_server(e) else "not running")
            return 0
        why = servers.start_server(e, s.min_free_gb, on_line=_p)
        _p(("✗ " + why) if why else f"✔ llama-server ready on {e.base_url}")
        return 2 if why else 0
    if a.cmd == "run":
        return cmd_run(a, s)
    if a.cmd == "sweep":
        return cmd_sweep(a, s)
    if a.cmd == "matrix":
        return cmd_matrix(a, s)
    if a.cmd == "compare":
        return cmd_compare(a, s)
    if a.cmd == "report":
        return cmd_report(a, s)
    if a.cmd == "play":
        return cmd_play(a, s)
    if a.cmd == "results":
        for run in R.list_runs(s.results_path):
            tags = [x for x in (run.quant, run.adapter, run.eval_harness, run.tool_mode if run.tool_mode != "normal" else "") if x]
            q = f"[{' · '.join(tags)}] " if tags else ""
            note = f"  ✎ {run.notes}" if run.notes else ""
            _p(f"{run.id:<50} {run.status:<9} {q}{run.headline}{note}")
        return 0
    if a.cmd == "leaderboard":
        lbd = leaderboard.write(s, a.comparable)
        _p(leaderboard.to_markdown(lbd))
        _p(f"written: {leaderboard.paths(s)['md']} (.html, .json)")
        return 0
    if a.cmd in ("export", "distill"):
        runs = R.list_runs(s.results_path)
        if a.runs:
            runs = [r for r in runs if any(q == r.id or q in r.id for q in a.runs)]
            if not runs:
                _p(f"✗ no matching runs found for: {', '.join(a.runs)}")
                return 2
        export.export(
            runs, Path(a.out).expanduser(), a.format, getattr(a, "failures", False), _p,
            schema=a.schema, split_filter=a.split_filter, reasoning_mode=a.reasoning,
        )
        return 0
    if a.cmd == "prune":
        _p(f"removed: {', '.join(sandbox.prune()) or 'nothing'}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
