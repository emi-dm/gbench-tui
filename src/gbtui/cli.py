"""Command line: `gbench-tui` opens the TUI; subcommands script the same things.

    gbench-tui                       the TUI
    gbench-tui setup                 build the Docker Sandbox template (gbench + its eval extras) and the catalog
    gbench-tui endpoints             list endpoints (probe each)
    gbench-tui serve [NAME] [--stop] start / stop llama-server for an endpoint's GGUF
    gbench-tui run --tier golden|evals|perf|preflight [--evals a b] [--limit N] [--endpoint NAME] …
    gbench-tui results               list runs
    gbench-tui leaderboard           print (and write) the leaderboard
    gbench-tui export OUT [--format jsonl|parquet|both] [--failures] [--runs ID …]
    gbench-tui prune                 remove sandboxes left by dead processes
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

from . import __version__, config, export, leaderboard, results as R, runner, sandbox, servers


def _p(*a):
    print(*a, flush=True)


def cmd_run(a, s) -> int:
    ep = s.endpoint(a.endpoint)
    if not ep:
        _p(f"✗ no endpoint {a.endpoint or s.active!r} (gbench-tui endpoints)")
        return 2
    d = s.run
    spec = runner.RunSpec(endpoint=ep, tier=a.tier or d.tier, evals=a.evals or d.evals,
                          golden_tasks=a.golden_tasks or d.golden_tasks,
                          eval_limit=d.eval_limit if a.limit is None else a.limit,
                          max_output_tokens=a.max_output_tokens or d.max_output_tokens,
                          thinking=a.thinking or d.thinking, temperature=a.temperature or d.temperature,
                          concurrency=a.concurrency or d.concurrency, suite_timeout=d.suite_timeout,
                          perf_mode=a.perf_mode or d.perf_mode)
    if a.memory:
        s.sandbox_memory_gb = a.memory
    _p("gbench " + runner.command_preview(spec).removeprefix("gbench "))
    why = runner.start(spec, s)
    if why:
        _p("✗ " + why)
        return 2
    signal.signal(signal.SIGTERM, lambda *_: runner.cancel())
    shown = 0
    try:
        while runner.LIVE.active or shown < len(runner.LIVE.lines):
            lines = runner.LIVE.lines
            while shown < len(lines):
                _p(lines[shown])
                shown += 1
            time.sleep(0.4)
    except KeyboardInterrupt:
        _p("■ cancelling…")
        runner.cancel()
        while runner.LIVE.active:
            time.sleep(0.5)
    while shown < len(runner.LIVE.lines):
        _p(runner.LIVE.lines[shown])
        shown += 1
    if runner.LIVE.result_dir:
        run = R.load_run(__import__("pathlib").Path(runner.LIVE.result_dir))
        if run:
            _p(f"\nresults: {run.path}")
            for sc in run.scores:
                v = "—" if sc.value is None else (f"{sc.value:.1f}%" if sc.unit == "%" else f"{sc.value:.2f} {sc.unit}")
                _p(f"  {sc.kind:<7} {sc.name:<28} {v:>12}  n={sc.n} {sc.status}")
    if a.stop_server and ep.gguf:
        servers.stop_server(ep)
    return 0 if runner.LIVE.phase == "done" else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gbench-tui", description="A terminal UI for Google DeepMind's gbench")
    ap.add_argument("--version", action="version", version=f"gbench-tui {__version__}")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("setup", help="build the Docker Sandbox template with gbench")
    sub.add_parser("catalog", help="refresh gbench's suite catalog (from inside a sandbox)")
    sub.add_parser("endpoints", help="list endpoints")
    sv = sub.add_parser("serve", help="start / stop llama-server for an endpoint's GGUF")
    sv.add_argument("name", nargs="?")
    sv.add_argument("--stop", action="store_true")
    r = sub.add_parser("run", help="run gbench in a sandbox")
    r.add_argument("--endpoint")
    r.add_argument("--tier", choices=list(runner.TIERS))
    r.add_argument("--evals", nargs="+")
    r.add_argument("--golden-tasks", nargs="+")
    r.add_argument("--limit", type=int)
    r.add_argument("--max-output-tokens", type=int)
    r.add_argument("--concurrency", type=int)
    r.add_argument("--temperature")
    r.add_argument("--thinking", action="store_true")
    r.add_argument("--perf-mode", choices=["personal", "quick"])
    r.add_argument("--memory", type=int, help="sandbox RAM in GB")
    r.add_argument("--stop-server", action="store_true", help="stop llama-server when the run ends")
    sub.add_parser("results", help="list runs")
    lb = sub.add_parser("leaderboard", help="print and write the leaderboard")
    lb.add_argument("--comparable", action="store_true")
    ex = sub.add_parser("export", help="export runs as a dataset")
    ex.add_argument("out")
    ex.add_argument("--format", default="jsonl", choices=["jsonl", "parquet", "both"])
    ex.add_argument("--failures", action="store_true")
    ex.add_argument("--runs", nargs="+")
    sub.add_parser("prune", help="remove sandboxes left by dead processes")
    a = ap.parse_args(argv)
    s = config.load()

    if a.cmd is None:
        from .tui import run_tui
        run_tui()
        return 0
    if a.cmd == "setup":
        return 0 if sandbox.setup(_p, cpus=s.cpus()) else 2
    if a.cmd == "catalog":
        return 0 if sandbox.refresh_catalog(_p) else 2
    if a.cmd == "endpoints":
        for e in s.endpoints:
            pr = servers.probe(e, timeout=2)
            _p(f"{'★' if e.name == s.active else ' '} {e.name:<20} {e.base_url:<32} model={e.model or '-'} "
               f"tokenizer={e.tokenizer or '-'} {'● ' + ','.join(pr['models'][:3]) if pr['ok'] else '○ ' + pr['error']}")
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
        sys.exit(cmd_run(a, s))
    if a.cmd == "results":
        for run in R.list_runs(s.results_path):
            _p(f"{run.id:<48} {run.status:<9} {run.headline}")
        return 0
    if a.cmd == "leaderboard":
        lbd = leaderboard.write(s, a.comparable)
        _p(leaderboard.to_markdown(lbd))
        _p(f"written: {leaderboard.paths(s)['md']} (.html, .json)")
        return 0
    if a.cmd == "export":
        runs = R.list_runs(s.results_path)
        if a.runs:
            runs = [r for r in runs if r.id in a.runs]
        export.export(runs, __import__("pathlib").Path(a.out).expanduser(), a.format, a.failures, _p)
        return 0
    if a.cmd == "prune":
        _p(f"removed: {', '.join(sandbox.prune()) or 'nothing'}")
        return 0
    return 0


if __name__ == "__main__":
    main()
