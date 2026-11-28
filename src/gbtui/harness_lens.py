"""Harness Lens: side-by-side dissection of open-source coding agent harnesses (inspired by harness-hub/harness-lens)
plus interactive harness launcher (Hermes, Pi, OpenCode) against any local/remote endpoint.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import config, sandbox, servers


@dataclass(frozen=True)
class HarnessSpec:
    id: str
    name: str
    tagline: str
    repo: str
    min_ctx: int
    prompt_tokens_est: int
    code_mode: str                  # native | plugin | optional | none
    edit_strategy: str
    compaction: str
    sandbox_model: str
    tools: tuple[str, ...]
    system_prompt_sections: tuple[tuple[str, str], ...]
    cli_bins: tuple[str, ...]
    install_hint: str
    strengths: str
    caveats: str
    system_prompt: str = ""


HARNESS_SPECS: tuple[HarnessSpec, ...] = (
    HarnessSpec(
        id="pi",
        name="Pi",
        tagline="Minimal, hyper token-efficient coding agent designed for local & frontier models",
        repo="github.com/earendil-works/pi",
        min_ctx=32768,
        prompt_tokens_est=750,
        code_mode="native (pi-mcp-adapter / bash scripting)",
        edit_strategy="Strict exact-string replacement (edits[].oldText must match uniquely; batched edits)",
        compaction="Lightweight rolling summary + minimal system preamble; high KV-cache hit rate",
        sandbox_model="Runs inside Harbor Docker task container or host shell with extension hooks",
        tools=("bash", "read", "edit", "write"),
        system_prompt_sections=(
            ("Preamble", "Concise coding assistant role (~45 tokens); zero filler."),
            ("Tools (<tools>)", "Only 4 core primitives (read, bash, edit, write) keeping schema overhead tiny."),
            ("Rules (<rules>)", "Prefer read over cat/sed; exact unique oldText for edit; batch edits per file."),
            ("Docs & Extensions", "Lazy-loaded docs & TypeScript extensions (max-turns, MCP adapter)."),
        ),
        cli_bins=("pi",),
        install_hint="npm i -g @earendil-works/pi-coding-agent",
        strengths="Lowest prompt overhead (~750 tok), fastest iterations on small local models (1.7B–8B), strong KV cache reuse.",
        caveats="Strict exact-match edits can fail if model hallucinates indentation; fewer built-in guardrails.",
        system_prompt="""You are an expert coding assistant operating inside pi, a minimal coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
- read: Read file contents (supports offset and limit for large files)
- bash: Execute bash commands (ls, grep, find, tests, scripts, git)
- edit: Make surgical edits to files (exact text replacement; edits[].oldText must match uniquely)
- write: Create or overwrite files

Guidelines:
- Prefer `read` to examine files before editing. Do not use `cat` or `sed` via bash when `read` suffices.
- Use `edit` for surgical changes; ensure `oldText` matches exact whitespace and indentation uniquely.
- Batch related edits to the same file in one tool call when possible.
- Use `bash` for file discovery (`rg`, `fd`, `find`), running tests, and verifying your changes.
- Be concise in your responses. Show file paths clearly when summarizing changes.
- Verify fixes by running the relevant tests or reproduction commands before finishing.""",
    ),
    HarnessSpec(
        id="opencode",
        name="OpenCode",
        tagline="Open-source terminal coding agent with file snapshots, undo & multi-strategy edits",
        repo="github.com/sst/opencode",
        min_ctx=65536,
        prompt_tokens_est=2400,
        code_mode="native (code-mode MCP & bash orchestration)",
        edit_strategy="Multi-strategy fuzzy/indentation-tolerant edit with automatic git-like snapshot & undo",
        compaction="Automatic session compaction & tool-output truncation at context thresholds",
        sandbox_model="Per-workspace permission policies + Harbor Docker container isolation",
        tools=("bash", "read", "edit", "write", "glob", "grep", "list", "todowrite", "webfetch"),
        system_prompt_sections=(
            ("Identity & Workflow", "Structured software engineering workflow with verification & todo tracking."),
            ("Tool Contracts", "Rich tool schemas (glob/grep/edit/bash/todowrite) with explicit safety constraints."),
            ("Snapshotting", "Tracks workspace state snapshots across turns for safe rollback."),
            ("Provider Routing", "Supports @ai-sdk/openai-compatible via custom `local/<model>` provider."),
        ),
        cli_bins=("opencode",),
        install_hint="npm i -g opencode-ai",
        strengths="Forgiving edit recovery, structured planning (todowrite), great for multi-file refactors.",
        caveats="Higher per-turn context (~2.4k+ system/tool tokens); needs >=64k ctx on local models.",
        system_prompt="""You are OpenCode, an interactive CLI coding agent that helps users with software engineering tasks.

# Tone and style
- Be concise, direct, and technical. Avoid unnecessary preamble or postamble.
- Only use emojis if the user explicitly requests them.
- When referencing code, include exact file paths and line numbers.

# Professional objectivity
- Prioritize technical accuracy and truthfulness over agreeing with the user.
- Verify assumptions by inspecting files and running commands rather than guessing.

# Task Management
- Use the `todowrite` tool frequently on multi-step tasks to plan and track progress.
- Mark tasks as `in_progress` before starting and `completed` immediately after verification.

# Tool Usage Policy
- Prefer specialized tools (`glob`, `grep`, `read`, `edit`, `write`) over raw shell equivalents.
- Always `read` a file before calling `edit`. Preserve existing code conventions and indentation.
- Call multiple independent read-only tools in parallel to minimize turns.
- Use `bash` for builds, tests, git operations, and verification commands.
- Workspace state is snapshotted across edits so changes can be reviewed or reverted safely.

# Code Conventions
- Never assume a library is available without checking `package.json`, `pyproject.toml`, or imports.
- Follow the repository's existing formatting, naming, typing, and testing patterns.
- Run lint/test commands after modifications and report the verified outcome.""",
    ),
    HarnessSpec(
        id="hermes",
        name="Hermes Agent",
        tagline="Nous Research general-purpose agent with persistent memory, toolsets & QuickJS Code Mode",
        repo="github.com/NousResearch/hermes-agent",
        min_ctx=65536,
        prompt_tokens_est=3200,
        code_mode="plugin (codemode / QuickJS code-as-orchestration + bash)",
        edit_strategy="Multi-tool file patcher + bash + optional codemode programmatic orchestration",
        compaction="Persistent memory profiles + session summarization + configurable toolsets",
        sandbox_model="Toolset permission boundaries + QuickJS sandbox for codemode + Harbor Docker",
        tools=("terminal", "read_file", "write_file", "patch_file", "search_files", "codemode", "memory", "browser"),
        system_prompt_sections=(
            ("Persona & Memory", "Persistent user/project memory injection and multi-turn assistant framing."),
            ("Toolsets (--toolsets)", "Modular toolsets: core, coding, web, memory, and `codemode` plugin."),
            ("Code Mode", "Exposes programmatic execution to batch multi-tool calls in one inference turn."),
            ("Provider Routing", "Native `openai-api` provider honours OPENAI_BASE_URL + OPENAI_API_KEY."),
        ),
        cli_bins=("hermes", "hermes-agent"),
        install_hint="uv tool install hermes-agent  (or pip install hermes-agent)",
        strengths="Most versatile assistant harness; modular `--toolsets` makes normal vs Code Mode A/B comparisons easy.",
        caveats="Largest prompt footprint (~3.2k tokens); benefits strongly from Hermes-adapted LoRA + >=64k ctx.",
        system_prompt="""You are Hermes, a capable autonomous AI assistant built by Nous Research.
You operate inside a modular toolset environment with persistent memory, terminal access, file patching, and optional Code Mode execution.

<identity>
- Be helpful, resourceful, direct, and honest about what you have verified vs. what you infer.
- Break complex tasks into clear steps and verify every change in the real environment.
</identity>

<toolsets>
Active toolsets depend on `--toolsets` configuration:
- terminal: Execute shell commands in the workspace and inspect exit codes / stdout / stderr.
- file: `read_file`, `write_file`, `patch_file`, `search_files` for safe, targeted code edits.
- codemode (when enabled): `execute_code` runs sandboxed scripts that compose multiple tool calls, loops, filtering, and data transformations inside a single inference turn.
- memory: Read and update persistent user/project notes across sessions when relevant.
</toolsets>

<code_mode_policy>
When the `codemode` toolset is enabled:
- Prefer batching multi-step file inspection, filtering, aggregation, and verification into a concise programmatic script (`execute_code`) rather than issuing many sequential single-tool round trips.
- Return only the compact, relevant results from the script to preserve context budget.
</code_mode_policy>

<execution_rules>
- Inspect existing files and environment state before editing.
- Use `patch_file` for surgical edits; verify syntax and run tests via `terminal`.
- Never fabricate command outputs or test results. If a command fails, diagnose the root cause and iterate.""",
    ),
    HarnessSpec(
        id="claude-code",
        name="Claude Code",
        tagline="Anthropic agentic coding CLI with rich search, subagents, hooks and MCP",
        repo="github.com/anthropics/claude-code",
        min_ctx=65536,
        prompt_tokens_est=4500,
        code_mode="optional (bash + MCP tool search)",
        edit_strategy="Strict Read-before-Edit string replacement with line-numbered inspection",
        compaction="Auto-compact conversation history with structured summary preservation",
        sandbox_model="Granular tool allowlists, hooks, OS sandbox / Harbor Docker",
        tools=("Bash", "Read", "Edit", "Write", "Glob", "Grep", "Task", "WebFetch", "MCP"),
        system_prompt_sections=(
            ("System & Tone", "Concise engineering guidelines, safety & git hygiene rules."),
            ("Tool Policy", "Read before Edit, dedicated Glob/Grep over bash find/grep, parallel calls."),
            ("Subagents & Hooks", "Delegated exploration subagents and pre/post tool hooks."),
        ),
        cli_bins=("claude",),
        install_hint="npm i -g @anthropic-ai/claude-code",
        strengths="Industry-leading edit precision, subagent delegation and permission UX.",
        caveats="Heavy system prompt (~4.5k+ tokens); tuned primarily for Anthropic API format.",
        system_prompt="""You are Claude Code, an interactive CLI agent for software engineering tasks.

# Doing tasks
- Do not propose changes to code you have not read. Always inspect files with `Read` before calling `Edit`.
- Keep solutions minimal, focused, and free of over-engineering or speculative abstractions.
- Avoid introducing security vulnerabilities (command injection, XSS, SQL injection, path traversal).
- Do not commit, push, or run destructive git operations unless explicitly requested by the user.

# Using your tools
- Prefer dedicated tools over `Bash`: use `Read` instead of `cat`, `Edit` instead of `sed`, `Glob` instead of `find`, and `Grep` instead of `grep`/`rg`.
- Issue independent read-only tool calls in parallel.
- Delegate broad codebase exploration to the `Task` subagent when searching across many files.
- Reserve `Bash` for tests, builds, package managers, git inspection, and runtime verification.

# Output efficiency
- Keep responses concise and direct; lead with the outcome and exact file paths modified.""",
    ),
    HarnessSpec(
        id="openhands",
        name="OpenHands",
        tagline="Full-featured software engineering agent with event stream & Jupyter/bash runtime",
        repo="github.com/All-Hands-AI/OpenHands",
        min_ctx=65536,
        prompt_tokens_est=3600,
        code_mode="native (IPython / bash execution loop)",
        edit_strategy="StrReplaceEditor + persistent bash + IPython cell execution",
        compaction="Event-stream condenser summarizing old observation events",
        sandbox_model="Dedicated Docker action-execution server per session",
        tools=("execute_bash", "execute_ipython_cell", "str_replace_editor", "think", "finish"),
        system_prompt_sections=(
            ("Event Stream", "Structured Action/Observation protocol with explicit finish action."),
            ("Code Execution", "Dual persistent bash + IPython kernel for programmatic exploration."),
            ("Condenser", "Configurable history condenser for long SWE-bench trajectories."),
        ),
        cli_bins=("openhands",),
        install_hint="uv tool install openhands-ai",
        strengths="Battle-tested on SWE-bench; native IPython execution acts as built-in Code Mode.",
        caveats="Heavier container/runtime startup overhead.",
        system_prompt="""You are OpenHands agent, a helpful AI software engineer that interacts with a computer inside a sandboxed container to solve engineering tasks.

<ROLE>
Your primary role is to solve software engineering tasks by exploring the repository, reproducing bugs, implementing minimal clean fixes, and verifying them with automated tests.
</ROLE>

<EFFICIENCY>
- Each action emits an observation in the event stream. Batch related inspections into single `execute_bash` or `execute_ipython_cell` calls when helpful.
- Use `str_replace_editor` (`view`, `str_replace`, `create`, `insert`) for exact file modifications.
- Use `think` sparingly to plan complex multi-file edits before touching code.
</EFFICIENCY>

<VERIFICATION>
- Always run the project's test suite and edge-case checks after editing.
- Inspect `git diff` to ensure no stray debug files or unintended modifications remain.
- Call the `finish` tool only when all requirements have been verified.""",
    ),
    HarnessSpec(
        id="aider",
        name="Aider",
        tagline="Git-native pair-programming CLI with repo-map and SEARCH/REPLACE diffs",
        repo="github.com/Aider-AI/aider",
        min_ctx=32768,
        prompt_tokens_est=1400,
        code_mode="none (diff-driven + optional /run test commands)",
        edit_strategy="Whole-file or SEARCH/REPLACE diff blocks with automatic git commits",
        compaction="Tree-sitter repo map ranked by graph centrality + chat history summarizer",
        sandbox_model="Git-tracked working tree with automatic undo; Harbor Docker for evals",
        tools=("repo_map", "search_replace_edit", "git_commit", "run_shell"),
        system_prompt_sections=(
            ("Edit Format", "Strict SEARCH/REPLACE or udiff block syntax instructions."),
            ("Repo Map", "Compact AST symbol map of repository files within token budget."),
            ("Lint & Test Loop", "Optional auto-lint / auto-test feedback loop on each commit."),
        ),
        cli_bins=("aider",),
        install_hint="uv tool install aider-chat",
        strengths="Very compact prompts + git commits per change; works well on small/medium local models.",
        caveats="Not a general autonomous bash agent; focused strictly on code editing.",
        system_prompt="""Act as an expert software developer.
Always use best practices when coding. Respect and use existing conventions, libraries, etc. that are already present in the code base.

Take requests for changes to the supplied code. If the request is ambiguous, ask questions.
Once you understand the request you MUST:
1. Think step-by-step and explain the needed changes in a few short sentences.
2. Describe each change with a *SEARCH/REPLACE block* per the rules below.

Every *SEARCH/REPLACE block* must use this exact format:
path/to/file.py
<<<<<<< SEARCH
exact existing lines to match
=======
replacement lines
>>>>>>> REPLACE

Rules:
- The SEARCH section must match existing lines in the file character-for-character, including leading whitespace and comments.
- Keep SEARCH/REPLACE blocks concise: include only enough surrounding context to match uniquely.
- Only edit files that have been added to the chat; ask to add other files if needed.""",
    ),
)


def all_specs() -> tuple[HarnessSpec, ...]:
    return HARNESS_SPECS


def by_id(harness_id: str) -> HarnessSpec | None:
    key = config.normalize_harness(harness_id)
    return next((h for h in HARNESS_SPECS if h.id == key), None)


get_spec = by_id


def captured_system_prompt(spec: HarnessSpec) -> tuple[str, str]:
    """Looks for a real system prompt captured in saved Harbor ATIF trajectories for `spec`.

    Returns `(prompt_text, source_label)` or `("", "")` if none is cached locally.
    """
    search_roots = []
    try:
        res_dir = config.load().results_path
        if res_dir.is_dir():
            search_roots.append((res_dir, "trajectories/*.json"))
    except Exception:                               # noqa: BLE001
        pass
    jobs_root = sandbox.work_dir() / "harbor-jobs"
    if jobs_root.is_dir():
        search_roots.append((jobs_root, "*/agent/trajectory.json"))

    for root, pattern in search_roots:
        for jdir in sorted(root.iterdir(), reverse=True):
            if not jdir.is_dir() or f"-harness-{spec.id}" not in jdir.name.lower():
                continue
            for traj in jdir.glob(pattern):
                try:
                    data = json.loads(traj.read_text(encoding="utf-8", errors="replace"))
                except (OSError, ValueError):
                    continue
                if not isinstance(data, dict):
                    continue
                for st in data.get("steps") or []:
                    if isinstance(st, dict) and str(st.get("source") or "").lower() == "system":
                        msg = st.get("message")
                        if isinstance(msg, str) and msg.strip():
                            return msg.strip(), f"captured from Harbor run {jdir.name}"
    return "", ""


def load_system_prompt(spec: HarnessSpec, prefer_captured: bool = True) -> tuple[str, str]:
    """Returns `(system_prompt_text, source_label)` — live captured Harbor prompt if present, else reference prompt."""
    if prefer_captured:
        cap, src = captured_system_prompt(spec)
        if cap:
            return cap, src
    return (spec.system_prompt or "").strip(), f"reference prompt ({spec.repo})"


def format_plain(harness_id: str = "", show_prompt: bool = False) -> str:
    """Plain-text summary (or full system prompt) of one or all harnesses for `gbt lens`."""
    if harness_id:
        sp = by_id(harness_id)
        if not sp:
            return f"Unknown harness {harness_id!r}. Available: {', '.join(h.id for h in HARNESS_SPECS)}"
        if show_prompt:
            prompt_txt, src = load_system_prompt(sp)
            est_tok = max(1, len(prompt_txt) // 4)
            return (
                f"# {sp.name} ({sp.id}) · System Prompt\n"
                f"# source: {src} · {len(prompt_txt.splitlines())} lines · ~{est_tok:,} tok\n\n"
                f"{prompt_txt}"
            )
        lines = [
            f"{sp.name} ({sp.id}) — {sp.tagline}",
            f"  repo          : {sp.repo}",
            f"  prompt size   : ~{sp.prompt_tokens_est:,} tok · min ctx {sp.min_ctx:,}",
            f"  tools ({len(sp.tools):>2})    : {', '.join(sp.tools)}",
            f"  edit strategy : {sp.edit_strategy}",
            f"  code mode     : {sp.code_mode}",
            f"  compaction    : {sp.compaction}",
            f"  sandbox       : {sp.sandbox_model}",
            f"  install       : {sp.install_hint}",
            f"  strengths     : {sp.strengths}",
            f"  caveats       : {sp.caveats}",
            "  prompt sections:",
        ]
        for sec_name, sec_desc in sp.system_prompt_sections:
            lines.append(f"    - {sec_name}: {sec_desc}")
        lines.append(f"\nRead full system prompt: gbt lens {sp.id} --prompt")
        return "\n".join(lines)
    lines = ["Harness Lens — Open-Source Coding Agent Harnesses", ""]
    for sp in HARNESS_SPECS:
        inst = "● CLI installed" if installed_bin(sp) else "○ built-in REPL / Harbor"
        lines.append(f"  {sp.id:<12} {sp.name:<14} ~{sp.prompt_tokens_est:>5,} tok · {len(sp.tools):>2} tools · {sp.code_mode[:30]:<30}  {inst}")
    lines += ["", "Inspect: gbt lens hermes  ·  Read prompt: gbt lens hermes --prompt  ·  Compare: gbt lens --compare pi hermes"]
    return "\n".join(lines)


def installed_bin(spec: HarnessSpec) -> str | None:
    for b in spec.cli_bins:
        p = shutil.which(b)
        if p:
            return p
    return None


def launch_command(spec: HarnessSpec, ep: config.Endpoint) -> tuple[list[str], dict[str, str], str]:
    """Returns `(argv, env, mode_note)` to launch an interactive session of `spec` against `ep`.

    Uses the native CLI if installed (`hermes`, `pi`, `opencode`, `aider`, `claude`). If not installed locally, falls
    back to a lightweight interactive Python agent REPL using the harness's system prompt & endpoint `/v1` API so the
    user can always play with the model + LoRA immediately without extra setup.
    """
    env = dict(os.environ)
    key = os.environ.get(ep.api_key_env, "") if ep.api_key_env else ""
    base = ep.base_url.rstrip("/")
    model = ep.model or ep.name or "default"
    env["OPENAI_BASE_URL"] = base
    env["OPENAI_API_BASE"] = base
    env["OPENAI_API_KEY"] = key or "local"

    bin_path = installed_bin(spec)
    if bin_path:
        if spec.id == "hermes":
            return [bin_path, "--provider", "openai-api", "--model", model], env, f"native {bin_path}"
        if spec.id == "pi":
            return [bin_path, "--model", f"openai/{model}"], env, f"native {bin_path}"
        if spec.id == "opencode":
            return [bin_path, "--model", f"local/{model}"], env, f"native {bin_path}"
        if spec.id == "aider":
            return [bin_path, "--model", f"openai/{model}", "--openai-api-base", base, "--openai-api-key", key or "local"], env, f"native {bin_path}"
        return [bin_path], env, f"native {bin_path}"
    return [], env, "built-in interactive playground"


def run_interactive_playground(spec: HarnessSpec, ep: config.Endpoint, min_free_gb: float = 4.0) -> int:
    """Starts `llama-server` if needed, then launches either the native harness CLI or a built-in interactive REPL."""
    if ep.gguf and not (servers.server_pid(ep) and servers._config_matches_running(ep) and servers.probe(ep, timeout=1.5)["ok"]):
        print(f"▸ Starting llama-server for {ep.name}…", flush=True)
        why = servers.start_server(ep, min_free_gb, on_line=print)
        if why:
            print(f"✗ {why}", flush=True)
            return 2
    argv, env, mode = launch_command(spec, ep)
    ad = ep.effective_adapter or "base"
    print(f"\n╭─ {spec.name} Playground · {ep.name} ({ep.effective_quant or '—'} · {ad}) ─╮")
    print(f"│ endpoint : {ep.base_url}  model={ep.model or ep.name}")
    print(f"│ mode     : {mode}" + ("" if argv else f"  (install native CLI: {spec.install_hint})"))
    print("╰────────────────────────────────────────────────────────────────╯\n", flush=True)
    if argv:
        print("$ " + " ".join(shlex.quote(x) for x in argv), flush=True)
        return subprocess.call(argv, env=env)
    return _builtin_repl(spec, ep)


def _builtin_repl(spec: HarnessSpec, ep: config.Endpoint) -> int:
    """Zero-dependency interactive chat/tool playground framed with the harness's real system prompt."""
    import json
    import urllib.error
    import urllib.request

    sys_prompt, _src = load_system_prompt(spec)
    if not sys_prompt:
        sys_prompt = (
            f"You are an AI assistant running inside the {spec.name} harness.\n"
            f"Available tools: {', '.join(spec.tools)}.\n"
            f"Edit strategy: {spec.edit_strategy}.\n"
            f"Code mode: {spec.code_mode}.\n"
            "Be concise, precise, and show your reasoning clearly."
        )
    history: list[dict] = [{"role": "system", "content": sys_prompt}]
    key = os.environ.get(ep.api_key_env, "") if ep.api_key_env else ""
    url = ep.base_url.rstrip("/") + "/chat/completions"
    model = ep.model or ep.name or "default"
    print("Type a prompt (commands: /reset, /system, /quit):\n")
    while True:
        try:
            user_in = input(f"[{spec.id}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_in:
            continue
        if user_in in ("/quit", "/exit", "exit", "quit"):
            return 0
        if user_in == "/reset":
            history = [history[0]]
            print("↺ conversation cleared\n")
            continue
        if user_in == "/system":
            print(sys_prompt + "\n")
            continue
        history.append({"role": "user", "content": user_in})
        body = json.dumps({"model": model, "messages": history, "max_tokens": 2048, "temperature": 0.2}).encode()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read().decode("utf-8", errors="replace"))
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            reason = str(msg.get("reasoning_content") or "").strip()
            content = str(msg.get("content") or "").strip()
            usage = data.get("usage") or {}
            if reason:
                print(f"\n[reasoning]\n{reason}\n")
            print(f"\n{content}\n")
            if usage:
                print(f"  ({usage.get('prompt_tokens', '?')} in · {usage.get('completion_tokens', '?')} out tokens)\n")
            history.append({"role": "assistant", "content": content})
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"✗ {e}\n")
            history.pop()
