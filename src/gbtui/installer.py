"""Modular setup & dependency installer (`gbt setup` and TUI Sandbox → `i`).

Installs everything needed for the user's chosen workflow so the only optional manual tool on the host is `llama.cpp`
(when serving local `.gguf` models):

  Profiles:
    - `llm`    : Hardened Docker image (`gbtui-gbench:v3`) + gbench eval catalog (Golden Set + ~90 suites)
    - `agents` : Harbor runner (`uv tool install harbor`) + optional local harness CLIs (`pi`, `opencode`, `hermes`, `aider`)
    - `full`   : Everything (`llm` + `agents`)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass

from . import catalog, harbor_runner, metrics, sandbox


@dataclass(frozen=True)
class Component:
    id: str
    label: str
    group: str          # llm | agents | optional
    desc: str
    requires: str = ""  # docker | uv_or_pip | npm


COMPONENTS: tuple[Component, ...] = (
    Component("gbench_docker", "gbench Docker image + catalog", "llm",
              "Builds hardened gbtui-gbench:v3 container + ~90 eval suites catalog", "docker"),
    Component("harbor", "Harbor agent runner", "agents",
              "Runs coding agent harnesses (Pi, OpenCode, Hermes…) on Terminal-Bench / SWE-bench", "uv_or_pip"),
    Component("pi", "Pi harness CLI", "agents",
              "Minimal coding agent (@earendil-works/pi-coding-agent) for interactive Play", "npm"),
    Component("opencode", "OpenCode harness CLI", "agents",
              "Terminal coding agent (opencode-ai) with snapshots & undo", "npm"),
    Component("hermes", "Hermes Agent CLI", "agents",
              "Nous Research agent harness with toolsets & Code Mode", "uv_or_pip"),
    Component("aider", "Aider harness CLI", "optional",
              "Git-native coding assistant (aider-chat) for interactive Play", "uv_or_pip"),
    Component("claude-code", "Claude Code CLI", "optional",
              "Anthropic coding agent (@anthropic-ai/claude-code)", "npm"),
)

PROFILES: dict[str, dict] = {
    "llm": {
        "label": "LLMs only (gbench in Docker)",
        "desc": "Builds the hardened Docker image + catalog for Golden Set, ~90 evals & perf",
        "components": ["gbench_docker"],
    },
    "agents": {
        "label": "Agentic evals & harnesses (Harbor + Pi/OpenCode/Hermes)",
        "desc": "Installs Harbor + local harness CLIs for Terminal-Bench, SWE-bench & Play",
        "components": ["harbor", "pi", "opencode", "hermes"],
    },
    "full": {
        "label": "Full suite (LLMs + Agents + Harnesses)",
        "desc": "Installs Docker gbench image, Harbor, and Pi / OpenCode / Hermes CLIs",
        "components": ["gbench_docker", "harbor", "pi", "opencode", "hermes"],
    },
}


def component_installed(comp_id: str) -> bool:
    if comp_id == "gbench_docker":
        return sandbox.status() is None
    if comp_id == "harbor":
        return harbor_runner.installed()
    if comp_id == "pi":
        return shutil.which("pi") is not None
    if comp_id == "opencode":
        return shutil.which("opencode") is not None
    if comp_id == "hermes":
        return shutil.which("hermes") is not None or shutil.which("hermes-agent") is not None
    if comp_id == "aider":
        return shutil.which("aider") is not None
    if comp_id == "claude-code":
        return shutil.which("claude") is not None
    return False


def inspect_status() -> dict[str, dict]:
    """Returns installation status of all setup components + host prerequisites."""
    out: dict[str, dict] = {}
    for c in COMPONENTS:
        out[c.id] = {"component": c, "installed": component_installed(c.id)}
    out["_prereqs"] = {
        "docker": bool(shutil.which("docker")),
        "uv": bool(shutil.which("uv")),
        "npm": bool(shutil.which("npm")),
        "llama_server": bool(shutil.which("llama-server")),
    }
    return out


def _run_cmd(cmd: list[str], on_line) -> bool:
    on_line("$ " + " ".join(cmd))
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             errors="replace", env=sandbox._env())
        assert p.stdout is not None
        for ln in p.stdout:
            s = ln.rstrip("\r\n")
            if s:
                on_line(s)
        return p.wait() == 0
    except OSError as e:
        on_line(f"✗ failed to run {cmd[0]}: {e}")
        return False


def _install_python_tool(pkg: str, on_line, git_fallback: str = "") -> bool:
    uv = shutil.which("uv")
    if uv:
        if _run_cmd([uv, "tool", "install", "--upgrade", pkg], on_line):
            return True
        if git_fallback:
            return _run_cmd([uv, "tool", "install", "--upgrade", git_fallback], on_line)
        return False
    pipx = shutil.which("pipx")
    if pipx:
        return _run_cmd([pipx, "install", "--force", pkg], on_line)
    return _run_cmd([sys.executable, "-m", "pip", "install", "--user", "--upgrade", pkg], on_line)


def _install_npm_tool(pkg: str, on_line) -> bool:
    npm = shutil.which("npm")
    if not npm:
        on_line(f"⚠ npm not found on PATH — install Node.js to auto-install {pkg} (or run via Harbor Docker)")
        return False
    return _run_cmd([npm, "install", "-g", pkg], on_line)


def install_component(comp_id: str, on_line=print, cpus: int = 4) -> bool:
    """Installs a single modular setup component, streaming progress to `on_line`."""
    if comp_id == "gbench_docker":
        on_line("━━━ [LLM] Building hardened gbench Docker image & catalog ━━━")
        ok = sandbox.setup(on_line, cpus=cpus)
        if ok:
            catalog.reload()
            metrics.reload()
        return ok
    if comp_id == "harbor":
        on_line("━━━ [Agents] Installing Harbor agent benchmark runner ━━━")
        return _install_python_tool("harbor", on_line)
    if comp_id == "pi":
        on_line("━━━ [Harness] Installing Pi coding agent CLI ━━━")
        return _install_npm_tool("@earendil-works/pi-coding-agent", on_line)
    if comp_id == "opencode":
        on_line("━━━ [Harness] Installing OpenCode CLI ━━━")
        return _install_npm_tool("opencode-ai", on_line)
    if comp_id == "hermes":
        on_line("━━━ [Harness] Installing Hermes Agent CLI ━━━")
        ok = _install_python_tool("hermes-agent", on_line, git_fallback="git+https://github.com/NousResearch/hermes-agent.git")
        if not ok:
            on_line("⚠ Hermes CLI install failed — you can still use the built-in Hermes playground (`gbt play --harness hermes`)")
        return ok
    if comp_id == "aider":
        on_line("━━━ [Harness] Installing Aider CLI ━━━")
        return _install_python_tool("aider-chat", on_line)
    if comp_id == "claude-code":
        on_line("━━━ [Harness] Installing Claude Code CLI ━━━")
        return _install_npm_tool("@anthropic-ai/claude-code", on_line)
    on_line(f"✗ unknown component: {comp_id}")
    return False


def run_setup(component_ids: list[str], on_line=print, cpus: int = 4) -> bool:
    """Runs modular setup for the selected list of components and summarizes what succeeded."""
    all_ok = True
    done: list[str] = []
    failed: list[str] = []
    for cid in component_ids:
        ok = install_component(cid, on_line=on_line, cpus=cpus)
        if ok:
            done.append(cid)
        else:
            failed.append(cid)
            all_ok = False
    on_line("")
    if done:
        on_line(f"✔ installed / ready: {', '.join(done)}")
    if failed:
        on_line(f"⚠ skipped / failed: {', '.join(failed)}")
    if not shutil.which("llama-server"):
        on_line("ℹ optional: install llama.cpp (`brew install llama.cpp`) only if you want to serve local .gguf files")
    return all_ok


def format_status_plain() -> str:
    st = inspect_status()
    pr = st["_prereqs"]
    lines = [
        "gbt setup — Modular Environment Status",
        f"  host tools: docker={'✔' if pr['docker'] else '✗'}  uv={'✔' if pr['uv'] else '✗'}  "
        f"npm={'✔' if pr['npm'] else '○'}  llama-server={'✔' if pr['llama_server'] else '○ (optional for local GGUFs)'}",
        "",
    ]
    for c in COMPONENTS:
        ok = st[c.id]["installed"]
        mark = "✔ ready  " if ok else "○ missing"
        lines.append(f"  {mark}  {c.id:<15} [{c.group:<8}] {c.label} — {c.desc}")
    lines += [
        "",
        "Profiles:",
        "  gbt setup --profile llm     # hardened Docker gbench + ~90 suites",
        "  gbt setup --profile agents  # Harbor + Pi, OpenCode, Hermes",
        "  gbt setup --profile full    # everything (LLMs + Agents + Harnesses)",
    ]
    return "\n".join(lines)
