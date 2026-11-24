"""Settings and the endpoints to evaluate, kept in ~/.config/gbench-tui/config.json.

An *endpoint* is what gbench calls `--remote-endpoint`: any OpenAI-compatible `/v1` server. It may also name a local
GGUF file, in which case gbench-tui starts `llama-server` for it on demand (and stops it when you ask, or on exit).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


def config_dir() -> Path:
    d = Path(os.environ.get("GBTUI_HOME") or Path.home() / ".config" / "gbench-tui")
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    """Data shared with the run sandboxes (HF cache, in-flight results). Mounted into each sandbox."""
    d = Path(os.environ.get("GBTUI_CACHE") or Path.home() / ".cache" / "gbench-tui")
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Endpoint:
    name: str
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = ""                 # --models: the id the endpoint serves (or its HF id)
    tokenizer: str = ""             # --tokenizer: HF id used by gbench to size prompts / render chat templates
    api_key_env: str = ""           # name of the environment variable holding the key (never the key itself)
    gguf: str = ""                  # optional local .gguf: gbench-tui starts llama-server for it
    ctx: int = 16384                # llama-server context (total, shared by the slots)
    parallel: int = 2               # llama-server slots (concurrent requests)
    extra_args: str = ""            # extra llama-server arguments
    notes: str = ""

    @property
    def port(self) -> int:
        m = re.search(r":(\d+)(/|$)", self.base_url.split("//", 1)[-1])
        return int(m.group(1)) if m else 80

    @property
    def is_local(self) -> bool:
        host = self.base_url.split("//", 1)[-1].split("/", 1)[0].split(":")[0]
        return host in ("127.0.0.1", "localhost", "0.0.0.0", "::1")


@dataclass
class RunDefaults:
    tier: str = "golden"            # golden | evals | perf
    evals: list = field(default_factory=lambda: ["gsm8k", "ifeval"])
    golden_tasks: list = field(default_factory=list)   # empty = all
    eval_limit: int = 20
    max_output_tokens: int = 4096
    thinking: bool = False
    temperature: str = ""           # "" = gbench's default
    concurrency: int = 2
    suite_timeout: int = 3600
    perf_mode: str = "personal"     # personal | quick


@dataclass
class Settings:
    endpoints: list = field(default_factory=list)
    active: str = ""                 # name of the endpoint being evaluated
    sandbox_memory_gb: int = 6       # RAM cap of each run sandbox (microVM)
    sandbox_cpus: int = 0            # 0 = auto (half the cores, at most 6)
    min_free_gb: float = 4.0         # refuse to start a sandbox / llama-server if the host would drop below this
    results_dir: str = ""            # "" = ~/.local/share/gbench-tui/results
    stop_server_on_exit: bool = True
    run: RunDefaults = field(default_factory=RunDefaults)

    # ---- helpers ----
    def endpoint(self, name: str | None = None) -> Endpoint | None:
        name = name or self.active
        return next((e for e in self.endpoints if e.name == name), None)

    @property
    def results_path(self) -> Path:
        p = Path(self.results_dir).expanduser() if self.results_dir else Path.home() / ".local" / "share" / "gbench-tui" / "results"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def cpus(self) -> int:
        return self.sandbox_cpus or max(2, min(6, (os.cpu_count() or 4) // 2))


def _path() -> Path:
    return config_dir() / "config.json"


def default_endpoints() -> list:
    """A sensible first endpoint: the LFM3 GGUF in the home directory if there is one, else a local llama-server."""
    gguf = next(iter(sorted(Path.home().glob("LFM3*Q8_0.gguf"))), None) or next(iter(sorted(Path.home().glob("*.gguf"))), None)
    if gguf and "LFM3" in gguf.name:
        return [Endpoint(name="lfm3-1.7b", model="lfm3-1.7b", tokenizer="LiquidAI/LFM3-1.7B", gguf=str(gguf))]
    return [Endpoint(name="local-llama", model="", tokenizer="")]


def load() -> Settings:
    try:
        raw = json.loads(_path().read_text())
    except (OSError, ValueError):
        s = Settings(endpoints=default_endpoints())
        s.active = s.endpoints[0].name if s.endpoints else ""
        save(s)
        return s
    eps = [Endpoint(**{k: v for k, v in e.items() if k in Endpoint.__dataclass_fields__}) for e in raw.pop("endpoints", [])]
    run = RunDefaults(**{k: v for k, v in raw.pop("run", {}).items() if k in RunDefaults.__dataclass_fields__})
    s = Settings(endpoints=eps, run=run, **{k: v for k, v in raw.items() if k in Settings.__dataclass_fields__})
    if not s.active and s.endpoints:
        s.active = s.endpoints[0].name
    return s


def save(s: Settings) -> None:
    p = _path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(s), indent=2))
    tmp.replace(p)
