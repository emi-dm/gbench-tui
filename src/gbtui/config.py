"""Settings and the endpoints to evaluate, kept in ~/.config/gbench-tui/config.json.

An *endpoint* is what gbench calls `--remote-endpoint`: any OpenAI-compatible `/v1` server. It may also name a local
GGUF file (and optional LoRA GGUF adapter, e.g. from Hugging Face's `harness-hub`), in which case gbench-tui starts
`llama-server` for it on demand (and stops it when you ask, or on exit).
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path

_QUANT_RE = re.compile(
    r"(?:^|(?<=[\s._:/@(-]))("
    r"UD[-_](?:IQ[1-4][_-](?:XXS|XS|NL|[SML])|Q[2-8][_-](?:K(?:[_-](?:XXS|XS|XL|[SML]))?|[01]|NL))"
    r"|IQ[1-4][_-](?:XXS|XS|NL|[SML])"
    r"|TQ[12][_-][0-9]"
    r"|Q[2-8][_-]K(?:[_-](?:XXS|XS|XL|[SML]))?"
    r"|Q[458][_-][01]"
    r"|Q[2-8]_NL"
    r"|MXFP[48]|NVFP4"
    r"|BF16|FP32|F32|FP16|F16|FP8(?:[_-]E[45]M[23])?|FP4"
    r"|INT[48]|W[248]A(?:4|8|16)"
    r"|AWQ|GPTQ|EXL2"
    r"|[23468]BIT"
    r")(?=$|[\s._:/@)-])",
    re.IGNORECASE,
)

_QUANT_BITS: dict[str, float] = {
    "IQ1_S": 1.56, "TQ1_0": 1.70, "IQ1_M": 1.75,
    "2BIT": 2.00, "IQ2_XXS": 2.06, "TQ2_0": 2.10, "IQ2_XS": 2.31,
    "IQ2_S": 2.50, "Q2_K_S": 2.50, "Q2_K": 2.63, "IQ2_M": 2.70, "Q2_K_L": 2.90,
    "3BIT": 3.00, "IQ3_XXS": 3.06, "IQ3_XS": 3.30, "IQ3_S": 3.50, "Q3_K_S": 3.50,
    "IQ3_M": 3.70, "Q3_K_S": 3.50, "Q3_K": 3.80, "Q3_K_M": 3.91, "Q3_K_L": 4.27, "Q3_K_XL": 4.40,
    "4BIT": 4.00, "INT4": 4.00, "FP4": 4.00, "W4A16": 4.00, "AWQ": 4.00, "GPTQ": 4.00,
    "IQ4_XS": 4.25, "MXFP4": 4.25, "NVFP4": 4.25, "IQ4_NL": 4.50, "Q4_0": 4.50,
    "Q4_K_S": 4.58, "Q4_K": 4.80, "Q4_K_M": 4.85, "Q4_1": 5.00, "Q4_K_L": 5.00, "Q4_K_XL": 5.10,
    "Q5_0": 5.50, "Q5_K_S": 5.54, "Q5_K": 5.65, "Q5_K_M": 5.69, "Q5_K_L": 5.80, "Q5_1": 6.00,
    "6BIT": 6.00, "Q6_K": 6.56, "Q6_K_L": 6.70, "Q6_K_XL": 6.80,
    "8BIT": 8.00, "INT8": 8.00, "FP8": 8.00, "FP8_E4M3": 8.00, "FP8_E5M2": 8.00,
    "W8A8": 8.00, "W8A16": 8.00, "MXFP8": 8.25, "Q8_0": 8.50, "Q8_K": 8.50, "Q8_K_XL": 8.70,
    "F16": 16.0, "FP16": 16.0, "BF16": 16.0, "F32": 32.0, "FP32": 32.0,
}

# Known agent harnesses (ordered longest-first so compound names match before substrings)
KNOWN_HARNESSES = ("super-adapter", "claude-code", "openhands", "opencode", "hermes", "codex", "aider", "pi")
_HARNESS_RE = re.compile(
    r"(?:^|(?<=[\s._:/@(-]))("
    r"super[-_]?adapter|claude[-_]?code|open[-_]?hands|open[-_]?code|hermes|codex|aider|pi"
    r")(?=$|[\s._:/@)-]|[-_](?:lora|adapted|gguf|agent))",
    re.IGNORECASE,
)
_LORA_SUFFIX_RE = re.compile(
    r"(?:[-_](?:harbor|think|thinking))?(?:[-_](?:super[-_]?adapter|claude[-_]?code|open[-_]?hands|open[-_]?code|hermes|codex|aider|pi))"
    r"(?:[-_](?:adapted|lora|agent|gguf))*(?:[-_](?:f16|f32|bf16|q8_0))?$",
    re.IGNORECASE,
)
HARNESS_MIN_CTX = 32768


def normalize_harness(h: str) -> str:
    """Canonical lowercase harness name: 'OpenCode' -> 'opencode', 'super_adapter' -> 'super-adapter'."""
    s = str(h or "").strip().lower().replace("_", "-")
    if not s:
        return ""
    aliases = {
        "superadapter": "super-adapter",
        "super-adapter": "super-adapter",
        "claudecode": "claude-code",
        "claude-code": "claude-code",
        "open-code": "opencode",
        "opencode": "opencode",
        "open-hands": "openhands",
        "openhands": "openhands",
        "hermes-agent": "hermes",
        "hermes": "hermes",
        "pi-agent": "pi",
        "pi": "pi",
        "codex": "codex",
        "aider": "aider",
    }
    return aliases.get(s, s)


def detect_harness(text: str) -> str:
    """Detect a harness identifier (pi, opencode, hermes, super-adapter, …) from a filename, repo or endpoint name."""
    s = str(text or "").strip()
    if not s:
        return ""
    if s.lower().endswith(".gguf"):
        s = Path(s).stem
    elif "/" in s:
        s = s.rsplit("/", 1)[-1]
    m = _HARNESS_RE.search(s)
    return normalize_harness(m.group(1)) if m else ""


def normalize_adapter(adapter: str) -> str:
    """Canonical lowercase adapter label (e.g. 'pi-lora', 'opencode-lora', 'super-adapter'; '' for base)."""
    s = str(adapter or "").strip()
    if not s or s.lower() in ("base", "none", "-", "—"):
        return ""
    low = s.lower().replace("_", "-")
    if low in ("super-adapter", "superadapter", "super-adapter-lora"):
        return "super-adapter"
    h = normalize_harness(low.removesuffix("-lora").removesuffix("-adapted"))
    if h in KNOWN_HARNESSES and h != "super-adapter":
        return f"{h}-lora"
    return re.sub(r"[^a-z0-9._-]+", "-", low).strip("-")


def detect_adapter(lora_path: str = "", model: str = "", name: str = "", target_harness: str = "") -> str:
    """Auto-detect canonical adapter label from LoRA path / harness / model / endpoint name; '' means base model."""
    h = normalize_harness(target_harness) or detect_harness(lora_path)
    if h == "super-adapter":
        return "super-adapter"
    if h:
        return f"{h}-lora"
    if lora_path:
        stem = Path(str(lora_path)).stem
        # Strip trailing precision tags like -F16 / -F32 / -BF16
        cleaned = re.sub(r"[-_](?:f16|fp16|f32|fp32|bf16|q8_0)$", "", stem, flags=re.IGNORECASE)
        h2 = detect_harness(cleaned)
        if h2 == "super-adapter":
            return "super-adapter"
        if h2:
            return f"{h2}-lora"
        return normalize_adapter(cleaned)
    # Check explicit lora/adapted/super-adapter mentions in model or name (avoid false positives on bare names)
    for cand in (model, name):
        s = str(cand or "")
        if re.search(r"lora|adapted|super[-_]?adapter", s, re.IGNORECASE):
            h3 = detect_harness(s)
            if h3 == "super-adapter":
                return "super-adapter"
            if h3:
                return f"{h3}-lora"
    return ""


def normalize_quant(q: str) -> str:
    """Canonical uppercase quantization tag, e.g. 'q4-k-m' -> 'Q4_K_M', 'ud_q4_k_xl' -> 'UD-Q4_K_XL'."""
    s = str(q or "").strip().upper()
    if not s:
        return ""
    if s.startswith(("UD-", "UD_")):
        return "UD-" + s[3:].replace("-", "_")
    return s.replace("-", "_")


def detect_quant(text: str) -> str:
    """Detect a quantization tag from a GGUF path, model id, or endpoint name; '' if none."""
    s = str(text or "").strip()
    if not s:
        return ""
    if s.lower().endswith(".gguf"):
        s = Path(s).stem
    elif "/" in s and "." in s.rsplit("/", 1)[-1]:
        s = s.rsplit("/", 1)[-1]
    m = _QUANT_RE.search(s)
    return normalize_quant(m.group(1)) if m else ""


def split_model_quant(text: str, explicit_quant: str = "") -> tuple[str, str]:
    """Return (base_model, canonical_quant) with any embedded quant token stripped from base_model."""
    raw = str(text or "").strip()
    if raw.lower().endswith(".gguf"):
        raw = Path(raw).stem
    raw = re.sub(r"[-_]\d{5}-of-\d{5}$", "", raw)
    q = normalize_quant(explicit_quant) or detect_quant(raw)
    if not raw:
        return ("", q)
    base = raw
    if q:
        # Remove parenthesized quant like 'lfm3-1.7b (Q8_0)' first
        pat = re.escape(q).replace("_", "[-_]").replace(r"\-", "[-_]")
        base = re.sub(rf"\s*\(\s*{pat}\s*\)\s*", " ", base, flags=re.IGNORECASE)
        base = re.sub(rf"(?:^|[\s._:/@-]){pat}(?=$|[\s._:/@-])", "", base, count=1, flags=re.IGNORECASE)
        base = re.sub(r"[\s._:@-]+$", "", base).strip()
        base = re.sub(r"^[\s._:@-]+", "", base).strip()
    return (base or raw, q)


def strip_adapter_suffix(base: str, adapter: str = "", harness: str = "") -> str:
    """Strip trailing adapter / harness tokens (e.g. '-pi-lora', '-harbor-opencode') from a base model name."""
    s = str(base or "").strip()
    if not s or (not adapter and not harness):
        return s
    cleaned = _LORA_SUFFIX_RE.sub("", s).strip(" -_.:")
    if cleaned and cleaned != s:
        return cleaned
    for tok in filter(None, (adapter, harness)):
        pat = re.escape(tok).replace(r"\-", "[-_]")
        cleaned = re.sub(rf"(?:[-_\s]+(?:harbor[-_\s]+)?){pat}(?:[-_\s]+(?:lora|adapted))?$", "", s, flags=re.IGNORECASE)
        cleaned = cleaned.strip(" -_.:")
        if cleaned and cleaned != s:
            return cleaned
    return s


def quant_bits(quant: str) -> float | None:
    """Approximate effective bits-per-weight for ordering & quantization charts."""
    q = normalize_quant(quant)
    if not q:
        return None
    if q in _QUANT_BITS:
        return _QUANT_BITS[q]
    if q.startswith("UD-"):
        inner = _QUANT_BITS.get(q[3:])
        if inner is not None:
            return round(inner + 0.05, 2)
    m = re.match(r"(?:UD-)?(?:I|T)?Q([1-8])", q)
    if m:
        return float(m.group(1)) + 0.5
    m = re.search(r"([23468])BIT", q)
    if m:
        return float(m.group(1))
    return None


def config_dir() -> Path:
    d = Path(os.environ.get("GBTUI_HOME") or Path.home() / ".config" / "gbench-tui")
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    """Data shared with the run containers (HF cache, LoRAs, in-flight results)."""
    d = Path(os.environ.get("GBTUI_CACHE") or Path.home() / ".cache" / "gbench-tui")
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Endpoint:
    name: str
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = ""                 # --models: the id the endpoint serves (or its HF id)
    tokenizer: str = ""             # --tokenizer: HF id used by gbench to size prompts / render chat templates
    quant: str = ""                 # quantization tag (e.g. Q4_K_M, Q8_0, BF16, FP8); auto-detected if empty
    api_key_env: str = ""           # name of the environment variable holding the key (never the key itself)
    gguf: str = ""                  # optional local .gguf: gbench-tui starts llama-server for it
    lora: str = ""                  # optional local LoRA .gguf (e.g. from harness-hub)
    lora_scale: float = 1.0         # LoRA scale passed via --lora-scaled when != 1.0
    adapter_name: str = ""          # explicit adapter tag (e.g. pi-lora, opencode-lora, super-adapter); auto if empty
    target_harness: str = ""        # target harness (pi, opencode, hermes, super-adapter, …); auto if empty
    ctx: int = 16384                # llama-server context (total, shared by the slots)
    parallel: int = 2               # llama-server slots (concurrent requests)
    extra_args: str = ""            # extra llama-server arguments
    notes: str = ""

    @property
    def effective_quant(self) -> str:
        return (normalize_quant(self.quant)
                or detect_quant(self.gguf)
                or detect_quant(self.model)
                or detect_quant(self.name))

    @property
    def effective_harness(self) -> str:
        return (normalize_harness(self.target_harness)
                or detect_harness(self.lora)
                or detect_harness(self.adapter_name)
                or (detect_harness(self.model) if re.search(r"lora|adapted|harbor|super", self.model or "", re.I) else "")
                or (detect_harness(self.name) if re.search(r"lora|adapted|harbor|super", self.name or "", re.I) else ""))

    @property
    def effective_adapter(self) -> str:
        return (normalize_adapter(self.adapter_name)
                or detect_adapter(self.lora, self.model, self.name, self.effective_harness))

    @property
    def base_model(self) -> str:
        raw = self.model or self.name or (Path(self.gguf).stem if self.gguf else "")
        base = split_model_quant(raw, self.effective_quant)[0]
        return strip_adapter_suffix(base, self.effective_adapter, self.effective_harness)

    @property
    def ctx_warning(self) -> str:
        if self.gguf and (self.lora or self.effective_harness) and self.ctx < HARNESS_MIN_CTX:
            return (f"ctx {self.ctx} is low for agent harnesses "
                    f"(recommend {HARNESS_MIN_CTX}–65536 for Pi / OpenCode / Hermes)")
        return ""

    @property
    def port(self) -> int:
        p = urllib.parse.urlparse(self.base_url if "://" in self.base_url else "http://" + self.base_url)
        try:
            if p.port is not None:
                return p.port
        except ValueError:
            pass
        return 443 if p.scheme == "https" else 80

    @property
    def is_local(self) -> bool:
        p = urllib.parse.urlparse(self.base_url if "://" in self.base_url else "http://" + self.base_url)
        return (p.hostname or "") in ("127.0.0.1", "localhost", "0.0.0.0", "::1")


@dataclass
class RunDefaults:
    tier: str = "golden"            # golden | evals | perf | preflight | harness
    evals: list = field(default_factory=lambda: ["gsm8k", "ifeval"])
    golden_tasks: list = field(default_factory=list)   # empty = all
    eval_limit: int = 20
    max_output_tokens: int = 4096
    thinking: bool = False
    temperature: str = ""           # "" = gbench's default
    concurrency: int = 2
    suite_timeout: int = 3600
    perf_mode: str = "personal"     # personal | quick
    harness: str = "pi"             # Harbor agent: pi | opencode | hermes | claude-code | openhands | aider
    harness_dataset: str = "terminal-bench@2.0"
    harness_tasks: list = field(default_factory=list)  # optional --include-task-name patterns
    harness_limit: int = 2          # --n-tasks for local Harbor runs
    tool_mode: str = "normal"       # normal | code_mode


# Built-in reusable evaluation presets (also extensible via Settings.presets in config.json)
PRESETS: dict[str, dict] = {
    "smoke": {
        "label": "Smoke (3 min)",
        "desc": "Golden Set + GSM8K(10) + IFEval(10) — fast sanity check",
        "steps": [
            {"tier": "golden"},
            {"tier": "evals", "evals": ["gsm8k", "ifeval"], "eval_limit": 10},
        ],
    },
    "core": {
        "label": "Core ★",
        "desc": "Golden Set + GSM8K(20) + IFEval(20) + MMLU-Pro(20) + perf",
        "steps": [
            {"tier": "golden"},
            {"tier": "evals", "evals": ["gsm8k", "ifeval", "mmlu_pro"], "eval_limit": 20},
            {"tier": "perf", "perf_mode": "personal"},
        ],
    },
    "coding": {
        "label": "Coding & Math",
        "desc": "Golden Set + CRUXEval(20) + GSM8K(20)",
        "steps": [
            {"tier": "golden"},
            {"tier": "evals", "evals": ["cruxeval", "gsm8k"], "eval_limit": 20},
        ],
    },
    "quant-shootout": {
        "label": "Quant Shootout",
        "desc": "Ideal for Quant Sweeps: Golden + GSM8K(20) + IFEval(20) + perf",
        "steps": [
            {"tier": "golden"},
            {"tier": "evals", "evals": ["gsm8k", "ifeval"], "eval_limit": 20},
            {"tier": "perf", "perf_mode": "personal"},
        ],
    },
    "harness-smoke": {
        "label": "Harness Smoke",
        "desc": "Harbor Pi + OpenCode + Hermes (1 task each on terminal-bench@2.0)",
        "steps": [
            {"tier": "harness", "harness": "pi", "harness_dataset": "terminal-bench@2.0", "eval_limit": 1},
            {"tier": "harness", "harness": "opencode", "harness_dataset": "terminal-bench@2.0", "eval_limit": 1},
            {"tier": "harness", "harness": "hermes", "harness_dataset": "terminal-bench@2.0", "eval_limit": 1},
        ],
    },
}


@dataclass
class Settings:
    endpoints: list = field(default_factory=list)
    active: str = ""                 # name of the endpoint being evaluated
    sandbox_memory_gb: int = 4       # RAM cap of each run's Docker container
    sandbox_cpus: int = 0            # 0 = auto (half the cores, at most 6)
    min_free_gb: float = 4.0         # refuse to start a container / llama-server if the host would drop below this
    results_dir: str = ""            # "" = ~/.local/share/gbench-tui/results
    stop_server_on_exit: bool = True
    notify_on_finish: bool = True    # bell + OS notification when a run / sweep / matrix ends
    theme: str = "aurora"            # one of theme.THEME_ORDER
    presets: dict = field(default_factory=dict)  # user-saved custom presets
    run: RunDefaults = field(default_factory=RunDefaults)

    def all_presets(self) -> dict[str, dict]:
        merged = {k: dict(v) for k, v in PRESETS.items()}
        if isinstance(self.presets, dict):
            for k, v in self.presets.items():
                if isinstance(v, dict) and v.get("steps"):
                    merged[k] = v
        return merged

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
    gguf = next(iter(sorted(Path.home().glob("LFM3*Q8_0.gguf"))), None) or next(
        iter(sorted(p for p in Path.home().glob("*.gguf") if "lora" not in p.name.lower())), None)
    if gguf and "LFM3" in gguf.name:
        base, quant = split_model_quant(gguf.stem)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", base.lower()).strip(".") or "lfm3-1.7b"
        name = f"{slug}-{quant.lower()}" if quant else slug
        return [Endpoint(name=name, model=name, tokenizer=f"LiquidAI/{base}", quant=quant, gguf=str(gguf))]
    return [Endpoint(name="local-llama", model="", tokenizer="")]


def load() -> Settings:
    p = _path()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        s = Settings(endpoints=default_endpoints())
        s.active = s.endpoints[0].name if s.endpoints else ""
        save(s)
        return s
    except (OSError, ValueError):
        # Never clobber an unreadable/corrupt user config.json; fall back in memory.
        s = Settings(endpoints=default_endpoints())
        s.active = s.endpoints[0].name if s.endpoints else ""
        return s
    if not isinstance(raw, dict):
        s = Settings(endpoints=default_endpoints())
        s.active = s.endpoints[0].name if s.endpoints else ""
        return s
    raw_eps = raw.pop("endpoints", [])
    eps = []
    if isinstance(raw_eps, list):
        for e in raw_eps:
            if isinstance(e, dict) and e.get("name"):
                eps.append(Endpoint(**{k: v for k, v in e.items() if k in Endpoint.__dataclass_fields__}))
    raw_run = raw.pop("run", {})
    run = RunDefaults(**{k: v for k, v in raw_run.items() if k in RunDefaults.__dataclass_fields__}) if isinstance(raw_run, dict) else RunDefaults()
    s = Settings(endpoints=eps, run=run, **{k: v for k, v in raw.items() if k in Settings.__dataclass_fields__})
    if s.endpoints and s.active not in {e.name for e in s.endpoints}:
        s.active = s.endpoints[0].name
    elif not s.endpoints:
        s.active = ""
    return s


def save(s: Settings) -> None:
    p = _path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")
    tmp.replace(p)
