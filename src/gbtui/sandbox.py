"""Where gbench runs: always inside an ephemeral, hardened Docker container, never on the host.

    host: gbench-tui (curses, stdlib only) ── llama-server (Metal/CUDA) or any /v1 endpoint
            │ docker exec                                ▲
            ▼                                            │ host.docker.internal:<port>
    docker  gbtui-run-<pid>-<hex>  ← from image gbtui-gbench:v<N>, RAM/CPU/PID capped
            ├─ /opt/gbench (venv): gbench + its eval extras, the backend that does all the work
            ├─ /work  ← ~/.cache/gbench-tui/work (HF cache + run results; never mounts $HOME or docker.sock)
            └─ hardened: --read-only, --cap-drop=ALL, no-new-privileges, non-root UID:GID, --network=none when offline

`setup()` builds the Docker image once from a pinned checkout of gbench, dumps gbench's own suite / Golden Set catalog
to JSON (inside an offline container), and removes any older `gbtui-gbench:v*` image. Every run then gets a fresh
container from the image, removed when the run ends (finished, failed or cancelled); one left behind by a process that
died is pruned automatically on the next run.
"""
from __future__ import annotations

import io
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import urllib.parse
from pathlib import Path

from .config import cache_dir, config_dir

VERSION = "3"
IMAGE = os.environ.get("GBTUI_IMAGE") or os.environ.get("GBTUI_TEMPLATE") or f"gbtui-gbench:v{VERSION}"
TEMPLATE = IMAGE                                # backwards-compatible alias
BASE_IMAGE = os.environ.get("GBTUI_BASE_IMAGE") or "python:3.12-slim"
PREFIX = "gbtui-run-"
GBENCH_REPO = "https://github.com/google-deepmind/gbench"
GBENCH_REF = os.environ.get("GBTUI_GBENCH_REF", "main")
VENV = "/opt/gbench"
PYTHON = "3.12"
CONTAINER_WORK = "/work"
# Host directory bind-mounted at CONTAINER_WORK inside each container (HF cache and in-flight results)
WORK = str(cache_dir() / "work")
MARKER = "/etc/gbtui-sandbox"
PIDS_LIMIT = 256
# Hosts a run typically reaches besides the endpoint: datasets and tokenizers
RUN_HOSTS = ["huggingface.co", "**.huggingface.co", "**.hf.co", "raw.githubusercontent.com", "github.com",
             "objects.githubusercontent.com", "release-assets.githubusercontent.com", "data.statmt.org"]
# Python packages installed next to gbench[evals]: the tokenizer stack (no torch) for --tokenizer
EXTRA_PIP = ["transformers<6", "jinja2"]
# gbench[evals] packages skipped on linux/arm64 (no wheel and no sdist; only gorilla_apibench imports them)
SKIP_EXTRAS_ARM64 = ["tree-sitter", "tree-sitter-python"]

_SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|AUTH", re.I)


def _env() -> dict:
    """The environment docker is started with: the user's, minus anything that looks like a secret."""
    return {k: v for k, v in os.environ.items() if not _SECRET.search(k)}


def docker(*args, timeout: float = 120, input: bytes | None = None, text: bool = True) -> subprocess.CompletedProcess:
    kw = {"input": input} if input is not None else {}
    return subprocess.run(["docker", *args], capture_output=True, text=text and input is None,
                          timeout=timeout, env=_env(), **kw)


def _out(r: subprocess.CompletedProcess) -> str:
    o = (r.stdout or "") + (r.stderr or "")
    return o.decode(errors="replace") if isinstance(o, bytes) else o


# ------------------------------------------------------------------------------------------------
# status
# ------------------------------------------------------------------------------------------------
def installed() -> bool:
    return shutil.which("docker") is not None


def status() -> str | None:
    """None when runs can start; otherwise a short reason."""
    if not installed():
        return "Docker is not installed (install Docker Desktop, OrbStack, Colima, or Docker Engine)"
    try:
        r = docker("info", "--format", "{{.ServerVersion}}", timeout=20)
    except (OSError, subprocess.SubprocessError) as e:
        return f"Docker not answering: {e}"
    if r.returncode != 0:
        out = _out(r).strip().splitlines()
        msg = out[-1][-160:] if out else "daemon not reachable"
        return f"Docker daemon not running (start Docker Desktop / OrbStack / Colima): {msg}"
    if not template_ready():
        return "Docker image not built yet (Sandbox → Build image, or `gbench-tui setup`)"
    return None


def templates() -> list[str]:
    """Local Docker images as `repository:tag`."""
    try:
        r = docker("images", "--format", "{{.Repository}}:{{.Tag}}", timeout=30)
        if r.returncode != 0:
            return []
        out = []
        for line in (r.stdout or "").splitlines():
            ref = line.strip()
            if ref and "<none>" not in ref:
                out.append(ref)
        return out
    except (OSError, subprocess.SubprocessError):
        return []


def _same(ref: str, want: str) -> bool:
    return ref == want or ref.endswith("/" + want)


def template_ready() -> bool:
    try:
        r = docker("image", "inspect", IMAGE, timeout=20)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _parse_labels(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in str(raw or "").split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def sandboxes() -> list[dict]:
    """Managed gbtui Docker containers (running or exited)."""
    try:
        r = docker("ps", "-a", "--filter", "label=gbtui.managed=1", "--format", "{{json .}}", timeout=30)
        if r.returncode != 0:
            return []
        out = []
        for line in (r.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            labels = _parse_labels(row.get("Labels", ""))
            name = str(row.get("Names") or "").lstrip("/")
            out.append({
                "name": name,
                "status": str(row.get("Status") or ""),
                "state": str(row.get("State") or "").lower(),
                "memory": labels.get("gbtui.memory", ""),
                "pid": labels.get("gbtui.pid", ""),
                "id": str(row.get("ID") or ""),
            })
        return out
    except (OSError, subprocess.SubprocessError):
        return []


# ------------------------------------------------------------------------------------------------
# lifecycle
# ------------------------------------------------------------------------------------------------
def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def prune() -> list[str]:
    """Removes run containers whose owner process is gone (a crash, kill -9, closed terminal) or that already exited."""
    gone = []
    for s in sandboxes():
        n = s.get("name") or ""
        m = re.match(re.escape(PREFIX) + r"(\d+)-", n)
        if not m:
            continue
        owner_dead = not _alive(int(m.group(1)))
        exited = s.get("state") in ("exited", "dead", "created")
        if owner_dead or exited:
            destroy(n)
            gone.append(n)
    return gone


def work_dir() -> Path:
    d = cache_dir() / "work"
    (d / "hf").mkdir(parents=True, exist_ok=True)
    (d / "runs").mkdir(parents=True, exist_ok=True)
    return d


def run_argv(name: str, memory_gb: int, cpus: int, allow: list[str]) -> list[str]:
    """Hardened `docker run -d` command line for an ephemeral gbench container."""
    cmd = [
        "docker", "run", "-d",
        "--name", name,
        "--label", "gbtui.managed=1",
        "--label", f"gbtui.pid={os.getpid()}",
        "--label", f"gbtui.memory={memory_gb}g",
        "--memory", f"{memory_gb}g",
        "--memory-swap", f"{memory_gb}g",
        "--cpus", str(cpus),
        "--pids-limit", str(PIDS_LIMIT),
        "--read-only",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g",
        "--cap-drop=ALL",
        "--security-opt", "no-new-privileges:true",
    ]
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
    cmd += [
        "-e", "HOME=/tmp",
        "-e", "TMPDIR=/tmp",
        "-e", "XDG_CACHE_HOME=/tmp/.cache",
        "-e", "XDG_CONFIG_HOME=/tmp/.config",
        # The hardened Docker container itself provides process, filesystem and resource isolation
        # (bwrap inside Docker would require weakening seccomp / user-namespace restrictions).
        "-e", "GBENCH_SANDBOX=none",
        "-v", f"{work_dir()}:{CONTAINER_WORK}:rw",
        "-w", CONTAINER_WORK,
    ]
    if not allow:
        cmd += ["--network", "none"]
    else:
        cmd += ["--add-host", f"{host_alias()}:host-gateway"]
    cmd += [IMAGE, "sleep", "infinity"]
    return cmd


def create(memory_gb: int, cpus: int, allow: list[str], on_line=None) -> str:
    """Starts an ephemeral, hardened Docker container from IMAGE. Raises RuntimeError on failure."""
    say = on_line or (lambda _m: None)
    if not template_ready():
        raise RuntimeError("Docker image not built yet (Sandbox → Build image, or `gbench-tui setup`)")
    prune()
    name = f"{PREFIX}{os.getpid()}-{secrets.token_hex(3)}"
    net_desc = "none (offline)" if not allow else "bridge (endpoint + datasets)"
    say(f"starting container {name} ({memory_gb} GB RAM, {cpus} CPUs, ro-rootfs, cap-drop=ALL, net: {net_desc})")
    argv = run_argv(name, memory_gb, cpus, allow)
    r = subprocess.run(argv, capture_output=True, text=True, timeout=120, env=_env())
    if r.returncode != 0:
        destroy(name)
        raise RuntimeError(f"could not start container {name}: " + _out(r).strip()[-300:])
    return name


def destroy(name: str) -> None:
    try:
        docker("rm", "--force", name, timeout=120)
    except (OSError, subprocess.SubprocessError):
        pass


def exec_argv(name: str, argv: list[str], env: dict | None = None, user: str | None = None) -> list[str]:
    """The host command line that runs argv inside the container (for Popen with streamed output)."""
    cmd = ["docker", "exec"]
    if user:
        cmd += ["-u", user]
    for k, v in (env or {}).items():
        cmd += ["-e", f"{k}={v}"]
    return cmd + [name, *argv]


def run_exec(name: str, argv: list[str], env: dict | None = None, timeout: float = 300) -> subprocess.CompletedProcess:
    """Runs argv inside a running container synchronously and returns the CompletedProcess."""
    return subprocess.run(exec_argv(name, argv, env=env), capture_output=True, text=True, timeout=timeout, env=_env())


def host_alias() -> str:
    """How a container names the host (where llama-server listens). `host.docker.internal` on macOS & Linux."""
    try:
        return json.loads((config_dir() / "sandbox.json").read_text()).get("host_alias") or "host.docker.internal"
    except (OSError, ValueError):
        return "host.docker.internal"


def sandbox_url(base_url: str) -> str:
    """The endpoint URL as seen from inside the container: localhost becomes the host alias."""
    return re.sub(r"//(127\.0\.0\.1|localhost|0\.0\.0\.0|\[::1\])(?=[:/]|$)", "//" + host_alias(), base_url.strip())


def allow_for(base_url: str) -> list[str]:
    """Hosts needed by a run against base_url (endpoint + dataset hosts)."""
    surl = sandbox_url(base_url)
    p = urllib.parse.urlparse(surl if "://" in surl else "http://" + surl)
    host = p.hostname or ""
    if not host:
        return list(RUN_HOSTS)
    try:
        port = p.port
    except ValueError:
        port = None
    hfmt = f"[{host}]" if ":" in host else host
    rules = [f"{hfmt}:{port}"] if port is not None else [f"{hfmt}:443", f"{hfmt}:80"]
    if host == host_alias():
        if port is not None:
            rules.append(f"localhost:{port}")
        else:
            rules += ["localhost:443", "localhost:80"]
    return rules + RUN_HOSTS


# ------------------------------------------------------------------------------------------------
# image build
# ------------------------------------------------------------------------------------------------
def gbench_source(on_line=print) -> Path | None:
    """A pinned checkout of gbench on the host (shallow clone; reused). The Docker image installs from it."""
    src = cache_dir() / "gbench-src"
    if (src / "pyproject.toml").is_file():
        return src
    on_line(f"cloning {GBENCH_REPO} ({GBENCH_REF})…")
    r = subprocess.run(["git", "clone", "--depth", "1", "--branch", GBENCH_REF, GBENCH_REPO, str(src)],
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        on_line("✗ git clone failed: " + r.stderr.strip()[-300:])
        return None
    return src


def gbench_commit() -> str:
    src = cache_dir() / "gbench-src"
    try:
        return subprocess.run(["git", "-C", str(src), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _tar_source(src: Path) -> bytes:
    """gbench's tree without .git, tests and docs images: what pip needs to install it."""
    buf = io.BytesIO()
    skip = {".git", "tests", "examples", "assets", "__pycache__"}
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(src.rglob("*")):
            rel = p.relative_to(src)
            if any(part in skip for part in rel.parts) or not p.is_file():
                continue
            tar.add(p, arcname=str(rel))
    return buf.getvalue()


def dockerfile() -> str:
    """Dockerfile that builds IMAGE: Python 3.12 + gbench[evals] + tokenizers + NLTK punkt, ready for --read-only."""
    extra_pkgs = " ".join(shlex.quote(p) for p in EXTRA_PIP)
    skip_json = json.dumps(SKIP_EXTRAS_ARM64)
    return f"""FROM {BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive \\
    UV_NO_CACHE=1 \\
    NLTK_DATA=/opt/nltk_data \\
    PATH="{VENV}/bin:${{PATH}}"
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git \\
    && rm -rf /var/lib/apt/lists/*
COPY gbench-src.tgz /tmp/gbench-src.tgz
RUN mkdir -p /opt/gbench-src /opt/nltk_data {CONTAINER_WORK} \\
    && tar -xzf /tmp/gbench-src.tgz -C /opt/gbench-src \\
    && rm -f /tmp/gbench-src.tgz \\
    && python3 -m venv /tmp/uvenv \\
    && /tmp/uvenv/bin/pip install --no-cache-dir -q uv \\
    && /tmp/uvenv/bin/uv venv -q --python python3 {VENV} \\
    && {VENV}/bin/python -c 'import platform, re, tomllib; skip = {skip_json} if platform.machine().lower() in ("aarch64", "arm64") else []; reqs = tomllib.load(open("/opt/gbench-src/pyproject.toml", "rb"))["project"]["optional-dependencies"]["evals"]; open("/tmp/extras.txt", "w").write("\\n".join(r for r in reqs if re.split(r"[<>=!~\\[; ]", r, maxsplit=1)[0].lower() not in skip) + "\\n")' \\
    && /tmp/uvenv/bin/uv pip install -q --python {VENV}/bin/python /opt/gbench-src -r /tmp/extras.txt {extra_pkgs} \\
    && rm -rf /tmp/uvenv /tmp/extras.txt \\
    && {VENV}/bin/python -m nltk.downloader -q -d /opt/nltk_data punkt punkt_tab \\
    && test -d /opt/nltk_data/tokenizers/punkt_tab \\
    && ln -sf {VENV}/bin/gbench /usr/local/bin/gbench \\
    && chmod -R a+rX {VENV} /opt/gbench-src /opt/nltk_data \\
    && chmod 1777 {CONTAINER_WORK} \\
    && echo {VERSION} > {MARKER} \\
    && {VENV}/bin/gbench --version
WORKDIR {CONTAINER_WORK}
"""


CATALOG_PY = r'''
import json, sys
from pathlib import Path
import gbench
from gbench.runners.evals import BUILTIN_PILLARS, NAMED_ONLY_SUITES
from gbench.runners.eval_suites import SUITES, suite_requires_tool_calling
try:
    from gbench.runners.eval_suites.canonical_sync import canonical_sync_for
except Exception:
    canonical_sync_for = lambda n: {}
pillar_of = {e: p for p, evs in BUILTIN_PILLARS for e in evs}
suites = []
for n in sorted(SUITES):
    try:
        tools = bool(suite_requires_tool_calling(n))
    except Exception:
        tools = False
    cs = canonical_sync_for(n) or {}
    suites.append({"name": n, "pillar": pillar_of.get(n, "Other"), "tools": tools,
                   "named_only": n in NAMED_ONLY_SUITES, "synced": cs.get("synced", ""), "status": cs.get("status", "")})
golden = []
gd = Path(gbench.__file__).parent / "golden_dataset"
for fp in sorted(gd.glob("*.json")):
    try:
        d = json.loads(fp.read_text())
    except Exception:
        continue
    golden.append({"id": d.get("id", fp.stem), "category": d.get("category", ""),
                   "description": d.get("description") or d.get("title") or d.get("name", ""),
                   "match_type": (d.get("golden_truth") or {}).get("match_type", ""),
                   "file": fp.name})
print(json.dumps({"gbench_version": getattr(gbench, "__version__", ""), "pillars": [p for p, _ in BUILTIN_PILLARS],
                  "suites": suites, "golden": golden}))
'''


def setup(on_line=print, cpus: int = 4) -> bool:
    """Builds the Docker image (idempotent). Returns True when the image is ready."""
    if not installed():
        on_line("✗ Docker is not installed (install Docker Desktop, OrbStack, Colima, or Docker Engine)")
        return False
    st = status()
    if st and "image" not in st.lower():
        on_line("✗ " + st)
        return False
    if template_ready():
        on_line(f"✔ Docker image {IMAGE} ready")
        if not (config_dir() / "catalog.json").is_file():
            refresh_catalog(on_line)
        return True
    src = gbench_source(on_line)
    if not src:
        return False
    ok = False
    log_path = config_dir() / "logs" / "setup.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gbtui-build-") as tmp:
        ctx = Path(tmp)
        (ctx / "Dockerfile").write_text(dockerfile(), encoding="utf-8")
        on_line(f"packaging gbench {gbench_commit() or GBENCH_REF}…")
        (ctx / "gbench-src.tgz").write_bytes(_tar_source(src))
        on_line(f"building Docker image {IMAGE} from {BASE_IMAGE} (a few minutes)…")
        cmd = ["docker", "build", "--progress=plain", "-t", IMAGE, str(ctx)]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=_env())
            tail: list[str] = []
            with log_path.open("w", encoding="utf-8") as log:
                assert p.stdout is not None
                for line in p.stdout:
                    log.write(line)
                    line = line.rstrip()
                    tail = (tail + [line])[-12:]
                    if line and not line.startswith(" ") and "WARNING" not in line and "cache has been disabled" not in line:
                        on_line("  " + line[-200:])
            if p.wait() != 0:
                for line in tail:
                    on_line("  │ " + line[-200:])
                on_line(f"✗ docker build failed (full log: {log_path})")
                return False
            ok = template_ready()
            if not ok:
                on_line(f"✗ Docker image {IMAGE} not found after build")
                return False
        except (OSError, subprocess.SubprocessError) as e:
            on_line(f"✗ {e}")
            return False
    if ok:
        repo = IMAGE.rsplit(":", 1)[0]
        for old in templates():                   # clean up older gbtui-gbench:v* images
            old_repo = old.rsplit(":", 1)[0]
            if (old_repo == repo or old_repo.endswith("/" + repo)) and not _same(old, IMAGE):
                docker("rmi", old, timeout=120)
                on_line(f"removed old image {old}")
        refresh_catalog(on_line)
        on_line(f"✔ Docker image {IMAGE} ready: each run gets a fresh hardened container, removed when it ends")
    return ok


def refresh_catalog(on_line=print) -> bool:
    """Asks gbench (inside an offline container) for its suites, pillars and Golden Set; caches it as JSON."""
    try:
        name = create(2, 2, [], on_line)
    except RuntimeError as e:
        on_line(f"✗ {e}")
        return False
    try:
        r = run_exec(name, [f"{VENV}/bin/python", "-c", CATALOG_PY], timeout=300)
        if r.returncode != 0:
            on_line("✗ catalog: " + _out(r).strip()[-300:])
            return False
        data = json.loads(r.stdout.strip().splitlines()[-1])
        data["gbench_commit"] = gbench_commit()
        (config_dir() / "catalog.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
        on_line(f"✔ catalog: {len(data['suites'])} eval suites in {len(data['pillars'])} pillars, "
                f"{len(data['golden'])} Golden Set tasks")
        return True
    except (ValueError, IndexError, OSError, subprocess.SubprocessError) as e:
        on_line(f"✗ catalog: {e}")
        return False
    finally:
        destroy(name)
