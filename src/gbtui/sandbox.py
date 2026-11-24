"""Where gbench runs: always inside a Docker Sandbox (`sbx` microVM), never on the host.

    host: gbench-tui (curses, stdlib only) ── llama-server (Metal) or any /v1 endpoint
            │ sbx exec                                ▲
            ▼                                         │ the only host port the sandbox may reach
    sbx  gbtui-run-<pid>-<hex>  ← from the template gbtui-gbench:v<N>, its own RAM/CPU cap
         ├─ /opt/gbench (venv): gbench + its eval extras, the backend that does all the work
         ├─ /work  (bind mount of ~/.cache/gbench-tui/work): HF cache and the run's results
         └─ network: deny-all, then allow only the endpoint and the dataset hosts (Hugging Face, GitHub raw)

`setup()` builds the template once: a builder sandbox installs gbench from a pinned checkout (PyPI only, meanwhile),
dumps gbench's own suite / Golden Set catalog to JSON, is saved with `sbx template save` and removed. Every run then
gets a fresh sandbox from the template, removed when the run ends (finished, failed or cancelled); one left behind by a
process that died is pruned the next time a sandbox is created.
"""
from __future__ import annotations

import io
import json
import os
import re
import secrets
import shutil
import subprocess
import tarfile
from pathlib import Path

from .config import cache_dir, config_dir

VERSION = "1"
TEMPLATE = os.environ.get("GBTUI_TEMPLATE") or f"gbtui-gbench:v{VERSION}"
BUILDER = "gbtui-build"
PREFIX = "gbtui-run-"
GBENCH_REPO = "https://github.com/google-deepmind/gbench"
GBENCH_REF = os.environ.get("GBTUI_GBENCH_REF", "main")
VENV = "/opt/gbench"
PYTHON = "3.12"
WORK = "/work"                                 # mount point of cache_dir()/work inside the sandbox
MARKER = "/etc/gbtui-sandbox"
BUILD_MEMORY = "4g"
# Hosts the builder may reach while installing (then denied again)
BUILD_HOSTS = ["pypi.org", "files.pythonhosted.org", "raw.githubusercontent.com", "github.com",
               "objects.githubusercontent.com", "release-assets.githubusercontent.com", "astral.sh", "deb.debian.org",
               "ports.ubuntu.com", "archive.ubuntu.com", "security.ubuntu.com"]
# Hosts a run may reach besides the endpoint: datasets and tokenizers
RUN_HOSTS = ["huggingface.co", "**.huggingface.co", "**.hf.co", "raw.githubusercontent.com", "github.com",
             "objects.githubusercontent.com", "release-assets.githubusercontent.com", "data.statmt.org"]
# python packages installed next to gbench[evals]: the tokenizer stack (no torch) for --tokenizer
EXTRA_PIP = ["transformers<6", "jinja2"]
# gbench[evals] packages left out: no wheel for linux/aarch64 (the sandbox's arch) and no sdist. Only
# gorilla_apibench imports them; gbench reports that suite as infra_required instead.
SKIP_EXTRAS = ["tree-sitter", "tree-sitter-python"]

_SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|AUTH", re.I)


_SBX_AUTH = ("DOCKER_PAT", "DOCKER_ACCESS_TOKEN")


def _docker_token() -> str:
    """sbx's own sign-in, for hosts where sbx cannot reach the macOS Keychain (e.g. launched from an agent):
    $DOCKER_PAT / $DOCKER_ACCESS_TOKEN, or a token kept in ~/.config/gbench-tui/docker-pat (mode 600).
    It is only given to the sbx CLI, never to a sandbox."""
    p = config_dir() / "docker-pat"
    try:
        if p.stat().st_mode & 0o077:
            p.chmod(0o600)
        return p.read_text().strip()
    except OSError:
        return ""


def _env() -> dict:
    """The environment sbx is started with: the user's, minus anything that looks like a secret
    (except sbx's own Docker sign-in token, if one is configured)."""
    env = {k: v for k, v in os.environ.items() if not _SECRET.search(k)}
    for k in _SBX_AUTH:
        if os.environ.get(k):
            env[k] = os.environ[k]
    if not any(k in env for k in _SBX_AUTH) and (tok := _docker_token()):
        env["DOCKER_PAT"] = tok
    return env


def sbx(*args, timeout: float = 120, input: bytes | None = None, text: bool = True) -> subprocess.CompletedProcess:
    kw = {"input": input} if input is not None else {}
    return subprocess.run(["sbx", "--no-input", *args], capture_output=True, text=text and input is None,
                          timeout=timeout, env=_env(), **kw)


def _out(r: subprocess.CompletedProcess) -> str:
    o = (r.stdout or "") + (r.stderr or "")
    return o.decode(errors="replace") if isinstance(o, bytes) else o


# ------------------------------------------------------------------------------------------------
# status
# ------------------------------------------------------------------------------------------------
def installed() -> bool:
    return shutil.which("sbx") is not None


def status() -> str | None:
    """None when runs can start; otherwise a short reason."""
    if not installed():
        return "Docker Sandboxes (sbx) is not installed: brew install docker/tap/sbx"
    try:
        r = sbx("ls", "--json", timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return f"sbx not answering: {e}"
    if r.returncode != 0:
        out = _out(r)
        if "not signed in" in out or "login" in out:
            return "Docker Sandboxes needs a sign-in: run `sbx login` once"
        return "sbx daemon not running (sbx daemon start): " + out.strip()[-160:]
    if not signed_in():
        return "Docker Sandboxes needs a sign-in: run `sbx login` once"
    if not template_ready():
        return "sandbox template not built yet (Sandbox → Build template, or `gbench-tui setup`)"
    return None


_signed: dict = {}


def signed_in() -> bool:
    """`sbx create` needs a Docker sign-in; `sbx diagnose` says whether there is one (cached once it is true)."""
    if _signed.get("ok"):
        return True
    try:
        r = sbx("diagnose", timeout=90)
        ok = "✗ Authentication" not in _out(r)
    except (OSError, subprocess.SubprocessError):
        ok = True                                    # unknown: let create report it
    _signed["ok"] = ok
    return ok


def sandboxes() -> list[dict]:
    try:
        r = sbx("ls", "--json", timeout=30)
        data = json.loads(r.stdout or "[]") if r.returncode == 0 else []
        return data if isinstance(data, list) else data.get("sandboxes", [])
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def templates() -> list[str]:
    try:
        r = sbx("template", "ls", "--json", timeout=60)
        data = json.loads(r.stdout or "[]") if r.returncode == 0 else []
        return [f"{t.get('repository')}:{t.get('tag')}" for t in data]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def _same(ref: str, want: str) -> bool:
    return ref == want or ref.endswith("/" + want)


def template_ready() -> bool:
    return any(_same(t, TEMPLATE) for t in templates())


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
    """Removes run sandboxes whose owner process is gone (a crash, kill -9, a closed terminal)."""
    gone = []
    for s in sandboxes():
        n = s.get("name") or ""
        m = re.match(re.escape(PREFIX) + r"(\d+)-", n)
        if m and not _alive(int(m.group(1))):
            destroy(n)
            gone.append(n)
    return gone


def work_dir() -> Path:
    d = cache_dir() / "work"
    (d / "hf").mkdir(parents=True, exist_ok=True)
    (d / "runs").mkdir(parents=True, exist_ok=True)
    return d


def create(memory_gb: int, cpus: int, allow: list[str], on_line=None) -> str:
    """A run sandbox from the template: deny-all network, then only `allow` (host:port patterns). Raises RuntimeError."""
    say = on_line or (lambda _m: None)
    if not template_ready():
        raise RuntimeError("sandbox template not built yet (Sandbox → Build template)")
    prune()
    name = f"{PREFIX}{os.getpid()}-{secrets.token_hex(3)}"
    say(f"creating sandbox {name} ({memory_gb} GB RAM, {cpus} CPUs, network: deny-all + {len(allow)} hosts)")
    r = sbx("create", "--no-progress", "--name", name, "--memory", f"{memory_gb}g", "--cpus", str(cpus),
            "--pull", "never", "-t", TEMPLATE, "shell", f"{work_dir()}:{WORK}", timeout=600)
    if r.returncode != 0:
        destroy(name)
        if "not signed in" in _out(r):
            raise RuntimeError("Docker Sandboxes needs a sign-in: quit and run `sbx login` once")
        raise RuntimeError(f"could not create sandbox {name}: " + _out(r).strip()[-300:])
    try:
        set_network(name, allow)
    except RuntimeError:
        destroy(name)
        raise
    return name


def set_network(name: str, allow: list[str]) -> None:
    """sbx's network is deny-by-default (global `default-deny-all` plus Docker's allowlist); this adds the hosts the
    sandbox needs on top. An empty list closes it completely: a sandbox-scoped `deny **` (deny beats every allow)."""
    if not allow:
        r = sbx("policy", "deny", "network", "--sandbox", name, "**", timeout=60)
        if r.returncode != 0 and "already" not in _out(r).lower():
            raise RuntimeError("could not close the network: " + _out(r).strip()[-200:])
        return
    r = sbx("policy", "allow", "network", "--sandbox", name, ",".join(allow), timeout=60)
    if r.returncode != 0 and "already" not in _out(r).lower():
        raise RuntimeError("could not allow " + ",".join(allow) + ": " + _out(r).strip()[-200:])


def destroy(name: str) -> None:
    try:
        sbx("rm", "--force", name, timeout=300)
    except (OSError, subprocess.SubprocessError):
        pass


def exec_argv(name: str, argv: list[str], env: dict | None = None, user: str | None = None) -> list[str]:
    """The host command line that runs argv inside the sandbox (for Popen with streamed output)."""
    cmd = ["sbx", "--no-input", "exec"]
    if user:
        cmd += ["-u", user]
    for k, v in (env or {}).items():
        cmd += ["-e", f"{k}={v}"]
    return cmd + [name, *argv]


def upload(name: str, files: dict[str, bytes], dest: str, user: str | None = None) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for n, data in files.items():
            ti = tarfile.TarInfo(n)
            ti.size, ti.mode = len(data), 0o644
            tar.addfile(ti, io.BytesIO(data))
    args = ["exec", "-i"] + (["-u", user] if user else []) + [name, "sh", "-c", f"mkdir -p {dest} && tar -xf - -C {dest}"]
    r = sbx(*args, input=buf.getvalue(), timeout=300)
    if r.returncode != 0:
        raise RuntimeError("upload failed: " + _out(r)[-300:])


def host_alias() -> str:
    """How a sandbox names the host (where llama-server listens). Detected by setup(); host.docker.internal otherwise."""
    try:
        return json.loads((config_dir() / "sandbox.json").read_text()).get("host_alias") or "host.docker.internal"
    except (OSError, ValueError):
        return "host.docker.internal"


def sandbox_url(base_url: str) -> str:
    """The endpoint URL as seen from inside the sandbox: localhost becomes the host alias."""
    return re.sub(r"//(127\.0\.0\.1|localhost|0\.0\.0\.0)(?=[:/])", "//" + host_alias(), base_url)


def allow_for(base_url: str) -> list[str]:
    """Network allow rules for a run against base_url (plus the dataset hosts)."""
    hostport = sandbox_url(base_url).split("//", 1)[-1].split("/", 1)[0]
    host = hostport.split(":")[0]
    rules = [hostport] if ":" in hostport else [host + ":443", host + ":80"]
    if host == host_alias():
        rules.append(hostport.replace(host, "localhost"))
    return rules + RUN_HOSTS


# ------------------------------------------------------------------------------------------------
# template build
# ------------------------------------------------------------------------------------------------
def gbench_source(on_line=print) -> Path | None:
    """A pinned checkout of gbench on the host (shallow clone; reused). The template installs from it."""
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
    """Builds the template (idempotent). Returns True when the template is ready."""
    if not installed():
        on_line("✗ Docker Sandboxes (sbx) is not installed: brew install docker/tap/sbx")
        return False
    st = status()
    if st and "template" not in st:
        on_line("✗ " + st)
        return False
    if template_ready():
        on_line(f"✔ template {TEMPLATE} ready")
        if not (config_dir() / "catalog.json").is_file():
            refresh_catalog(on_line)
        return True
    src = gbench_source(on_line)
    if not src:
        return False
    if any(s.get("name") == BUILDER for s in sandboxes()):
        destroy(BUILDER)
    on_line(f"creating the builder sandbox {BUILDER} ({BUILD_MEMORY} RAM, {cpus} CPUs, no host files)…")
    r = sbx("create", "--no-progress", "--name", BUILDER, "--memory", BUILD_MEMORY, "--cpus", str(cpus), "shell",
            timeout=900)
    if r.returncode != 0:
        on_line("✗ " + _out(r).strip()[-400:])
        return False
    ok = False
    try:
        set_network(BUILDER, BUILD_HOSTS)
        on_line("uploading gbench " + (gbench_commit() or GBENCH_REF) + "…")
        upload(BUILDER, {"gbench-src.tgz": _tar_source(src)}, "/tmp/gbtui", user="root")
        script = f"""
set -e
mkdir -p /opt/gbench-src && tar -xzf /tmp/gbtui/gbench-src.tgz -C /opt/gbench-src && rm -rf /tmp/gbtui
export HOME=/root UV_PYTHON_INSTALL_DIR=/opt/python UV_NO_CACHE=1
# gbench's eval extras pin wheels that only exist up to CPython 3.13: use uv and a standalone CPython {PYTHON}
python3 -m venv /tmp/uvenv && /tmp/uvenv/bin/pip install -q uv
/tmp/uvenv/bin/uv venv -q --python {PYTHON} {VENV}
echo "python: $({VENV}/bin/python --version)"
echo "installing gbench[evals] (a few minutes)…"
{VENV}/bin/python - > /tmp/extras.txt <<'PYEOF'
import re, tomllib
skip = {SKIP_EXTRAS!r}
for req in tomllib.load(open("/opt/gbench-src/pyproject.toml", "rb"))["project"]["optional-dependencies"]["evals"]:
    if re.split(r"[<>=!~\[; ]", req, maxsplit=1)[0].lower() not in skip:
        print(req)
PYEOF
/tmp/uvenv/bin/uv pip install -q --python {VENV}/bin/python /opt/gbench-src -r /tmp/extras.txt {' '.join(repr(p) for p in EXTRA_PIP)}
rm -rf /tmp/uvenv
{VENV}/bin/python -m nltk.downloader -q -d /opt/nltk_data punkt punkt_tab || echo "warning: nltk data not downloaded"
ln -sf {VENV}/bin/gbench /usr/local/bin/gbench
mkdir -p {WORK}
echo {VERSION} > {MARKER}
{VENV}/bin/gbench --version
"""
        on_line("installing gbench and its eval extras inside the sandbox (PyPI only)…")
        p = subprocess.Popen(exec_argv(BUILDER, ["sh", "-c", script], user="root"), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, env=_env())
        log_path = config_dir() / "logs" / "setup.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tail: list[str] = []
        with log_path.open("w") as log:
            for line in p.stdout:
                log.write(line)
                line = line.rstrip()
                tail = (tail + [line])[-12:]
                if line and not line.startswith(" ") and "WARNING" not in line and "cache has been disabled" not in line:
                    on_line("  " + line[-200:])
        if p.wait() != 0:
            for line in tail:
                on_line("  │ " + line[-200:])
            on_line(f"✗ installation failed (full log: {log_path})")
            return False
        set_network(BUILDER, [])
        on_line(f"saving the sandbox as the template {TEMPLATE}…")
        sbx("stop", BUILDER, timeout=300)
        r = sbx("template", "save", BUILDER, TEMPLATE, timeout=3600)
        if r.returncode != 0 or not template_ready():
            on_line("✗ could not save the template: " + _out(r).strip()[-300:])
            return False
        ok = True
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        on_line(f"✗ {e}")
    finally:
        destroy(BUILDER)
    if ok:
        refresh_catalog(on_line)
        on_line(f"✔ template {TEMPLATE} ready: each run gets a fresh sandbox from it, removed when the run ends")
    return ok


def refresh_catalog(on_line=print) -> bool:
    """Asks gbench (inside a short-lived sandbox) for its suites, pillars and Golden Set; caches it as JSON."""
    try:
        name = create(2, 2, [], on_line)
    except RuntimeError as e:
        on_line(f"✗ {e}")
        return False
    try:
        r = sbx("exec", name, f"{VENV}/bin/python", "-c", CATALOG_PY, timeout=300)
        if r.returncode != 0:
            on_line("✗ catalog: " + _out(r).strip()[-300:])
            return False
        data = json.loads(r.stdout.strip().splitlines()[-1])
        data["gbench_commit"] = gbench_commit()
        (config_dir() / "catalog.json").write_text(json.dumps(data, indent=1))
        on_line(f"✔ catalog: {len(data['suites'])} eval suites in {len(data['pillars'])} pillars, "
                f"{len(data['golden'])} Golden Set tasks")
        return True
    except (ValueError, IndexError, OSError, subprocess.SubprocessError) as e:
        on_line(f"✗ catalog: {e}")
        return False
    finally:
        destroy(name)
