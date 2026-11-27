"""Host memory: how much RAM is available right now, so that nothing gbench-tui starts makes the machine swap or hang.

On macOS "available" is free + inactive + speculative + purgeable pages (what the kernel can hand out without swapping),
read from `vm_stat`; on Linux, MemAvailable from /proc/meminfo.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess

GB = 1024 ** 3


def total_bytes() -> int:
    try:
        if platform.system() == "Darwin":
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def available_bytes() -> int:
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = int(re.search(r"page size of (\d+)", out).group(1))
            pages = 0
            for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                m = re.search(rf"{key}:\s+(\d+)", out)
                pages += int(m.group(1)) if m else 0
            return pages * page
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        pass
    return 0


def swap_used_bytes() -> int:
    if platform.system() != "Darwin":
        return 0
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"used = ([\d.]+)([KMGT])", out)
        if m:
            mult = {"K": 1024, "M": 1024 ** 2, "G": GB, "T": GB * 1024}[m.group(2)]
            return int(float(m.group(1)) * mult)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 0


def rss_bytes(pid: int) -> int:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout
        return int(out.strip() or 0) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def fmt(b: float) -> str:
    return f"{b / GB:.1f} GB"


def check(need_bytes: int, min_free_gb: float) -> str | None:
    """None when `need_bytes` more can be used and the host still keeps `min_free_gb` free; otherwise the reason."""
    avail = available_bytes()
    if not avail:
        return None                          # unknown: don't block
    left = avail - need_bytes
    if left < min_free_gb * GB:
        return (f"not enough free memory: {fmt(avail)} available, this needs {fmt(need_bytes)} and gbench-tui keeps "
                f"{min_free_gb:g} GB free for the system (Settings → min free memory)")
    return None
