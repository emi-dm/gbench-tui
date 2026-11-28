"""Zero-dependency completion notifications: terminal bell + native macOS (`osascript`) / Linux (`notify-send`)."""
from __future__ import annotations

import shutil
import subprocess
import sys


def notify(title: str, body: str, bell: bool = True, native: bool = True) -> None:
    """Rings the terminal bell and triggers an OS desktop notification when available. Never raises."""
    if bell:
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except OSError:
            pass
    if not native:
        return
    clean_t = str(title or "gbench-tui").replace('"', "'").replace("\\", "")[:80]
    clean_b = str(body or "").replace('"', "'").replace("\\", "")[:200]
    try:
        if sys.platform == "darwin" and shutil.which("osascript"):
            script = f'display notification "{clean_b}" with title "{clean_t}"'
            subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform.startswith("linux") and shutil.which("notify-send"):
            subprocess.Popen(["notify-send", clean_t, clean_b], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass
