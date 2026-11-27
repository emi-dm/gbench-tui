"""Multi-theme vivid terminal palettes (xterm-256) and 2D animation helpers for gbench-tui.

Default theme is `aurora` (Gemini / DeepMind Prism: sapphire → ultraviolet → orchid → coral with emerald, cyan and
amber semantic accents on sleek obsidian-slate surfaces). Switch live with `t` on Home or in Settings:
    aurora · cyber · sunset · emerald · deepmind
"""
from __future__ import annotations

import math

THEME_ORDER = ("aurora", "cyber", "sunset", "emerald", "deepmind")

THEMES: dict[str, dict] = {
    "aurora": {
        "label": "Gemini Aurora",
        "desc": "Sapphire → ultraviolet → orchid → coral prism on obsidian slate",
        "NAVY": 17, "DEEP": 63, "BLUE": 75, "CYAN": 87, "SKY": 81, "VIOLET": 141,
        "PURPLE": 177, "ICE": 117, "PINK": 212, "PALE": 183, "CORAL": 210, "MIST": 195,
        "GREEN": 85, "GREEN_BRIGHT": 121, "YELLOW": 221, "GOLD": 222, "RED": 203, "RED_SOFT": 210,
        "INK": 233, "BAR": 235, "BAR_ALT": 236, "SEL": 237,
        "RULE": 60, "RULE_DIM": 238, "DIM": 103, "MUTED": 146, "SOFT": 252, "LIGHT": 231,
        "GRADIENT": [63, 69, 75, 81, 105, 141, 177, 212, 211, 216],
        "BRAND_GRAD": [27, 63, 99, 135, 170],
        "WAVE_GRAD": [60, 63, 105, 141, 177, 212, 231, 212, 177, 141, 105, 63, 60],
        "MENU_COLORS": [87, 85, 212, 141, 216, 75, 221, 80, 105, 183],
        "PILLAR_COLORS": [81, 85, 177, 216, 221, 141, 87, 212],
        "HEX": {
            "bg": "#0B0E17", "card": "#121726", "card2": "#171D30", "ink": "#EEF2FF", "muted": "#94A3C8",
            "rule": "#27314F", "accent1": "#4F7CFF", "accent2": "#A855F7", "accent3": "#F472B6",
            "good": "#34D399", "mid": "#38BDF8", "warn": "#FBBF24", "bad": "#F87171",
            "deep": "#4F7CFF", "blue": "#7C5CFF", "sky": "#38BDF8", "light": "#C084FC", "pale": "#E9D5FF",
            "white": "#FFFFFF", "navy": "#0B0E17",
        },
    },
    "cyber": {
        "label": "Cyber Neon",
        "desc": "Electric cyan → laser magenta → acid mint → neon gold",
        "NAVY": 17, "DEEP": 39, "BLUE": 45, "CYAN": 51, "SKY": 87, "VIOLET": 171,
        "PURPLE": 207, "ICE": 123, "PINK": 213, "PALE": 219, "CORAL": 204, "MIST": 195,
        "GREEN": 84, "GREEN_BRIGHT": 120, "YELLOW": 227, "GOLD": 221, "RED": 197, "RED_SOFT": 204,
        "INK": 232, "BAR": 234, "BAR_ALT": 236, "SEL": 237,
        "RULE": 61, "RULE_DIM": 238, "DIM": 103, "MUTED": 146, "SOFT": 253, "LIGHT": 231,
        "GRADIENT": [45, 51, 87, 123, 141, 177, 207, 206, 213, 228],
        "BRAND_GRAD": [31, 37, 91, 127, 163],
        "WAVE_GRAD": [61, 39, 51, 123, 207, 213, 231, 213, 207, 123, 51, 39, 61],
        "MENU_COLORS": [51, 84, 207, 177, 123, 87, 227, 49, 141, 213],
        "PILLAR_COLORS": [51, 84, 207, 227, 141, 204, 123, 213],
        "HEX": {
            "bg": "#080A12", "card": "#101322", "card2": "#161B30", "ink": "#F0FDFF", "muted": "#8DA2C8",
            "rule": "#283052", "accent1": "#00E5FF", "accent2": "#D946EF", "accent3": "#FACC15",
            "good": "#4ADE80", "mid": "#22D3EE", "warn": "#FDE047", "bad": "#FB7185",
            "deep": "#00B8D9", "blue": "#00E5FF", "sky": "#67E8F9", "light": "#E879F9", "pale": "#F5D0FE",
            "white": "#FFFFFF", "navy": "#080A12",
        },
    },
    "sunset": {
        "label": "Solar Ember",
        "desc": "Deep rose → warm coral → tangerine → amber gold → soft lilac",
        "NAVY": 52, "DEEP": 167, "BLUE": 209, "CYAN": 215, "SKY": 216, "VIOLET": 176,
        "PURPLE": 183, "ICE": 222, "PINK": 211, "PALE": 223, "CORAL": 203, "MIST": 230,
        "GREEN": 114, "GREEN_BRIGHT": 157, "YELLOW": 221, "GOLD": 222, "RED": 203, "RED_SOFT": 210,
        "INK": 233, "BAR": 235, "BAR_ALT": 236, "SEL": 237,
        "RULE": 95, "RULE_DIM": 238, "DIM": 138, "MUTED": 181, "SOFT": 253, "LIGHT": 231,
        "GRADIENT": [168, 204, 209, 215, 221, 222, 223, 218, 183, 231],
        "BRAND_GRAD": [125, 161, 167, 173, 172],
        "WAVE_GRAD": [95, 167, 209, 215, 221, 229, 231, 229, 221, 215, 209, 167, 95],
        "MENU_COLORS": [215, 114, 204, 183, 223, 222, 221, 115, 176, 218],
        "PILLAR_COLORS": [209, 114, 183, 221, 204, 216, 115, 218],
        "HEX": {
            "bg": "#120D11", "card": "#1B1319", "card2": "#241922", "ink": "#FFF7ED", "muted": "#C4A4B0",
            "rule": "#3D2936", "accent1": "#F43F5E", "accent2": "#FB923C", "accent3": "#FBBF24",
            "good": "#4ADE80", "mid": "#FB923C", "warn": "#FBBF24", "bad": "#F87171",
            "deep": "#E11D48", "blue": "#F97316", "sky": "#FB923C", "light": "#FDBA74", "pale": "#FED7AA",
            "white": "#FFFFFF", "navy": "#120D11",
        },
    },
    "emerald": {
        "label": "Nordic Emerald",
        "desc": "Deep fjord teal → emerald mint → glacier cyan → warm gold",
        "NAVY": 23, "DEEP": 36, "BLUE": 43, "CYAN": 86, "SKY": 80, "VIOLET": 115,
        "PURPLE": 158, "ICE": 122, "PINK": 194, "PALE": 159, "CORAL": 210, "MIST": 195,
        "GREEN": 85, "GREEN_BRIGHT": 121, "YELLOW": 221, "GOLD": 222, "RED": 203, "RED_SOFT": 210,
        "INK": 233, "BAR": 235, "BAR_ALT": 236, "SEL": 237,
        "RULE": 29, "RULE_DIM": 238, "DIM": 66, "MUTED": 109, "SOFT": 252, "LIGHT": 231,
        "GRADIENT": [29, 36, 43, 49, 85, 86, 122, 159, 194, 228],
        "BRAND_GRAD": [23, 29, 30, 36, 37],
        "WAVE_GRAD": [29, 36, 43, 85, 122, 159, 231, 159, 122, 85, 43, 36, 29],
        "MENU_COLORS": [86, 85, 221, 80, 194, 122, 222, 49, 43, 159],
        "PILLAR_COLORS": [85, 86, 221, 80, 122, 158, 49, 194],
        "HEX": {
            "bg": "#081211", "card": "#0F1D1B", "card2": "#152825", "ink": "#ECFDF5", "muted": "#86B3A9",
            "rule": "#20403A", "accent1": "#10B981", "accent2": "#2DD4BF", "accent3": "#FBBF24",
            "good": "#34D399", "mid": "#2DD4BF", "warn": "#FBBF24", "bad": "#F87171",
            "deep": "#059669", "blue": "#10B981", "sky": "#2DD4BF", "light": "#6EE7B7", "pale": "#CCFBF1",
            "white": "#FFFFFF", "navy": "#081211",
        },
    },
    "deepmind": {
        "label": "DeepMind Classic",
        "desc": "Google DeepMind sapphire blues and crisp white",
        "NAVY": 17, "DEEP": 25, "BLUE": 33, "CYAN": 39, "SKY": 75, "VIOLET": 75,
        "PURPLE": 111, "ICE": 117, "PINK": 153, "PALE": 153, "CORAL": 210, "MIST": 189,
        "GREEN": 79, "GREEN_BRIGHT": 121, "YELLOW": 221, "GOLD": 222, "RED": 203, "RED_SOFT": 210,
        "INK": 233, "BAR": 234, "BAR_ALT": 235, "SEL": 17,
        "RULE": 24, "RULE_DIM": 237, "DIM": 67, "MUTED": 110, "SOFT": 189, "LIGHT": 231,
        "GRADIENT": [25, 26, 27, 33, 39, 75, 111, 153, 189, 231],
        "BRAND_GRAD": [25, 26, 27, 33, 39],
        "WAVE_GRAD": [24, 25, 33, 39, 75, 117, 159, 231, 159, 117, 75, 39, 33, 25, 24],
        "MENU_COLORS": [39, 75, 117, 111, 159, 153, 221, 79, 75, 189],
        "PILLAR_COLORS": [75, 117, 111, 153, 39, 189, 79, 221],
        "HEX": {
            "bg": "#FFFFFF", "card": "#F8FAFF", "card2": "#EEF4FF", "ink": "#041E49", "muted": "#5F6B85",
            "rule": "#D3E3FD", "accent1": "#0B57D0", "accent2": "#1A73E8", "accent3": "#4285F4",
            "good": "#137333", "mid": "#1A73E8", "warn": "#B06000", "bad": "#C5221F",
            "deep": "#0B57D0", "blue": "#1A73E8", "sky": "#4285F4", "light": "#8AB4F8", "pale": "#D3E3FD",
            "white": "#FFFFFF", "navy": "#041E49",
        },
    },
}

ACTIVE = "aurora"

# Module-level colour tokens (rebound dynamically by set_theme)
NAVY = DEEP = BLUE = CYAN = SKY = VIOLET = PURPLE = ICE = PINK = PALE = CORAL = MIST = 0
GREEN = GREEN_BRIGHT = YELLOW = GOLD = RED = RED_SOFT = 0
INK = BAR = BAR_ALT = SEL = RULE = RULE_DIM = DIM = MUTED = SOFT = LIGHT = 0
GRADIENT: list[int] = []
BRAND_GRAD: list[int] = []
WAVE_GRAD: list[int] = []
MENU_COLORS: list[int] = []
PILLAR_COLORS: list[int] = []
HEX: dict[str, str] = {}
_ON_SEL: dict[int, int] = {}

WORDMARK = [
    " ██████╗ ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗",
    "██╔════╝ ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║",
    "██║  ███╗██████╔╝█████╗  ██╔██╗ ██║██║     ███████║",
    "██║   ██║██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║",
    "╚██████╔╝██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║",
    " ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝",
]
SMALL = "g·bench"


def set_theme(name: str) -> str:
    """Activate one of the built-in palettes and rebind module-level tokens."""
    global ACTIVE, _ON_SEL
    if name not in THEMES:
        name = "aurora"
    ACTIVE = name
    data = THEMES[name]
    g = globals()
    for k, v in data.items():
        if k not in ("label", "desc"):
            g[k] = list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v)
    # High-contrast promotion map for items drawn over SEL
    if data["SEL"] == 17:
        _ON_SEL = {
            data["DEEP"]: data["ICE"], 26: data["ICE"], 27: data["ICE"],
            data["BLUE"]: data["ICE"], data["CYAN"]: data["PALE"], data["VIOLET"]: data["PALE"],
            data["PURPLE"]: data["MIST"], data["RULE"]: data["MUTED"], data["RULE_DIM"]: data["MUTED"],
            data["DIM"]: data["MUTED"], data["MUTED"]: data["SOFT"],
            data["RED"]: data["RED_SOFT"], data["GREEN"]: data["GREEN_BRIGHT"],
        }
    else:
        # On sleek obsidian-graphite SEL (237), bright hues already pop; only lift very dark/dim tokens
        _ON_SEL = {
            data["RULE"]: data["MUTED"], data["RULE_DIM"]: data["MUTED"],
            data["DIM"]: data["SOFT"], data["MUTED"]: data["LIGHT"],
            25: data["ICE"], 26: data["ICE"], 27: data["ICE"],
            data["RED"]: data["RED_SOFT"], data["GREEN"]: data["GREEN_BRIGHT"],
        }
    return ACTIVE


def current_theme() -> str:
    return ACTIVE


def theme_info(name: str | None = None) -> dict:
    return THEMES.get(name or ACTIVE, THEMES["aurora"])


def cycle_theme(step: int = 1) -> str:
    idx = THEME_ORDER.index(ACTIVE) if ACTIVE in THEME_ORDER else 0
    return set_theme(THEME_ORDER[(idx + step) % len(THEME_ORDER)])


# Initialize default theme on import
set_theme("aurora")


def on_sel(fg: int | None) -> int | None:
    """Return a high-contrast foreground colour suitable for drawing on T.SEL."""
    if fg is None:
        return LIGHT
    return _ON_SEL.get(fg, fg)


def grad(i: int, n: int) -> int:
    """The active theme's prism gradient colour at position i of n."""
    if n <= 1:
        return GRADIENT[0]
    idx = int(i * (len(GRADIENT) - 1) / (n - 1) + 0.5)
    return GRADIENT[max(0, min(len(GRADIENT) - 1, idx))]


def shimmer(i: int, n: int, tick: int, period: int = 52) -> int:
    """1D prism gradient with a calm specular sheen sweeping left-to-right every `period` ticks."""
    return shimmer_2d(0, i, 1, n, tick, period=period)


def shimmer_2d(row: int, col: int, rows: int, cols: int, tick: int, period: int = 52) -> int:
    """2D diagonal living aurora gradient with slow breathing drift + periodic specular highlight."""
    if cols <= 1:
        return GRADIENT[0]
    # Gentle breathing drift along the diagonal
    drift = math.sin(tick * 0.065) * 0.14
    diag_span = max(1.0, (cols - 1) + (rows - 1) * 2.4)
    pos = (col + row * 2.4) / diag_span + drift
    pos = max(0.0, min(1.0, pos))
    base_idx = int(pos * (len(GRADIENT) - 1) + 0.5)
    base_idx = max(0, min(len(GRADIENT) - 1, base_idx))

    # Periodic diagonal diamond sheen
    phase = tick % period
    sweep_len = 22
    if phase < sweep_len:
        center = (phase / (sweep_len - 1)) * (diag_span + 12) - 6
        dist = abs((col + row * 2.4) - center)
        if dist < 2.1:
            return LIGHT
        if dist < 4.5:
            return GRADIENT[min(len(GRADIENT) - 1, base_idx + 2)]
        if dist < 7.0:
            return GRADIENT[min(len(GRADIENT) - 1, base_idx + 1)]
    return GRADIENT[base_idx]


def wave_cell(i: int, width: int, tick: int) -> tuple[str, int]:
    """Return (char, color) for column i of an animated indeterminate prism scanning wave."""
    if width <= 1:
        return "█", CYAN
    t = (1.0 - math.cos(tick * 0.28)) * 0.5
    center = t * (width - 1)
    span = max(3.2, width * 0.18)
    dist = abs(i - center) / span
    if dist < 0.35:
        return "█", LIGHT
    if dist < 0.72:
        return "█", PINK
    if dist < 1.10:
        return "▓", PURPLE
    if dist < 1.50:
        return "▒", VIOLET
    if dist < 1.95:
        return "░", BLUE
    return "─", RULE_DIM


def mem_color(frac: float) -> int:
    """Heat colour for host memory utilization (emerald → cyan → gold → coral)."""
    if frac < 0.55:
        return GREEN
    if frac < 0.75:
        return CYAN
    if frac < 0.88:
        return YELLOW
    return RED


def score_color(v: float | None) -> int:
    """Vivid multi-tonal heat spectrum for accuracy percentages (emerald ≥ 88 > cyan ≥ 72 > violet ≥ 55 > amber ≥ 35 > coral)."""
    if v is None:
        return MUTED
    if ACTIVE == "deepmind":
        if v >= 85:
            return LIGHT
        if v >= 70:
            return PALE
        if v >= 50:
            return PURPLE
        if v >= 30:
            return CYAN
        return RED
    if v >= 88:
        return GREEN
    if v >= 72:
        return CYAN
    if v >= 55:
        return VIOLET
    if v >= 35:
        return YELLOW
    return RED


def tier_color(tier: str) -> int:
    """Distinct vibrant accent colour per benchmark tier."""
    return {"golden": YELLOW, "evals": PURPLE, "perf": GREEN, "preflight": CYAN}.get(tier, ICE)


def status_color(status: str) -> int:
    """Consistent semantic colour for run / sandbox / endpoint status."""
    s = (status or "").lower()
    if s in ("ok", "done", "ready", "passed", "up"):
        return GREEN
    if s in ("failed", "error", "down"):
        return RED
    if s in ("cancelled", "warning", "timeout"):
        return YELLOW
    if s in ("server", "sandbox", "running", "collecting"):
        return PINK
    return MUTED


def pillar_color(index: int) -> int:
    """Distinct jewel-tone colour per eval suite pillar."""
    if not PILLAR_COLORS:
        return VIOLET
    return PILLAR_COLORS[index % len(PILLAR_COLORS)]


def menu_color(index: int) -> int:
    """Distinct jewel-tone colour per Home navigation screen."""
    if not MENU_COLORS:
        return CYAN
    return MENU_COLORS[index % len(MENU_COLORS)]
