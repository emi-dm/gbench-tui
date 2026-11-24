"""Google DeepMind palette for the terminal (xterm-256 approximations) and the gbench wordmark.

    blue   #4285F4 → 69     deep  #0B57D0 → 26     cyan  #24C1E0 → 38
    violet #9177C7 → 104    pink  #CA6673 → 168    ink   #0E1117 → 233
    green  #34A853 → 71     yellow #FBBC04 → 214   red   #EA4335 → 203

The wordmark and progress bars use the blue → violet → pink gradient of Google DeepMind / Gemini.
"""
from __future__ import annotations

BLUE, DEEP, CYAN, VIOLET, PURPLE, PINK = 69, 26, 38, 104, 99, 168
GREEN, YELLOW, RED = 71, 214, 203
INK, BAR, SEL, RULE, MUTED, SOFT, LIGHT = 233, 235, 237, 239, 246, 250, 255
GRADIENT = [27, 33, 69, 105, 141, 140, 176, 175, 168]

HEX = {"blue": "#4285F4", "deep": "#0B57D0", "cyan": "#24C1E0", "violet": "#9177C7", "pink": "#CA6673",
       "green": "#34A853", "yellow": "#FBBC04", "red": "#EA4335", "ink": "#0E1117"}

WORDMARK = [
    " ██████╗ ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗",
    "██╔════╝ ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║",
    "██║  ███╗██████╔╝█████╗  ██╔██╗ ██║██║     ███████║",
    "██║   ██║██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║",
    "╚██████╔╝██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║",
    " ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝",
]
SMALL = "g·bench"


def grad(i: int, n: int) -> int:
    """The gradient colour at position i of n."""
    if n <= 1:
        return GRADIENT[0]
    return GRADIENT[min(len(GRADIENT) - 1, int(i * (len(GRADIENT) - 1) / (n - 1) + 0.5))]


def score_color(v: float | None) -> int:
    """Heat for a percentage: red < 30 ≤ yellow < 60 ≤ green; violet for ≥ 85."""
    if v is None:
        return MUTED
    if v >= 85:
        return CYAN
    if v >= 60:
        return GREEN
    if v >= 30:
        return YELLOW
    return RED
