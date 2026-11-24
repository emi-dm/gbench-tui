"""Google DeepMind's blues and white for the terminal (xterm-256 approximations) and the gbench wordmark.

    deep   #0B57D0 → 25     blue   #1A73E8 → 33     sky    #4285F4 → 39
    light  #8AB4F8 → 111    pale   #D3E3FD → 153    white  #FFFFFF → 231

Everything is blue on white/ink: the wordmark and the bars run deep blue → white. Green, yellow and red are kept only
for what they mean (passed, warning, failed).
"""
from __future__ import annotations

# names kept from the first palette (used across the TUI); all of them are blues now
DEEP, BLUE, CYAN, VIOLET, PURPLE, PINK = 25, 33, 39, 75, 111, 153
GREEN, YELLOW, RED = 79, 221, 203
INK, BAR, SEL, RULE, MUTED, SOFT, LIGHT = 233, 234, 17, 24, 110, 189, 231
GRADIENT = [25, 26, 27, 33, 39, 75, 111, 153, 189, 231]

HEX = {"deep": "#0B57D0", "blue": "#1A73E8", "sky": "#4285F4", "light": "#8AB4F8", "pale": "#D3E3FD",
       "white": "#FFFFFF", "navy": "#041E49", "ink": "#0A0F1C"}

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
    """Heat for a percentage, in blues: white ≥ 85 > pale ≥ 60 > sky ≥ 30 > red."""
    if v is None:
        return MUTED
    if v >= 85:
        return LIGHT
    if v >= 60:
        return PURPLE
    if v >= 30:
        return CYAN
    return RED
