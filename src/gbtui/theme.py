"""Multi-theme vivid terminal palettes (xterm-256 + HEX) and 2D animation helpers for gbench-tui.

17 live-switchable palettes (press `t` / `T` on Home, in Settings, or inside `leaderboard.html`):
    aurora · antigravity · cyber · sunset · emerald · deepmind · tokyo · catppuccin · dracula ·
    gruvbox · monokai · nord · rosepine · kanagawa · solarized · onedark · synthwave
"""
from __future__ import annotations

import math

WORDMARK = [
    " ██████╗ ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗",
    "██╔════╝ ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║",
    "██║  ███╗██████╔╝█████╗  ██╔██╗ ██║██║     ███████║",
    "██║   ██║██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║",
    "╚██████╔╝██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║",
    " ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝",
]
SMALL = "g·bench"

THEME_ORDER = (
    "aurora", "antigravity", "cyber", "sunset", "emerald", "deepmind",
    "tokyo", "catppuccin", "dracula", "gruvbox", "monokai",
    "nord", "rosepine", "kanagawa", "solarized", "onedark", "synthwave",
)

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
        "MENU_COLORS": [87, 85, 212, 141, 216, 75, 221, 80, 105, 183, 121, 141],
        "PILLAR_COLORS": [81, 85, 177, 216, 221, 141, 87, 212],
        "HEX": {
            "bg": "#0B0E17", "card": "#121726", "card2": "#171D30", "ink": "#EEF2FF", "muted": "#94A3C8",
            "rule": "#27314F", "accent1": "#4F7CFF", "accent2": "#A855F7", "accent3": "#F472B6",
            "good": "#34D399", "mid": "#38BDF8", "warn": "#FBBF24", "bad": "#F87171",
            "deep": "#4F7CFF", "blue": "#7C5CFF", "sky": "#38BDF8", "light": "#C084FC", "pale": "#E9D5FF",
            "white": "#FFFFFF", "navy": "#0B0E17",
        },
    },
    "antigravity": {
        "label": "Antigravity Orbital",
        "desc": "Zero-G obsidian void → ion cyan → aurora mint → quantum violet & starlight",
        "NAVY": 17, "DEEP": 39, "BLUE": 75, "CYAN": 51, "SKY": 87, "VIOLET": 141,
        "PURPLE": 147, "ICE": 123, "PINK": 212, "PALE": 189, "CORAL": 210, "MIST": 195,
        "GREEN": 49, "GREEN_BRIGHT": 122, "YELLOW": 221, "GOLD": 227, "RED": 204, "RED_SOFT": 211,
        "INK": 232, "BAR": 234, "BAR_ALT": 235, "SEL": 237,
        "RULE": 61, "RULE_DIM": 238, "DIM": 103, "MUTED": 146, "SOFT": 253, "LIGHT": 231,
        "GRADIENT": [39, 45, 51, 87, 123, 147, 141, 177, 49, 221],
        "BRAND_GRAD": [25, 33, 39, 99, 49],
        "WAVE_GRAD": [61, 39, 51, 49, 141, 123, 231, 123, 141, 49, 51, 39, 61],
        "MENU_COLORS": [87, 49, 141, 75, 221, 123, 212, 147, 122, 189, 210, 87],
        "PILLAR_COLORS": [51, 49, 141, 221, 75, 212, 123, 147],
        "HEX": {
            "bg": "#060913", "card": "#0D1326", "card2": "#131C36", "ink": "#F1F5FF", "muted": "#8FA6CE",
            "rule": "#243358", "accent1": "#00E5FF", "accent2": "#2EE6A6", "accent3": "#A78BFA",
            "good": "#2EE6A6", "mid": "#38BDF8", "warn": "#FCD34D", "bad": "#FB7185",
            "deep": "#2563EB", "blue": "#38BDF8", "sky": "#00E5FF", "light": "#A78BFA", "pale": "#E0E7FF",
            "white": "#FFFFFF", "navy": "#060913",
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
        "MENU_COLORS": [51, 84, 207, 177, 123, 87, 227, 49, 141, 213, 120, 177],
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
        "MENU_COLORS": [215, 114, 204, 183, 223, 222, 221, 115, 176, 218, 157, 183],
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
        "MENU_COLORS": [86, 85, 221, 80, 194, 122, 222, 49, 43, 159, 121, 80],
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
        "NAVY": 17, "DEEP": 27, "BLUE": 33, "CYAN": 75, "SKY": 75, "VIOLET": 69,
        "PURPLE": 111, "ICE": 117, "PINK": 153, "PALE": 153, "CORAL": 210, "MIST": 195,
        "GREEN": 79, "GREEN_BRIGHT": 115, "YELLOW": 221, "GOLD": 222, "RED": 203, "RED_SOFT": 210,
        "INK": 234, "BAR": 235, "BAR_ALT": 236, "SEL": 17,
        "RULE": 25, "RULE_DIM": 238, "DIM": 67, "MUTED": 103, "SOFT": 252, "LIGHT": 231,
        "GRADIENT": [25, 26, 27, 33, 39, 69, 75, 111, 153, 195],
        "BRAND_GRAD": [18, 25, 26, 27, 33],
        "WAVE_GRAD": [24, 25, 27, 33, 75, 153, 231, 153, 75, 33, 27, 25, 24],
        "MENU_COLORS": [75, 79, 117, 111, 153, 39, 221, 79, 69, 153, 115, 111],
        "PILLAR_COLORS": [75, 79, 111, 153, 221, 39, 117, 69],
        "HEX": {
            "bg": "#0A1020", "card": "#101931", "card2": "#162242", "ink": "#F8FAFC", "muted": "#93A8CE",
            "rule": "#233663", "accent1": "#1A73E8", "accent2": "#4285F4", "accent3": "#7CC0FF",
            "good": "#34D399", "mid": "#60A5FA", "warn": "#FBBF24", "bad": "#F87171",
            "deep": "#0E3F8A", "blue": "#1A73E8", "sky": "#4285F4", "light": "#7CC0FF", "pale": "#D2E3FC",
            "white": "#FFFFFF", "navy": "#0A1020",
        },
    },
    "tokyo": {
        "label": "Tokyo Night",
        "desc": "Indigo night sky → neon azure → sakura violet → jade mint",
        "NAVY": 17, "DEEP": 62, "BLUE": 111, "CYAN": 117, "SKY": 110, "VIOLET": 141,
        "PURPLE": 176, "ICE": 153, "PINK": 204, "PALE": 189, "CORAL": 210, "MIST": 195,
        "GREEN": 114, "GREEN_BRIGHT": 150, "YELLOW": 179, "GOLD": 222, "RED": 203, "RED_SOFT": 210,
        "INK": 234, "BAR": 235, "BAR_ALT": 236, "SEL": 238,
        "RULE": 60, "RULE_DIM": 237, "DIM": 60, "MUTED": 103, "SOFT": 252, "LIGHT": 231,
        "GRADIENT": [62, 68, 111, 117, 141, 176, 204, 210, 179, 114],
        "BRAND_GRAD": [24, 61, 68, 111, 141],
        "WAVE_GRAD": [60, 62, 111, 117, 141, 204, 231, 204, 141, 117, 111, 62, 60],
        "MENU_COLORS": [117, 114, 204, 141, 179, 111, 150, 176, 110, 189, 210, 141],
        "PILLAR_COLORS": [111, 114, 141, 204, 179, 117, 176, 150],
        "HEX": {
            "bg": "#1A1B26", "card": "#1F2335", "card2": "#24283B", "ink": "#C0CAF5", "muted": "#7AA2F7",
            "rule": "#3B4261", "accent1": "#7AA2F7", "accent2": "#BB9AF7", "accent3": "#F7768E",
            "good": "#9ECE6A", "mid": "#7DCFFF", "warn": "#E0AF68", "bad": "#F7768E",
            "deep": "#3D59A1", "blue": "#7AA2F7", "sky": "#7DCFFF", "light": "#BB9AF7", "pale": "#C0CAF5",
            "white": "#FFFFFF", "navy": "#1A1B26",
        },
    },
    "catppuccin": {
        "label": "Catppuccin Mocha",
        "desc": "Pastel lavender → mauve → peach → sky → soft teal on espresso",
        "NAVY": 235, "DEEP": 141, "BLUE": 111, "CYAN": 116, "SKY": 117, "VIOLET": 183,
        "PURPLE": 176, "ICE": 159, "PINK": 218, "PALE": 189, "CORAL": 216, "MIST": 195,
        "GREEN": 151, "GREEN_BRIGHT": 157, "YELLOW": 223, "GOLD": 222, "RED": 211, "RED_SOFT": 217,
        "INK": 234, "BAR": 235, "BAR_ALT": 236, "SEL": 238,
        "RULE": 60, "RULE_DIM": 237, "DIM": 103, "MUTED": 146, "SOFT": 253, "LIGHT": 231,
        "GRADIENT": [111, 117, 116, 151, 183, 218, 216, 223, 189, 159],
        "BRAND_GRAD": [60, 104, 141, 183, 218],
        "WAVE_GRAD": [60, 111, 141, 183, 218, 223, 231, 223, 218, 183, 141, 111, 60],
        "MENU_COLORS": [117, 151, 218, 183, 216, 111, 223, 116, 141, 189, 157, 183],
        "PILLAR_COLORS": [111, 151, 183, 216, 223, 116, 218, 159],
        "HEX": {
            "bg": "#1E1E2E", "card": "#24273A", "card2": "#313244", "ink": "#CDD6F4", "muted": "#A6ADC8",
            "rule": "#45475A", "accent1": "#89B4FA", "accent2": "#CBA6F7", "accent3": "#F5C2E7",
            "good": "#A6E3A1", "mid": "#89DCEB", "warn": "#F9E2AF", "bad": "#F38BA8",
            "deep": "#74C7EC", "blue": "#89B4FA", "sky": "#89DCEB", "light": "#CBA6F7", "pale": "#F5E0DC",
            "white": "#FFFFFF", "navy": "#1E1E2E",
        },
    },
    "dracula": {
        "label": "Dracula Prism",
        "desc": "Orchid violet → vampiric pink → ice cyan → neon lime",
        "NAVY": 235, "DEEP": 141, "BLUE": 117, "CYAN": 123, "SKY": 117, "VIOLET": 141,
        "PURPLE": 177, "ICE": 159, "PINK": 212, "PALE": 183, "CORAL": 215, "MIST": 195,
        "GREEN": 84, "GREEN_BRIGHT": 120, "YELLOW": 228, "GOLD": 222, "RED": 203, "RED_SOFT": 210,
        "INK": 235, "BAR": 236, "BAR_ALT": 237, "SEL": 238,
        "RULE": 61, "RULE_DIM": 238, "DIM": 103, "MUTED": 146, "SOFT": 253, "LIGHT": 231,
        "GRADIENT": [141, 177, 212, 215, 228, 84, 123, 117, 183, 231],
        "BRAND_GRAD": [61, 99, 141, 177, 212],
        "WAVE_GRAD": [61, 141, 212, 123, 84, 228, 231, 228, 84, 123, 212, 141, 61],
        "MENU_COLORS": [123, 84, 212, 141, 215, 117, 228, 120, 177, 183, 159, 141],
        "PILLAR_COLORS": [141, 84, 212, 123, 228, 215, 177, 117],
        "HEX": {
            "bg": "#282A36", "card": "#2F3241", "card2": "#383B4E", "ink": "#F8F8F2", "muted": "#BD93F9",
            "rule": "#44475A", "accent1": "#BD93F9", "accent2": "#FF79C6", "accent3": "#8BE9FD",
            "good": "#50FA7B", "mid": "#8BE9FD", "warn": "#F1FA8C", "bad": "#FF5555",
            "deep": "#6272A4", "blue": "#8BE9FD", "sky": "#8BE9FD", "light": "#BD93F9", "pale": "#F8F8F2",
            "white": "#FFFFFF", "navy": "#282A36",
        },
    },
    "gruvbox": {
        "label": "Gruvbox Forge",
        "desc": "Warm amber ochre → burnt orange → forest sage → soft cream",
        "NAVY": 234, "DEEP": 172, "BLUE": 109, "CYAN": 108, "SKY": 109, "VIOLET": 175,
        "PURPLE": 175, "ICE": 144, "PINK": 208, "PALE": 223, "CORAL": 167, "MIST": 229,
        "GREEN": 142, "GREEN_BRIGHT": 149, "YELLOW": 214, "GOLD": 222, "RED": 167, "RED_SOFT": 209,
        "INK": 234, "BAR": 235, "BAR_ALT": 236, "SEL": 237,
        "RULE": 95, "RULE_DIM": 237, "DIM": 101, "MUTED": 144, "SOFT": 223, "LIGHT": 229,
        "GRADIENT": [167, 208, 214, 142, 108, 109, 175, 223, 229, 214],
        "BRAND_GRAD": [130, 166, 172, 208, 214],
        "WAVE_GRAD": [95, 167, 208, 214, 142, 223, 229, 223, 142, 214, 208, 167, 95],
        "MENU_COLORS": [214, 142, 208, 108, 175, 109, 223, 149, 167, 222, 144, 108],
        "PILLAR_COLORS": [214, 142, 208, 108, 175, 109, 167, 223],
        "HEX": {
            "bg": "#1D2021", "card": "#282828", "card2": "#32302F", "ink": "#EBDBB2", "muted": "#A89984",
            "rule": "#504945", "accent1": "#FE8019", "accent2": "#FABD2F", "accent3": "#B8BB26",
            "good": "#B8BB26", "mid": "#83A598", "warn": "#FABD2F", "bad": "#FB4934",
            "deep": "#D65D0E", "blue": "#83A598", "sky": "#8EC07C", "light": "#D3869B", "pale": "#EBDBB2",
            "white": "#FBF1C7", "navy": "#1D2021",
        },
    },
    "monokai": {
        "label": "Monokai Pro",
        "desc": "Coral magenta → electric chartreuse → sun yellow → ice cyan",
        "NAVY": 234, "DEEP": 197, "BLUE": 81, "CYAN": 81, "SKY": 117, "VIOLET": 141,
        "PURPLE": 141, "ICE": 159, "PINK": 197, "PALE": 183, "CORAL": 208, "MIST": 230,
        "GREEN": 148, "GREEN_BRIGHT": 154, "YELLOW": 227, "GOLD": 221, "RED": 197, "RED_SOFT": 204,
        "INK": 234, "BAR": 235, "BAR_ALT": 236, "SEL": 238,
        "RULE": 59, "RULE_DIM": 237, "DIM": 102, "MUTED": 145, "SOFT": 253, "LIGHT": 231,
        "GRADIENT": [197, 208, 227, 148, 81, 141, 204, 154, 117, 231],
        "BRAND_GRAD": [125, 161, 197, 208, 227],
        "WAVE_GRAD": [59, 197, 208, 227, 148, 81, 231, 81, 148, 227, 208, 197, 59],
        "MENU_COLORS": [81, 148, 197, 141, 208, 227, 154, 117, 204, 183, 159, 141],
        "PILLAR_COLORS": [197, 148, 81, 227, 141, 208, 154, 117],
        "HEX": {
            "bg": "#1E1F1C", "card": "#272822", "card2": "#31322B", "ink": "#F8F8F2", "muted": "#A59F85",
            "rule": "#49483E", "accent1": "#F92672", "accent2": "#A6E22E", "accent3": "#66D9EF",
            "good": "#A6E22E", "mid": "#66D9EF", "warn": "#E6DB74", "bad": "#F92672",
            "deep": "#AE81FF", "blue": "#66D9EF", "sky": "#66D9EF", "light": "#AE81FF", "pale": "#F8F8F2",
            "white": "#FFFFFF", "navy": "#1E1F1C",
        },
    },
    "nord": {
        "label": "Nord Frost",
        "desc": "Arctic polar slate → frost cyan → glacier blue → aurora green",
        "NAVY": 236, "DEEP": 67, "BLUE": 68, "CYAN": 110, "SKY": 116, "VIOLET": 139,
        "PURPLE": 139, "ICE": 152, "PINK": 175, "PALE": 153, "CORAL": 174, "MIST": 195,
        "GREEN": 108, "GREEN_BRIGHT": 150, "YELLOW": 222, "GOLD": 222, "RED": 167, "RED_SOFT": 174,
        "INK": 235, "BAR": 236, "BAR_ALT": 237, "SEL": 238,
        "RULE": 60, "RULE_DIM": 238, "DIM": 67, "MUTED": 109, "SOFT": 253, "LIGHT": 231,
        "GRADIENT": [67, 68, 74, 109, 110, 116, 152, 108, 139, 175],
        "BRAND_GRAD": [60, 67, 68, 110, 116],
        "WAVE_GRAD": [60, 67, 68, 110, 116, 152, 231, 152, 116, 110, 68, 67, 60],
        "MENU_COLORS": [116, 108, 175, 110, 222, 139, 152, 174, 68, 150, 153, 110],
        "PILLAR_COLORS": [116, 108, 175, 222, 139, 110, 174, 152],
        "HEX": {
            "bg": "#242933", "card": "#2E3440", "card2": "#3B4252", "ink": "#ECEFF4", "muted": "#9AA5B8",
            "rule": "#4C566A", "accent1": "#88C0D0", "accent2": "#81A1C1", "accent3": "#B48EAD",
            "good": "#A3BE8C", "mid": "#88C0D0", "warn": "#EBCB8B", "bad": "#BF616A",
            "deep": "#5E81AC", "blue": "#81A1C1", "sky": "#88C0D0", "light": "#B48EAD", "pale": "#D8DEE9",
            "white": "#ECEFF4", "navy": "#242933",
        },
    },
    "rosepine": {
        "label": "Rosé Pine",
        "desc": "Moonlit pine → dusty rose → iris lavender → foam & warm gold",
        "NAVY": 234, "DEEP": 95, "BLUE": 67, "CYAN": 116, "SKY": 152, "VIOLET": 146,
        "PURPLE": 182, "ICE": 159, "PINK": 217, "PALE": 189, "CORAL": 211, "MIST": 224,
        "GREEN": 109, "GREEN_BRIGHT": 152, "YELLOW": 222, "GOLD": 223, "RED": 204, "RED_SOFT": 211,
        "INK": 234, "BAR": 235, "BAR_ALT": 236, "SEL": 237,
        "RULE": 60, "RULE_DIM": 237, "DIM": 103, "MUTED": 145, "SOFT": 253, "LIGHT": 231,
        "GRADIENT": [95, 132, 175, 217, 222, 152, 116, 146, 182, 224],
        "BRAND_GRAD": [59, 95, 139, 182, 217],
        "WAVE_GRAD": [60, 95, 146, 182, 217, 224, 231, 224, 217, 182, 146, 95, 60],
        "MENU_COLORS": [116, 217, 182, 222, 211, 152, 146, 204, 109, 224, 189, 222],
        "PILLAR_COLORS": [217, 182, 116, 222, 211, 152, 146, 204],
        "HEX": {
            "bg": "#191724", "card": "#1F1D2E", "card2": "#26233A", "ink": "#E0DEF4", "muted": "#908CAA",
            "rule": "#403D52", "accent1": "#EBBCBA", "accent2": "#C4A7E7", "accent3": "#9CCFD8",
            "good": "#9CCFD8", "mid": "#C4A7E7", "warn": "#F6C177", "bad": "#EB6F92",
            "deep": "#31748F", "blue": "#31748F", "sky": "#9CCFD8", "light": "#C4A7E7", "pale": "#E0DEF4",
            "white": "#FFFFFF", "navy": "#191724",
        },
    },
    "kanagawa": {
        "label": "Kanagawa Wave",
        "desc": "Sumi ink → Hokusai crystal wave → sakura pink → bamboo spring",
        "NAVY": 234, "DEEP": 61, "BLUE": 68, "CYAN": 109, "SKY": 110, "VIOLET": 139,
        "PURPLE": 140, "ICE": 152, "PINK": 175, "PALE": 182, "CORAL": 209, "MIST": 223,
        "GREEN": 107, "GREEN_BRIGHT": 149, "YELLOW": 179, "GOLD": 221, "RED": 167, "RED_SOFT": 203,
        "INK": 234, "BAR": 235, "BAR_ALT": 236, "SEL": 237,
        "RULE": 59, "RULE_DIM": 237, "DIM": 102, "MUTED": 144, "SOFT": 223, "LIGHT": 230,
        "GRADIENT": [61, 68, 109, 107, 179, 209, 175, 139, 110, 223],
        "BRAND_GRAD": [60, 61, 68, 109, 175],
        "WAVE_GRAD": [59, 61, 68, 109, 140, 175, 230, 175, 140, 109, 68, 61, 59],
        "MENU_COLORS": [110, 107, 175, 139, 209, 179, 109, 167, 68, 149, 182, 139],
        "PILLAR_COLORS": [110, 175, 107, 179, 139, 209, 109, 167],
        "HEX": {
            "bg": "#1F1F28", "card": "#2A2A37", "card2": "#363646", "ink": "#DCD7BA", "muted": "#938AA9",
            "rule": "#54546D", "accent1": "#7E9CD8", "accent2": "#957FB8", "accent3": "#D27E99",
            "good": "#98BB6C", "mid": "#7AA89F", "warn": "#E6C384", "bad": "#E46876",
            "deep": "#2D4F67", "blue": "#7E9CD8", "sky": "#7FB4CA", "light": "#957FB8", "pale": "#DCD7BA",
            "white": "#F2ECBC", "navy": "#1F1F28",
        },
    },
    "solarized": {
        "label": "Solarized Dark",
        "desc": "Precision ocean teal → cobalt blue → cyan → amber & magenta",
        "NAVY": 23, "DEEP": 24, "BLUE": 33, "CYAN": 37, "SKY": 73, "VIOLET": 61,
        "PURPLE": 125, "ICE": 109, "PINK": 162, "PALE": 152, "CORAL": 166, "MIST": 195,
        "GREEN": 64, "GREEN_BRIGHT": 106, "YELLOW": 136, "GOLD": 178, "RED": 160, "RED_SOFT": 167,
        "INK": 234, "BAR": 235, "BAR_ALT": 236, "SEL": 237,
        "RULE": 23, "RULE_DIM": 237, "DIM": 66, "MUTED": 109, "SOFT": 250, "LIGHT": 230,
        "GRADIENT": [24, 33, 37, 64, 136, 166, 160, 162, 61, 73],
        "BRAND_GRAD": [23, 24, 33, 37, 73],
        "WAVE_GRAD": [23, 24, 33, 37, 73, 152, 230, 152, 73, 37, 33, 24, 23],
        "MENU_COLORS": [37, 64, 162, 33, 166, 136, 73, 125, 61, 106, 109, 33],
        "PILLAR_COLORS": [33, 37, 64, 136, 162, 166, 61, 73],
        "HEX": {
            "bg": "#002B36", "card": "#073642", "card2": "#0E4654", "ink": "#EEE8D5", "muted": "#839496",
            "rule": "#215966", "accent1": "#268BD2", "accent2": "#2AA198", "accent3": "#D33682",
            "good": "#859900", "mid": "#2AA198", "warn": "#B58900", "bad": "#DC322F",
            "deep": "#268BD2", "blue": "#268BD2", "sky": "#2AA198", "light": "#6C71C4", "pale": "#EEE8D5",
            "white": "#FDF6E3", "navy": "#002B36",
        },
    },
    "onedark": {
        "label": "One Dark Pro",
        "desc": "Graphite slate → vivid blue → orchid purple → sage green & chalk gold",
        "NAVY": 234, "DEEP": 68, "BLUE": 75, "CYAN": 73, "SKY": 110, "VIOLET": 176,
        "PURPLE": 176, "ICE": 152, "PINK": 204, "PALE": 183, "CORAL": 173, "MIST": 195,
        "GREEN": 114, "GREEN_BRIGHT": 150, "YELLOW": 180, "GOLD": 222, "RED": 204, "RED_SOFT": 210,
        "INK": 234, "BAR": 235, "BAR_ALT": 236, "SEL": 238,
        "RULE": 59, "RULE_DIM": 237, "DIM": 102, "MUTED": 145, "SOFT": 252, "LIGHT": 231,
        "GRADIENT": [68, 75, 73, 114, 180, 173, 204, 176, 110, 152],
        "BRAND_GRAD": [60, 68, 75, 176, 204],
        "WAVE_GRAD": [59, 68, 75, 73, 176, 204, 231, 204, 176, 73, 75, 68, 59],
        "MENU_COLORS": [75, 114, 176, 73, 180, 204, 173, 110, 150, 183, 152, 73],
        "PILLAR_COLORS": [75, 114, 176, 180, 73, 204, 173, 110],
        "HEX": {
            "bg": "#1E222A", "card": "#282C34", "card2": "#313640", "ink": "#ABB2BF", "muted": "#7F848E",
            "rule": "#3E4451", "accent1": "#61AFEF", "accent2": "#C678DD", "accent3": "#56B6C2",
            "good": "#98C379", "mid": "#56B6C2", "warn": "#E5C07B", "bad": "#E06C75",
            "deep": "#528BFF", "blue": "#61AFEF", "sky": "#56B6C2", "light": "#C678DD", "pale": "#D7DAE0",
            "white": "#FFFFFF", "navy": "#1E222A",
        },
    },
    "synthwave": {
        "label": "Synthwave '84",
        "desc": "Outrun twilight → hot neon magenta → laser cyan → sunset gold",
        "NAVY": 53, "DEEP": 91, "BLUE": 45, "CYAN": 51, "SKY": 87, "VIOLET": 171,
        "PURPLE": 201, "ICE": 123, "PINK": 213, "PALE": 219, "CORAL": 209, "MIST": 225,
        "GREEN": 84, "GREEN_BRIGHT": 120, "YELLOW": 227, "GOLD": 221, "RED": 197, "RED_SOFT": 211,
        "INK": 233, "BAR": 234, "BAR_ALT": 235, "SEL": 237,
        "RULE": 90, "RULE_DIM": 53, "DIM": 133, "MUTED": 176, "SOFT": 254, "LIGHT": 231,
        "GRADIENT": [51, 45, 99, 171, 201, 213, 209, 214, 227, 84],
        "BRAND_GRAD": [53, 91, 165, 201, 51],
        "WAVE_GRAD": [53, 91, 171, 201, 213, 51, 231, 51, 213, 201, 171, 91, 53],
        "MENU_COLORS": [51, 213, 227, 84, 209, 171, 87, 197, 201, 123, 219, 84],
        "PILLAR_COLORS": [213, 51, 227, 84, 209, 171, 197, 87],
        "HEX": {
            "bg": "#1A102B", "card": "#24173B", "card2": "#31204F", "ink": "#F8F2FF", "muted": "#B69AD9",
            "rule": "#4E347A", "accent1": "#FF7EDB", "accent2": "#36F9F6", "accent3": "#FEDE5D",
            "good": "#72F1B8", "mid": "#36F9F6", "warn": "#FEDE5D", "bad": "#FE4450",
            "deep": "#B893CE", "blue": "#36F9F6", "sky": "#36F9F6", "light": "#FF7EDB", "pale": "#FFE4F7",
            "white": "#FFFFFF", "navy": "#1A102B",
        },
    },
}

# Module-level active palette attributes (mutated cleanly by set_theme)
ACTIVE = "aurora"
_ON_SEL: dict[int, int] = {}
NAVY = DEEP = BLUE = CYAN = SKY = VIOLET = PURPLE = ICE = PINK = PALE = CORAL = MIST = 0
GREEN = GREEN_BRIGHT = YELLOW = GOLD = RED = RED_SOFT = 0
INK = BAR = BAR_ALT = SEL = RULE = RULE_DIM = DIM = MUTED = SOFT = LIGHT = 0
GRADIENT: list[int] = []
BRAND_GRAD: list[int] = []
WAVE_GRAD: list[int] = []
MENU_COLORS: list[int] = []
PILLAR_COLORS: list[int] = []
HEX: dict[str, str] = {}


def set_theme(name: str) -> str:
    """Activate one of the built-in themes. Unknown names fall back to `aurora`."""
    global ACTIVE, _ON_SEL
    chosen = name if name in THEMES else "aurora"
    ACTIVE = chosen
    pal = THEMES[chosen]
    g = globals()
    for k, v in pal.items():
        if k not in ("label", "desc"):
            g[k] = list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v)
    if pal["SEL"] == 17:
        _ON_SEL = {
            pal["DEEP"]: pal["ICE"], 26: pal["ICE"], 27: pal["ICE"],
            pal["BLUE"]: pal["ICE"], pal["CYAN"]: pal["PALE"], pal["VIOLET"]: pal["PALE"],
            pal["PURPLE"]: pal["MIST"], pal["RULE"]: pal["MUTED"], pal["RULE_DIM"]: pal["MUTED"],
            pal["DIM"]: pal["MUTED"], pal["MUTED"]: pal["SOFT"],
            pal["RED"]: pal["RED_SOFT"], pal["GREEN"]: pal["GREEN_BRIGHT"],
        }
    else:
        _ON_SEL = {
            pal["RULE"]: pal["MUTED"], pal["RULE_DIM"]: pal["MUTED"],
            pal["DIM"]: pal["SOFT"], pal["MUTED"]: pal["LIGHT"],
            25: pal["ICE"], 26: pal["ICE"], 27: pal["ICE"],
            pal["RED"]: pal["RED_SOFT"], pal["GREEN"]: pal["GREEN_BRIGHT"],
        }
    return chosen


def current_theme() -> str:
    return ACTIVE


def theme_info(name: str | None = None) -> dict:
    key = name if name in THEMES else ACTIVE
    t = THEMES[key]
    return {"name": key, "label": t["label"], "desc": t["desc"], "HEX": t["HEX"]}


def cycle_theme(step: int = 1) -> str:
    idx = (THEME_ORDER.index(ACTIVE) + step) % len(THEME_ORDER)
    return set_theme(THEME_ORDER[idx])


set_theme("aurora")


def xterm_rgb(code: int) -> tuple[int, int, int]:
    """Approximate sRGB (0..255) for an xterm-256 palette index."""
    if code < 0:
        return (20, 22, 28)
    if code < 16:
        basic = [
            (0, 0, 0), (180, 0, 0), (0, 180, 0), (180, 180, 0),
            (0, 0, 180), (180, 0, 180), (0, 180, 180), (200, 200, 200),
            (100, 100, 100), (255, 85, 85), (85, 255, 85), (255, 255, 85),
            (85, 85, 255), (255, 85, 255), (85, 255, 255), (255, 255, 255),
        ]
        return basic[code]
    if code <= 231:
        c = code - 16
        steps = (0, 95, 135, 175, 215, 255)
        return (steps[c // 36], steps[(c // 6) % 6], steps[c % 6])
    v = 8 + (code - 232) * 10
    return (v, v, v)


def xterm_luminance(code: int) -> float:
    """Relative WCAG luminance (0.0 dark .. 1.0 white) of an xterm-256 index."""
    def lin(ch: int) -> float:
        x = ch / 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4
    r, g, b = xterm_rgb(code)
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast_fg(bg: int | None) -> int:
    """Returns high-contrast foreground (`INK` dark vs `LIGHT` white) for any solid pill/badge background."""
    if bg is None:
        return LIGHT
    return INK if xterm_luminance(bg) >= 0.34 else LIGHT


def readable_accent(fg: int, on_bg: int | None = None) -> int:
    """Ensures an accent colour (such as a Home menu number or badge) is neither washed-out white nor invisible dark."""
    lum = xterm_luminance(fg)
    if on_bg is not None:
        bg_lum = xterm_luminance(on_bg)
        hi, lo = max(lum, bg_lum), min(lum, bg_lum)
        if (hi + 0.05) / (lo + 0.05) < 3.0:
            return contrast_fg(on_bg)
    # Avoid blinding near-white (>0.86) or muddy ultra-dark (<0.12) accents on default terminal backgrounds
    if lum > 0.86:
        return CYAN if xterm_luminance(CYAN) <= 0.80 else BLUE
    if lum < 0.12:
        return ICE if xterm_luminance(ICE) <= 0.80 else PALE
    return fg


def on_sel(fg: int | None) -> int | None:
    """Return a high-contrast foreground colour suitable for drawing on T.SEL."""
    if fg is None:
        return LIGHT
    promoted = _ON_SEL.get(fg, fg)
    # Luminance safety net: guarantee ≥3.2:1 contrast against SEL in all 16 themes
    bg_l = xterm_luminance(SEL)
    fg_l = xterm_luminance(promoted)
    hi, lo = max(fg_l, bg_l), min(fg_l, bg_l)
    if (hi + 0.05) / (lo + 0.05) < 3.2:
        return contrast_fg(SEL)
    return promoted


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
    drift = math.sin(tick * 0.065) * 0.14
    diag_span = max(1.0, (cols - 1) + (rows - 1) * 2.4)
    pos = (col + row * 2.4) / diag_span + drift
    pos = max(0.0, min(1.0, pos))
    base_idx = int(pos * (len(GRADIENT) - 1) + 0.5)
    base_idx = max(0, min(len(GRADIENT) - 1, base_idx))

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
    return {"golden": YELLOW, "evals": PURPLE, "perf": GREEN, "preflight": CYAN, "harness": PINK}.get(tier, ICE)


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
    """Distinct, high-contrast colour per Home navigation screen (clamped away from washed-out white or dim dark)."""
    if not MENU_COLORS:
        return CYAN
    return readable_accent(MENU_COLORS[index % len(MENU_COLORS)])
