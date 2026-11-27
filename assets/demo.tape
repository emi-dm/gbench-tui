# gbench-tui demo: `vhs assets/demo.tape` from the repo root
Output assets/demo.gif

Set Shell "bash"
Set FontSize 15
Set Width 1280
Set Height 780
Set Padding 14
Set Framerate 12
Set TypingSpeed 55ms
Set Theme { "name": "Gemini Aurora", "background": "#0B0E17", "foreground": "#EEF2FF", "cursor": "#F472B6", "selection": "#27314F", "black": "#0B0E17", "red": "#F87171", "green": "#34D399", "yellow": "#FBBF24", "blue": "#4F7CFF", "magenta": "#C084FC", "cyan": "#38BDF8", "white": "#EEF2FF", "brightBlack": "#64748B", "brightRed": "#FB7185", "brightGreen": "#6EE7B7", "brightYellow": "#FDE047", "brightBlue": "#60A5FA", "brightMagenta": "#F472B6", "brightCyan": "#67E8F9", "brightWhite": "#FFFFFF" }

Hide
Type "export PYTHONPATH=src TERM=xterm-256color; alias gbench-tui='python3 -m gbtui'; clear"
Enter
Sleep 400ms
Show

Type "gbench-tui"
Sleep 350ms
Enter
Sleep 2.4s

# Live prism theme switcher on Home (t cycles all 5 palettes)
Type "t"
Sleep 1.1s
Type "t"
Sleep 1.1s
Type "t"
Sleep 0.7s
Type "t"
Sleep 0.7s
Type "t"
Sleep 1.2s

# 1 · Endpoints (with quantization & GGUF RAM footprint)
Type "1"
Sleep 2.2s
Down
Sleep 1.2s
Type "q"
Sleep 500ms

# 2 · Run (command preview & host memory guard)
Type "2"
Sleep 2.2s
Type "q"
Sleep 500ms

# 4 · Suites (pillars, prerequisites & per-benchmark metric badge)
Type "4"
Sleep 1.4s
Down 3
Sleep 1.4s
Type "q"
Sleep 500ms

# m · Metrics & Formulas (Core Formulas → By Benchmark → Quantization bpw table)
Type "m"
Sleep 2s
Type "2"
Sleep 1.6s
Down 3
Sleep 1.8s
Type "b"
Sleep 1.8s
Type "q"
Sleep 400ms
Type "q"
Sleep 500ms

# 5 · Results → Run detail → Samples → Sample inspector
Type "5"
Sleep 1.8s
Down
Sleep 0.6s
Enter
Sleep 1.8s
Down
Sleep 0.6s
Enter
Sleep 2s
Down 2
Sleep 0.7s
Enter
Sleep 2.2s
Type "q"
Sleep 400ms
Type "q"
Sleep 400ms
Type "q"
Sleep 400ms
Type "q"
Sleep 500ms

# 6 · Leaderboard (ranked variants with quantization column & heatmap)
Type "6"
Sleep 2.8s
Down
Sleep 1s
Type "q"
Sleep 400ms
Type "q"
Sleep 600ms
