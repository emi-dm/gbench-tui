# gbench-tui (`gbt`) demo: `vhs assets/demo.tape` from the repo root
Output assets/demo-aurora.gif

Set Shell "bash"
Set FontSize 15
Set Width 1280
Set Height 780
Set Padding 14
Set Framerate 12
Set TypingSpeed 55ms
Set Theme { "name": "Gemini Aurora", "background": "#0B0E17", "foreground": "#EEF2FF", "cursor": "#F472B6", "selection": "#27314F", "black": "#0B0E17", "red": "#F87171", "green": "#34D399", "yellow": "#FBBF24", "blue": "#4F7CFF", "magenta": "#C084FC", "cyan": "#38BDF8", "white": "#EEF2FF", "brightBlack": "#64748B", "brightRed": "#FB7185", "brightGreen": "#6EE7B7", "brightYellow": "#FDE047", "brightBlue": "#60A5FA", "brightMagenta": "#F472B6", "brightCyan": "#67E8F9", "brightWhite": "#FFFFFF" }

Hide
Type "export PYTHONPATH=src TERM=xterm-256color; alias gbt='python3 -m gbtui'; clear"
Enter
Sleep 400ms
Show

Type "gbt"
Sleep 350ms
Enter
Sleep 2.0s

# Live theme picker (T) with real-time preview across 16 palettes
Type "T"
Sleep 900ms
Down 5
Sleep 700ms
Down 2
Sleep 700ms
Down 3
Sleep 700ms
Down 5
Sleep 800ms
Escape
Sleep 700ms

# 1 · Endpoints (GGUF + harness-hub LoRAs + RAM footprint)
Type "1"
Sleep 1.8s
Down
Sleep 900ms
Down 2
Sleep 900ms
Type "q"
Sleep 400ms

# 2 · Run (presets, Harbor tool_mode, preflight & duration estimate)
Type "2"
Sleep 1.8s
Type "q"
Sleep 400ms

# 7 · Harness Lens (anatomy, full System Prompt `p`, side-by-side compare `c`)
Type "7"
Sleep 1.6s
Down 2
Sleep 1.2s
Type "p"
Sleep 1.6s
Type "q"
Sleep 400ms
Type "c"
Sleep 1.5s
Type "q"
Sleep 400ms

# 5 · Results (Harbor Code Mode + KV cache telemetry & per-sample inspector)
Type "5"
Sleep 1.5s
Enter
Sleep 1.8s
Type "q"
Sleep 400ms
Type "q"
Sleep 400ms

# 6 · Leaderboard (Rankings + `H` Adapter x Harness Heatmap Matrix)
Type "6"
Sleep 1.8s
Type "H"
Sleep 2.2s
Type "q"
Sleep 400ms

# d · Modular Setup & Hardened Docker Sandbox
Type "d"
Sleep 1.6s
Type "q"
Sleep 400ms

# Instant quit
Type "q"
Sleep 400ms
