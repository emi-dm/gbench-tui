# gbench-tui demo: `vhs assets/demo.tape` from the repo root (needs sbx signed in, the template built, LFM3 served)
Output assets/demo.gif

Set Shell "bash"
Set FontSize 15
Set Width 1280
Set Height 780
Set Padding 16
Set Framerate 12
Set TypingSpeed 60ms
Set Theme { "name": "DeepMind", "background": "#0E1117", "foreground": "#E8EAED", "cursor": "#8AB4F8", "selection": "#28304A", "black": "#0E1117", "red": "#F28B82", "green": "#81C995", "yellow": "#FDD663", "blue": "#4285F4", "magenta": "#9177C7", "cyan": "#24C1E0", "white": "#E8EAED", "brightBlack": "#5F6368", "brightRed": "#F6AEA9", "brightGreen": "#A8DAB5", "brightYellow": "#FDE293", "brightBlue": "#8AB4F8", "brightMagenta": "#C58AF9", "brightCyan": "#78D9EC", "brightWhite": "#FFFFFF" }

Hide
Type "export PYTHONPATH=src TERM=xterm-256color; alias gbench-tui='python3 -m gbtui'; clear"
Enter
Sleep 500ms
Show

Type "gbench-tui"
Sleep 400ms
Enter
Sleep 3s

# endpoints: the LFM3 GGUF, served by llama-server on demand
Type "1"
Sleep 3s
Type "q"
Sleep 600ms

# gbench's suites by pillar, with what each needs
Type "4"
Sleep 1.5s
Down 6
Sleep 1.5s
Type "q"
Sleep 600ms

# run the Golden Set in a fresh Docker Sandbox
Type "2"
Sleep 3s
Up
Sleep 800ms
Enter
Sleep 50s

# results of the run just finished: scores → samples → one sample
Type "r"
Sleep 2.5s
Enter
Sleep 2.5s
Down 7
Sleep 1s
Enter
Sleep 3.5s
Type "q"
Sleep 500ms
Type "q"
Sleep 500ms
Type "q"
Sleep 500ms
Type "q"
Sleep 500ms

# every run so far, then the leaderboard
Type "5"
Sleep 2.5s
Type "q"
Sleep 500ms
Type "6"
Sleep 4s
Type "q"
Sleep 500ms
Type "q"
Sleep 1s
