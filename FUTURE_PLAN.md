# Ghost-Shell Roadmap

Future features and improvements. PRs welcome.

---

## Completed

### Auto-Detect Opponent Moves
~~Currently you have to type opponent moves manually.~~ **DONE.** The bot now automatically detects opponent moves by comparing board state snapshots. Uses multi-sample voting, highlight-aware color classification, and handles normal moves, captures, castling, and en passant. Falls back to manual input if detection fails.

### Opening Book Integration
~~Instead of calculating from move 1, use known opening theory.~~ **DONE.** Hardcoded opening database in `core/openings.py` detects common openings (Sicilian, Queen's Gambit, Italian, French, Caro-Kann, etc.) and displays the name during play.

### Move History Export
~~Log all games played for review.~~ **DONE.** Full move history displayed in the console during play in algebraic notation.

---

## High Priority

### Multi-Monitor Support
Right now it scans monitor 1. Should detect which monitor has the chess board, or let the user pick.

### Piece Type Recognition
Currently the bot only detects piece presence and color (white/black), not what type of piece it is. Adding piece type detection (via template matching or ML) would enable:
- Smarter move validation
- Detection of illegal states
- Better disambiguation when multiple pieces could have moved

---

## Medium Priority

### Time Control Awareness
Adjust think time based on clock. Blitz = faster moves, classical = longer thinks.

### Pre-Move Detection
Detect opponent premoves and react accordingly.

### Adaptive Calibration
Re-calibrate square colors periodically during the game to handle changing lighting conditions or board themes that shift.

---

## Low Priority / Nice to Have

### Profile System
Save different play styles:
- Aggressive (high contempt)
- Solid (lower depth, more draw-ish)
- Blunder Mode (intentionally make occasional mistakes)

### Cross-Platform Support
Currently Windows-only (uses pywin32 for overlay). Add macOS/Linux support.

### Lichess API Integration
Instead of screen reading, connect directly to Lichess API for cleaner detection.

### GUI Settings Panel
A simple tkinter window to adjust thresholds, engine depth, and other settings without editing config files.

---

## Known Issues

- Side auto-detection isn't always accurate (use W/B manual selection for best results)
- Very fast opponents might trigger movement detection prematurely
- Promotion popup click position may vary by site/board theme
- Unusual board themes or very low contrast piece sets may produce '?' detections
- Yellow/green move highlights can occasionally shift color readings on highlighted squares

---

## Contributing

1. Fork the repo
2. Create feature branch
3. Make changes
4. Open PR

No formal process, just dont break stuff.
