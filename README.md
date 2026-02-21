# GHOST-SHELL

```
   ▄████  ██░ ██  ▒█████    ██████ ▄▄▄█████▓
  ██▒ ▀█▒▓██░ ██▒▒██▒  ██▒▒██    ▒ ▓  ██▒ ▓▒
 ▒██░▄▄▄░▒██▀▀██░▒██░  ██▒░ ▓██▄   ▒ ▓██░ ▒░
 ░▓█  ██▓░▓█ ░██ ▒██   ██░  ▒   ██▒░ ▓██▓ ░
 ░▒▓███▀▒░▓█▒░██▓░ ████▓▒░▒██████▒▒  ▒██▒ ░
  ░▒   ▒  ▒ ░░▒░▒░ ▒░▒░▒░ ▒ ▒▓▒ ▒ ░  ▒ ░░
        S H E L L   -   C H E S S
```

---

## Why does this exist?

My friend kept beating me at chess. So i challenged him again and told him if he was able to beat me, i would build this. Bruhhh, He beat me again in 4 moves. 4. No Way i was letting that slide. So Yeah, here we are"

---

## What is it?

Ghost-Shell watches your screen, finds the chessboard, thinks with Stockfish, then moves the mouse like a human would. Bezier curves, random delays, the whole thing. It auto-detects opponent moves, shows you the engine evaluation, and even tells you what opening you're in.

---

## Features

- **Auto Move Detection** - detects opponent moves automatically by tracking piece positions and colors across frames. No more typing moves manually.
- **Board Calibration** - learns the exact light/dark square colors of your board theme at startup for accurate piece detection
- **Highlight-Aware Vision** - handles Lichess/chess.com yellow/green move highlights without false detections (uses HSV saturation filtering)
- **Board Orientation** - works correctly whether you're playing as white or black (flipped board)
- **Evaluation Bar** - shows the engine's position eval after every move (e.g. +1.2 = white is winning)
- **Move History** - displays the full game in algebraic notation as you play
- **Opening Recognition** - detects known openings (Sicilian, Queen's Gambit, Italian, etc.)
- **Legal Moves Display** - shows all available moves on your turn
- **Turing Filter** - sometimes picks the 2nd best move so it doesn't look like an engine
- **Human Mouse Movement** - bezier curves, jitter, variable click timing
- **Auto Side Detection** - figures out if you're white or black
- **Think Time** - waits 1.5-6 seconds before moving like a real person
- **Promotion Clicks** - clicks the popup menu, not keyboard shortcuts
- **HUD Overlay** - transparent green arrow shows the move before it happens
- **Castling Detection** - correctly identifies castling moves (king + rook swap)
- **Debug Snapshots** - press `D` during a game to save an annotated image showing what the bot sees on every square
- **Manual Fallback** - if auto-detection fails, you can still type the move manually. Type `z` to undo a misdetected move.

---

## Install

```bash
git clone https://github.com/jeffharr1s/ghost-shell.git
cd ghost-shell
pip install -r requirements.txt
```

Download stockfish from https://stockfishchess.org/download/ and put `stockfish.exe` in the `assets` folder.

> **Note:** `python-dotenv` and `pywin32` are optional. The bot works without them - dotenv is only needed if you want to use a `.env` config file, and pywin32 is only for the transparent overlay (falls back to a basic overlay without it).

---

## Usage

```bash
python main.py
```

Or use the included launcher:
```bash
run_bot.bat
```

1. Open Lichess, chess.com, or any chess site
2. Make sure the full board is visible on screen
3. Choose **W/B/A** when prompted (White / Black / Auto-detect)
4. Press **S** to start
5. The bot finds the board, calibrates square colors, and starts playing

Press **Q** to quit at any time.

---

## How Opponent Move Detection Works

The bot automatically detects when your opponent moves:

1. **Watches for screen changes** - detects pixel differences in the board region
2. **Scans all 64 squares** - identifies which squares have pieces and classifies them as white (w) or black (b)
3. **Compares before/after** - figures out which piece moved where by diffing the two board states
4. **Handles special moves** - castling (2 pieces move), en passant (captured pawn disappears), captures (color changes on a square)

If auto-detection fails (e.g. unusual board theme or lighting), it falls back to manual input. You can also type **z** to undo a misdetected move.

---

## In-Game Controls

| Key | When | Action |
|-----|------|--------|
| **S** | At startup | Start board detection |
| **Q** | During game | Quit the bot |
| **D** | During game | Save debug snapshot (`debug_pieces.jpg`) |
| **Space** | After bot moves | Retry the move if it didn't register on screen |
| **z** | At move input | Undo the last opponent move |

---

## Config

Edit `.env` (optional - defaults are fine):

```env
ENGINE_DEPTH=15          # higher = stronger (but slower)
ENGINE_CONTEMPT=20       # higher = more aggressive style

MOUSE_SPEED=0.4
TARGET_JITTER=6          # pixels of randomness
THINK_TIME_MIN=1.5       # seconds before moving
THINK_TIME_MAX=6.0

PLAYER_SIDE=AUTO         # AUTO / WHITE / BLACK
```

---

## Project Structure

```
ghost-shell/
├── main.py              # game loop, orchestrates everything
├── core/
│   ├── vision.py        # screen capture, board detection, piece classification
│   ├── engine.py        # stockfish wrapper, returns move + eval score
│   ├── humanizer.py     # bezier mouse movement
│   └── openings.py      # opening recognition database
├── ui/
│   └── overlay.py       # transparent HUD with move arrows
├── utils/
│   ├── config.py        # loads .env (optional)
│   └── logger.py        # colored console output
├── assets/
│   └── stockfish.exe    # download separately
├── run_bot.bat           # windows launcher
├── launch_bot.vbs        # silent launcher (no console flash)
└── .env                  # optional config overrides
```

---

## Debug

### Debug Snapshot

Press **D** during a game to save `debug_pieces.jpg`. This shows:
- The detected board grid overlay
- Each square labeled with its chess name and detected piece color (`w`/`b`/`?`/empty)
- White boxes around white pieces, black boxes around black pieces, red boxes around uncertain pieces

### Board Grid Check

```bash
python -c "from core.vision import GhostVision; v = GhostVision(); v.find_board(); v.debug_draw_board()"
```

Saves `debug_vision.jpg` with the detected grid and square labels.

### Full Piece Detection Test

```bash
python core/vision.py
```

Waits 3 seconds, then finds the board, detects all pieces, prints the piece map, and saves `debug_pieces.jpg`.

---

## Disclaimer

This is for educational purposes. Using this on chess sites is against their TOS and is cheating. Dont be that guy. Or do, I'm not your mom.

---

MIT License - do whatever
