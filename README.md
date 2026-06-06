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

Ghost-Shell watches your screen, finds the chessboard, thinks with Stockfish, then moves the mouse like a human would. Bezier curves, random delays, the whole thing.

---

## Features

- **Vision** - finds any chessboard on screen via opencv
- **Turing Filter** - sometimes picks the 2nd best move so it doesnt look like an engine  
- **Human mouse movement** - curves, jitter, variable click timing
- **Auto side detection** - figures out if youre white or black
- **Think time** - waits 1.5-6 seconds before moving like a real person would
- **Promotion clicks** - clicks the popup menu, not keyboard shortcuts
- **HUD Overlay** - transparent green arrow shows the move before it happens

---

## Install

```bash
git clone https://github.com/zoecyber001/ghost-shell.git
cd ghost-shell
pip install -r requirements.txt
python -m playwright install chromium   # only needed for browser mode (see below)
```

Download stockfish from https://stockfishchess.org/download/ and put `stockfish.exe` in the `assets` folder.

---

## Usage

```bash
python main.py
```

1. Open chess.com or lichess or whatever
2. Make sure the board is visible
3. Paste a FEN and press Enter to resume, or press Enter on a blank prompt for a new game
4. Let it auto-detect the board and your side from the bottom of the board
5. Let it do its thing

Press `Q` to quit, or `N` while waiting for the opponent to restart `main.py` for a new game. In manual move prompts, type `new` or `restart` to relaunch. Or slam mouse to corner - theres a failsafe.

---

## Browser mode (keeps your mouse + terminal free)

By default the bot drives your real mouse, which steals focus from the terminal on every move. Set `USE_BROWSER=true` in `.env` to switch to **CDP browser mode**: the bot launches Chrome (your real Google Chrome) as an independent process and injects clicks straight into the page, so the hardware cursor never moves and the terminal keeps focus - you can keep hitting `Enter`/`y`/`r` without clicking back.

```env
USE_BROWSER=true
CHESS_URL=https://www.chess.com
BROWSER_CHANNEL=chrome   # real Chrome; "msedge" or blank (bundled Chromium) also work
CDP_PORT=9222            # debug port the bot connects to
```

Then:

1. `python main.py` - a Chrome window opens to chess.com
2. Log in (saved in `browser_profile/`, so only the first time) and open a game
3. The bot waits until it sees a board, then plays

**Chrome is yours to close.** Because it's launched detached, closing the bot only *disconnects* - Chrome stays open and you close it whenever you like. The next `python main.py` **reconnects** to that same open Chrome (same login, same game, no profile-lock hassle).

**Keep that browser window visible on screen** (it can be behind the focused terminal, just not minimized or covered) - vision still reads the board from screen pixels. First run needs `python -m playwright install chromium` (used as a fallback if Chrome isn't found).

---

## Entering Opponent Moves

When its your opponent's turn, you need to type their move. Format:

```
[from square][to square]
```

Examples:
- `e2e4` - pawn moves from e2 to e4
- `g1f3` - knight from g1 to f3  
- `e1g1` - castles kingside

Files are a-h (left to right), ranks are 1-8 (bottom to top for white). If you cant figure out `e2e4`, maybe stick to tic-tac-toe.

The bot shows legal moves to help. Uppercase works too.

**Coming soon:** Auto-detect opponent moves. For now, you gotta type em.

---

## Config

Edit `.env`:

```env
ENGINE_DEPTH=15          # higher = stronger
ENGINE_CONTEMPT=20       # higher = more aggressive

MOUSE_SPEED=0.4
TARGET_JITTER=6          # pixels
THINK_TIME_MIN=1.5       # seconds  
THINK_TIME_MAX=6.0

PLAYER_SIDE=AUTO         # AUTO / WHITE / BLACK

USE_BROWSER=false        # true = CDP browser clicks (no mouse); false = pyautogui mouse
CHESS_URL=https://www.chess.com  # page the browser opens in browser mode

QUICK_START=false        # true skips the first FEN/new-game prompt
DEFAULT_GAME_MODE=RAPID  # BLITZ / RAPID / CLASSIC
DEFAULT_START_MODE=N     # N = new game, F = resume from FEN
PAUSE_ON_EXIT=true       # keep console open after errors/exits
```

The current run log is always written to `COPY_THIS_LOG.txt` in the project folder. If the app crashes, paste that one file into chat so it can be reviewed.

Startup now defaults hard toward speed: paste FEN to resume, blank Enter means new game, game mode comes from `DEFAULT_GAME_MODE`, and your side is inferred from whichever color is at the bottom of the visible board.

---

## Project Structure

```
ghost-shell/
├── main.py              # orchestrator
├── core/
│   ├── vision.py        # screen capture, board detection
│   ├── engine.py        # stockfish wrapper
│   ├── humanizer.py     # mouse movement (legacy/pyautogui mode)
│   └── browser.py       # Playwright/CDP move injection (browser mode)
├── ui/
│   └── overlay.py       # transparent HUD with move arrows
├── utils/
│   ├── config.py        # loads .env
│   └── logger.py        # colored output
├── assets/
│   └── stockfish.exe
└── .env
```

---

## Debug

Check if vision is working:

```bash
python -c "from core.vision import GhostVision; v = GhostVision(); v.find_board(); v.debug_draw_board()"
```

Saves `debug_vision.jpg` with the detected grid.

---

## Disclaimer

This is for educational purposes. Using this on chess sites is against their TOS and is cheating. Dont be that guy. Or do, I'm not your mom.

---

MIT License - do whatever
