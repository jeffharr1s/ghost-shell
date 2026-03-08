# Ghost-Shell Project Specification Export

## 0. Repo Metadata

- **Repo root:** `C:\Users\jeff\ghost-shell\`
- **Entry point:** `ghost-shell/main.py` (run `python main.py` from project root)
- **Python version:** 3.7+ (inferred from f-strings and dict methods)
- **Platform:** Windows only (pywin32, pyautogui, stockfish.exe)
- **External Dependencies:**
  - `opencv-python` (cv2) - board detection, image processing
  - `python-chess` - move validation, board state
  - `stockfish` - engine wrapper (requires `assets/stockfish.exe`)
  - `keyboard` - global keypress detection
  - `pyautogui` - mouse movement
  - `mss` - screen capture
  - `numpy` - image arrays
  - `pywin32` (optional) - transparent overlay
  - `python-dotenv` (optional) - .env config loading

---

## 1. High Level Purpose

- Watches chess board on screen via OpenCV and automatically detects opponent moves by comparing frame differences
- Reads board state (piece positions, colors) via brightness thresholding and HSV-based highlighting
- Orchestrates chess engine (Stockfish) to generate moves via `chess.Board` and UGI interface
- Executes bot moves using humanized mouse control (Bezier curves, jitter, realistic timing)
- Provides fallback yellow highlight detection when auto-detection fails (reads board from highlighted squares)
- Supports white/black side detection and handles board orientation (flipped boards)
- Displays move evaluation, opening recognition (hardcoded database), and legal moves in console
- Implements "Turing Filter" (30% chance to pick 2nd-best move if top 2 moves are close in score)
- Handles special moves: castling, en passant, pawn promotion via popup clicking

---

## 2. Execution Entry Points

### Main Entry Point: `ghost-shell/main.py`

**Main function:** `GhostShell.run()` (line 98)

**CLI args/env vars:**
- None (interactive console prompts)
- Respects env vars via `utils/config.py`: `ENGINE_DEPTH`, `ENGINE_CONTEMPT`, `MOUSE_SPEED`, `TARGET_JITTER`, `THINK_TIME_MIN`, `THINK_TIME_MAX`, `PLAYER_SIDE`

**Primary runtime loop:** (lines 153-232)
```
while not board.is_game_over():
  if board.turn == user_side:
    → sleep(random think time)
    → get_human_move() from engine
    → humanizer.make_move()
    → board.push_uci()
  else:
    → wait_for_opponent_move() [pixel diff detection, max 20 attempts × 0.8s = 16s]
    → if movement detected: prompt user to enter move
    → if "yellow" (user pressed Y): detect_yellow_highlights()
    → else: fallback manual input
```

**Keyboard controls:**
- **S** at startup: start board detection (blocking `keyboard.wait('s')`)
- **Q** anytime: quit (checked every loop iteration)
- **Y** during opponent's turn: trigger yellow highlight detection (checked in `wait_for_opponent_move()`)

---

## 3. Module Map

| Module | File | Responsibility | Key Classes/Functions | Called By |
|--------|------|-----------------|----------------------|-----------|
| **GhostShell** | `main.py` | Game loop, orchestration, side selection, move execution | `GhostShell`, `run()`, `wait_for_opponent_move()`, `get_square_center()` | Entry point |
| **GhostVision** | `core/vision.py` | Board detection, piece detection, yellow highlighting | `GhostVision`, `find_board()`, `detect_yellow_highlights()`, `detect_player_side()`, `get_square_roi()` | GhostShell |
| **GhostEngine** | `core/engine.py` | Stockfish wrapper, move selection, Turing Filter | `GhostEngine`, `get_human_move()` | GhostShell |
| **Humanizer** | `core/humanizer.py` | Realistic mouse movement (Bezier), clicking, promotion selection | `Humanizer`, `make_move()`, `move_mouse()`, `click()`, `_click_promotion_piece()` | GhostShell |
| **GhostOverlay** | `ui/overlay.py` | Transparent HUD, move arrow drawing | `GhostOverlay`, `draw_move_arrow()`, `clear()`, `update_geometry()` | GhostShell |
| **openings** | `core/openings.py` | Hardcoded opening recognition database | `OPENINGS_DB` dict (6 opening branches) | UNUSED in current code |
| **Logger** | `utils/logger.py` | Colored console output (ANSI codes) | `Logger`, `log()`, `success()`, `warning()`, `error()`, `debug()` | All modules |
| **config** | `utils/config.py` | Loads .env or env vars, defaults | Constants: `ENGINE_DEPTH`, `ENGINE_CONTEMPT`, `MOUSE_SPEED`, `THINK_TIME_MIN/MAX`, `PLAYER_SIDE` | All modules |

---

## 4. Opponent Move Detection Spec (Authoritative)

**Function:** `GhostShell.wait_for_opponent_move()` (main.py:42-96)

**Trigger conditions:**
- Called when `board.turn != user_side` (main.py:188)
- Monitors for **pixel change > 1000 non-zero pixels** after binary threshold

**Loop mechanics:**
1. Sleep 1.0s to let own animation finish (line 48)
2. Capture baseline frame (line 49)
3. Loop up to 20 attempts (lines 54-80):
   - Sleep 0.8s (line 65)
   - Capture current frame
   - Compute `cv2.absdiff(prev_frame, curr_frame)` (line 68)
   - Convert to grayscale (line 69)
   - Binary threshold at value 30 (line 70)
   - Count non-zero pixels (line 71)
   - **If count > 1000: movement detected** (line 73) → sleep 1.0s → return `True` (line 76)
   - Update baseline, increment attempt (lines 79-80)

**Failure condition:**
- After 20 attempts with no detection, exit first loop
- Log: "Move detection failed after 20 attempts" (line 83)
- Log: "Press 'Y' to manually detect from yellow highlights" (line 84)
- Enter infinite fallback loop (lines 86-96) waiting only for 'q' (quit) or 'y' (yellow)

**Return values:**
- `True`: Movement detected → main loop prompts for manual move input
- `"yellow"`: Yellow button pressed → main loop calls `detect_yellow_highlights()`
- `None`: Quit ('q' pressed) → main loop breaks and ends game

**Debouncing & keypress handling:**
- 'y' press returns "yellow" with 0.5s debounce (lines 60-63, 91-94)
- 'q' press returns None immediately (lines 55-57, 87-89)
- Both checked on every loop iteration

**Constants:**
- `max_attempts = 20` (line 52)
- Sleep per attempt: 0.8s (line 65)
- After-detection wait: 1.0s (line 75)
- Pixel threshold: 1000 non-zero (line 73)
- Binary threshold: 30 (line 70)
- Debounce delay: 0.5s (line 62)

---

## 5. Yellow Highlight Detection (Authoritative)

**Function:** `GhostVision.detect_yellow_highlights()` (vision.py:211-310)

**Color space & thresholds:**
- Converted to HSV via `cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)` (line 233)
- Lower bound: `[15, 100, 100]` (Hue, Saturation, Value)
- Upper bound: `[45, 255, 255]` (yellow hue range ~20-40)
- Creates binary mask: `cv2.inRange(hsv, lower_yellow, upper_yellow)` (line 241)

**Square scanning method:**
1. Loops all 64 squares: `for rank_idx in range(8): for file_idx in range(8)` (lines 247-248)
2. Computes pixel coords from board_location and square_size: `(x1, y1, x2, y2)` (lines 249-252)
3. Extracts mask region: `sq_mask = yellow_mask[y1:y2, x1:x2]` (line 255)
4. Counts yellow pixels: `yellow_count = cv2.countNonZero(sq_mask)` (line 258)
5. Converts array indices to chess notation: `file = chr(ord('a') + file_idx)`, `rank = str(8 - rank_idx)` (lines 263-264)

**Threshold for "square is yellow":**
- `if yellow_count > (sq_size * sq_size * 0.3)` (line 261) → 30% yellow coverage

**From/To decision logic:**
1. For each yellow square, check brightness to determine if piece is present:
   - Extract square image, convert to grayscale, compute mean (lines 269-271)
   - `has_piece = avg_brightness < 100` (line 272)
2. Separate into `empty_squares` and `piece_squares` (lines 285-286)
3. Three cases:
   - **Perfect:** 1 empty + 1 piece → FROM = empty, TO = piece (lines 288-293)
   - **Capture:** 0 empty + 2 pieces → FROM = first detected, TO = second (lines 294-301)
   - **Ambiguous:** Other combinations → FROM = first, TO = second, log warning (lines 302-308)

**Capture handling assumption:**
- When piece is captured, it disappears from board
- Both source and destination appear highlighted
- If both have pieces, assumes capture scenario (uses first two detected squares in order)

**Return value & consumption:**
- Returns tuple `(from_square_str, to_square_str)` in chess notation (e.g., `('e2', 'e4')`) or `None` if detection fails
- Main loop (line 200): calls `detect_yellow_highlights()`, unpacks as `from_sq, to_sq`, builds `move_uci = from_sq + to_sq`
- Validates move via `board.push_uci(move_uci)` (line 207)
- Falls back to manual input if detection returns None (lines 213-221)

---

## 6. Vision Pipeline

**Board Finding Method** (lines 19-37):
1. Try color-based detection first (line 24): `_find_board_by_colors()`
2. Fallback to corner detection (line 29): `_find_board_by_corners()`

**Color-based detection** (lines 39-87):
- Grayscale conversion
- Gaussian blur (kernel 5×5)
- Canny edge detection (thresholds 50, 150)
- Find contours, filter for large squares (area > 10000, aspect ratio 0.8-1.2)
- Verify each candidate is chessboard by sampling all 64 squares for alternating brightness (line 81)
- Returns first verified candidate as `(x, y, w, h)`
- Sets `self.board_location = (x, y, w, h)` and `self.square_size = w / 8`

**Corner-based detection** (lines 135-162):
- Detect 7×7 grid of corners via `cv2.findChessboardCorners(gray, (7, 7), None)`
- Compute bounding box of corners
- Estimate board size from corner spacing
- Sets board_location and square_size

**Coordinate mapping:**
- Board location tuple: `(bx, by, bw, bh)` = (top-left x, top-left y, width, height)
- Square location: `rank_idx` (0-7 from top), `file_idx` (0-7 from left)
- Pixel coords: `x = bx + file_idx * sq_size`, `y = by + rank_idx * sq_size`
- Chess notation: `file = 'a' + file_idx`, `rank = 8 - rank_idx` (for white perspective)

**Debug drawing & outputs:**
- `debug_draw_board()` (lines 312-331): Draws grid overlay on board, saves as `debug_vision.jpg`
- Yellow detection logs each detected square with piece presence
- Movement detection logged on successful detection

---

## 7. Board State + Chess Rules Integration

**Python-chess usage:**
- `self.board = chess.Board()` initializes standard starting position (main.py:21)
- `board.turn` returns `chess.WHITE` or `chess.BLACK` (boolean check main.py:155)
- `board.push_uci(move_str)` validates move in UCI notation (e.g., 'e2e4') and updates board state (main.py:207, 228)
- `board.is_game_over()` checks for checkmate/stalemate (main.py:153)
- `board.legal_moves` generator lists all legal moves (main.py:164, 191)
- `board.fen()` returns board state as FEN string, passed to engine (main.py:160)

**Move validation:**
- All moves validated via `board.push_uci()` with try/except catching `ValueError` on illegal moves
- If detection provides invalid move, error logged and user prompted to re-enter

**Side to move tracking:**
- User selects W/B/A at startup (main.py:109-117)
- `self.user_side` stored as `chess.WHITE` or `chess.BLACK`
- Game loop checks `board.turn == user_side` to decide whose turn (main.py:155)

**Captures/Castling/Promotion:**
- Captures: Board state automatically updated by `chess` library, no special handling
- Castling: Detected as king move, `board.push_uci()` correctly recognizes as castling move
- Promotion: Move string includes promotion piece (e.g., 'e7e8q'), main.py passes to `humanizer.make_move()` which clicks promotion popup

---

## 8. Error Handling + Logging

**Logging framework:** Custom `Logger` class (utils/logger.py)
- ANSI colored output: cyan (info), green (success), yellow (warning), red (error), blue (debug)
- Format: `[HH:MM:SS] [MODULE_NAME] message`
- Methods: `log()`, `success()`, `warning()`, `error()`, `debug()`

**Common failure modes detectable from code:**
1. Board not found: `find_board()` returns None → log error, exit (main.py:124-126)
2. Stockfish not found: `assets/stockfish.exe` missing → engine.py:18 logs error and calls `sys.exit(1)`
3. Move detection timeout: 20 attempts fail → user must press Y for yellow or enter move manually
4. Yellow detection fails: No 2+ yellow squares found → returns None, falls back to manual input
5. Invalid move entered: `board.push_uci()` raises ValueError → caught, error logged, user reprompted
6. Promotion popup miss: Hardcoded offset may be wrong → no error, just wrong piece clicked (silent failure)

**Retries/timeouts:**
- Move detection: 20 attempts × 0.8s = 16s, then infinite wait loop (no timeout)
- Yellow detection: Called once on user request, returns None or result (no retry)
- Stockfish: Error handling wraps engine call, logs exception, returns None (main.py:161 checks for None)

---

## 9. Incomplete, Dead, or Experimental Code

**Orphaned files:**
- `core/vision2.py` - Exists but completely unused (not imported anywhere)
  - Contains partial implementations: `get_board_piece_map()`, `detect_move_from_maps()`, `detect_opponent_move_uci()`
  - Appears to be abandoned experimental piece detection
  - Should be deleted before v1.0

**Unused modules:**
- `core/openings.py` - Hardcoded opening database exists but is never imported or used in main game loop
- `ui/overlay.py` - HUD overlay drawn but content not used (move arrows shown but no evaluation display)

**TODO/FIXME markers:** None found in current code

**Dead code paths:**
- None identified; code is clean and follows through

**Piece type recognition:**
- **BLOCKING ISSUE:** System only detects piece COLOR (white/black), NOT TYPE (pawn/rook/queen/etc.)
- For v1.0, this is marked as required but not implemented
- Vision system only checks `avg_brightness < 100` to detect presence, not shape or position

---

## 10. v1 Readiness Checklist

### What Currently Works End-to-End
✅ **Happy path game flow:**
1. User selects side (W/B/A)
2. Board located via color or corner detection
3. Player makes move → opponent move auto-detected (pixel diff > 1000 within 16s)
4. Opponent move entered manually
5. Engine generates best move with Turing Filter (30% chance of 2nd-best if scores close)
6. Humanizer executes move with realistic Bezier mouse path + jitter + timing
7. Board state updated via `chess.Board()`
8. Loop continues until game over

✅ **Fallback mechanisms:**
- Yellow highlight detection works when manual button pressed (Y key)
- Manual move input always available as final fallback
- Side auto-detection works on most boards (corner brightness method)

✅ **Special move handling:**
- Promotion popup clicking (hardcoded offset, assumes queen first)
- Castling (handled by `chess` library automatically)
- En passant (handled by `chess` library automatically)

### What is Missing for Reliable v1 Ship
❌ **Piece type recognition** - CRITICAL BLOCKER
- Only color detected, not type
- Affects: Cannot validate move makes sense positionally
- Affects: Cannot distinguish pawn from other pieces on highlights
- Impact: Yellow detection may return impossible moves if piece type ambiguous

❌ **Side auto-detection reliability** - HIGH PRIORITY
- Current method: check corner brightness > 120
- Fails on many modern board themes (gradients, alternating colors, etc.)
- User must select W/B manually instead of using A option

❌ **Rapid move detection stability** - MEDIUM PRIORITY
- Pixel threshold (1000 non-zero) may be too high, missing subtle animations
- May miss fast opponent moves if threshold not tuned for board theme
- Recent commits show iterative tuning of piece detection thresholds

❌ **Promotion popup position** - MEDIUM PRIORITY
- Hardcoded offset assumes specific popup layout
- May click wrong piece or miss popup entirely on different sites

❌ **File structure cleanup** - LOW PRIORITY
- `core/vision2.py` should be deleted
- Orphaned opening recognition module not used

### 5 Most Likely Runtime Failures
1. **Board not detected** - Color/corner detection fails on unusual board themes → exit with error message
2. **Stockfish missing** - `assets/stockfish.exe` not downloaded → immediate crash on init
3. **Move detection timeout** - Pixel change < 1000 for > 16s → user must press Y or enter move (not failure, but requires intervention)
4. **Invalid yellow detection** - Piece threshold (100 brightness) wrong for theme → returns impossible move coordinates → ValueError on push_uci
5. **Promotion popup miss** - Hardcoded offset doesn't match site layout → clicks wrong piece, game state corrupted

---

## 11. Minimal Repro Instructions

### Prerequisites
- Windows 10/11 (uses pyautogui, mss, Windows API)
- Python 3.7+
- Stockfish: Download from https://stockfishchess.org/download/, place `stockfish.exe` in `C:\Users\jeff\ghost-shell\assets\stockfish.exe`

### Installation
```bash
cd C:\Users\jeff\ghost-shell
pip install -r requirements.txt
```

### Run
```bash
python ghost-shell\main.py
```

### Expected Output on Success
```
[HH:MM:SS] [MAIN] Initializing Ghost-Shell...

==================================================
Which side are you playing?
  [W] White (you move first)
  [B] Black (opponent moves first)
  [A] Auto-detect (may not be accurate)
==================================================
Enter W/B/A: W

[HH:MM:SS] [MAIN] Make sure the board is visible.
[HH:MM:SS] [MAIN] Press 'S' to start.
[HH:MM:SS] [VISION] Looking for board...
[HH:MM:SS] [VISION] Found board at (X, Y) size WxH
[HH:MM:SS] [MAIN] Board locked. Playing as White.
[HH:MM:SS] [ENGINE] Brain loaded. Depth: 15, Contempt: 20
[HH:MM:SS] [MAIN] My turn. Thinking for 2.3s...
[HH:MM:SS] [ENGINE] Best move: e2e4
[HH:MM:SS] [HUMANIZER] Making move...
[HH:MM:SS] [HUMANIZER] Done.
[HH:MM:SS] [MAIN] Played: e2e4
[HH:MM:SS] [MAIN] Opponent's turn.
[HH:MM:SS] [MAIN] Waiting for opponent...
```

### Trigger Yellow Fallback During Run
1. Game running, waiting for opponent move (after "Waiting for opponent..." message)
2. Opponent makes move on screen but auto-detection doesn't trigger (15+ seconds pass)
3. **Press Y key** on keyboard
4. System logs: "Yellow highlight detection triggered!"
5. Scans board for yellow squares, extracts move, attempts to push to board state

### Expected Yellow Detection Output
```
[HH:MM:SS] [VISION] Found yellow highlight at e2 (piece: False)
[HH:MM:SS] [VISION] Found yellow highlight at e4 (piece: True)
[HH:MM:SS] [VISION] Yellow move detected: e2 -> e4 (empty -> piece)
[HH:MM:SS] [MAIN] Move detected from yellow: e2e4
```

### Config (.env optional)
```env
ENGINE_DEPTH=15
ENGINE_CONTEMPT=20
MOUSE_SPEED=0.4
TARGET_JITTER=6
THINK_TIME_MIN=1.5
THINK_TIME_MAX=6.0
PLAYER_SIDE=AUTO
```

### Debug
Press **D** during game to save `debug_vision.jpg` (shows detected board grid) or **Z** to undo last move.
