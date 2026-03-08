# Ghost-Shell Project Snapshot - v1.0 Development

## 1. ARCHITECTURE OVERVIEW

**Entry Point:**
- `ghost-shell/main.py` - GhostShell class, run() method starts game loop

**Core Modules:**
- `core/vision.py` - Screen capture, board/piece detection, highlight parsing
- `core/engine.py` - Stockfish wrapper, best move selection with Turing Filter
- `core/humanizer.py` - Realistic mouse movement (Bezier curves, jitter, timing)
- `core/openings.py` - Hardcoded opening database (6 moves)
- `ui/overlay.py` - Transparent HUD with move arrows (Windows API)
- `utils/config.py` - ENV file loading and defaults
- `utils/logger.py` - Colored console output

**Vision Pipeline:**
- Capture screen via mss
- Board detection: color-based grid pattern OR corner detection
- Piece detection: brightness thresholding (white/black color)
- Yellow highlight detection: HSV color range masking
- Piece presence detection: average brightness < 100 threshold

**Move Detection Flow:**
1. wait_for_opponent_move() watches for pixel changes (16 fps sampling)
2. Compares frames using cv2.absdiff()
3. Pixel difference threshold: 1000+ non-zero pixels = move detected
4. After detection, waits 1.0 second for animation to settle
5. If detection fails after 20 attempts, waits for 'Y' button press
6. Returns: True (movement detected), "yellow" (manual trigger), or None (quit)

**Manual Fallback Logic:**
- 'Y' button can be pressed ANY TIME during wait_for_opponent_move()
- Calls detect_yellow_highlights() to scan board for yellow squares
- Identifies FROM (empty) and TO (has piece) squares
- Validates move against chess.Board legal moves
- Falls back to manual input if detection fails

**Dependencies:**
- opencv-python, python-chess, keyboard, stockfish, pyautogui, mss, numpy, pywin32 (optional), python-dotenv (optional)

---

## 2. CURRENT OPPONENT MOVE DETECTION FLOW

**Step-by-Step Logic:**

1. Opponent's turn starts in main game loop
2. wait_for_opponent_move() called
3. Sleep 1.0s to let own animation finish
4. Capture initial screen state as baseline
5. Loop for up to 20 attempts (16s total, 0.8s sleep per attempt):
   - Check if 'q' pressed (quit)
   - Check if 'y' pressed (manual yellow detection)
   - Sleep 0.8s
   - Capture current frame
   - Compute absolute difference: cv2.absdiff(prev_frame, curr_frame)
   - Convert to grayscale
   - Binary threshold at value=30
   - Count non-zero pixels
   - If count > 1000: movement detected, sleep 1.0s, return True
   - Otherwise: update baseline, increment attempt counter

**Failure Condition:**
- After 20 attempts with no detection (no pixel change > 1000), exit loop
- Enter fallback wait loop

**After 20 Attempts:**
- Log: "Move detection failed after 20 attempts"
- Log: "Press 'Y' to manually detect from yellow highlights"
- Enter infinite loop checking only for 'q' or 'y' press
- No timeout, waits indefinitely

**How Yellow Detection is Triggered:**
- User presses 'Y' key
- Function returns string "yellow"
- Main loop catches detected == "yellow" condition
- Calls vision.detect_yellow_highlights()

**Return Values:**
- True: Movement detected, user must manually input move
- "yellow": Yellow highlights found, attempting auto-detection
- None: Quit detected via 'q' key

**How Main Loop Consumes Result:**
```
if detected is None:
    break (exit game)
elif detected == "yellow":
    move_result = vision.detect_yellow_highlights()
    if move_result:
        from_sq, to_sq = move_result
        move_uci = from_sq + to_sq
        board.push_uci(move_uci)
    else:
        prompt manual input
elif detected:
    prompt manual input
```

---

## 3. YELLOW HIGHLIGHT DETECTION IMPLEMENTATION

**Method Name:**
- GhostVision.detect_yellow_highlights() at vision.py:211-310

**HSV Thresholds:**
- Lower bound: [15, 100, 100] (Hue, Saturation, Value)
- Upper bound: [45, 255, 255]
- Targets yellow/golden hue range

**Square Scanning Logic:**
1. Convert frame BGR → HSV
2. Create mask using cv2.inRange() with yellow bounds
3. Loop all 64 squares (rank_idx 0-7, file_idx 0-7)
4. For each square: count non-zero pixels in yellow_mask
5. If yellow_count > (sq_size * sq_size * 0.3): square is highlighted (30% threshold)
6. Convert (rank_idx, file_idx) → chess notation (a1-h8)

**From/To Decision Logic:**
1. For each highlighted square, extract region and check brightness
   - Convert to grayscale
   - Calculate average brightness
   - has_piece = avg_brightness < 100
2. Categorize into: empty_squares and piece_squares lists
3. If exactly 1 empty + 1 piece:
   - FROM = empty square
   - TO = piece square
   - Log: "(empty -> piece)"
4. If 2 pieces + 0 empty (capture):
   - FROM = yellow_squares[0]
   - TO = yellow_squares[1]
   - Log: "(capture)"
5. Else (ambiguous):
   - FROM = yellow_squares[0]
   - TO = yellow_squares[1]
   - Log: "Ambiguous highlights"

**Capture Handling Assumptions:**
- Captured piece is simply gone from board
- Both source and destination squares appear to have pieces (source has departing piece, dest has arriving piece)
- Detected as: both piece_squares, so falls to capture case

**Known Limitations:**
- Brightness threshold (100) may need tuning per board theme
- 30% yellow coverage threshold may miss faint highlights
- Assumes only 2 squares highlighted (not valid for castling with 4-square highlights)
- Does not distinguish piece type, only presence

---

## 4. KNOWN BUGS / INSTABILITIES

**Flaky Areas:**
- Piece color detection still under active threshold tuning (13+ commits in past month)
- Yellow highlights shift piece color readings on highlighted squares
- Side auto-detection ("A" option) unreliable on varied board themes

**Edge Cases Not Handled:**
- Castling: 4 squares may be highlighted (king+rook), detection will use first 2
- Promotion: popup position varies by site, hardcoded offset may fail
- Multiple highlight colors: assumes yellow only, ignores green/blue highlights
- Very fast opponents: pixel change detection may trigger before animation settles

**Timing Issues:**
- 1.0s wait after detection may be insufficient for some board animations
- 0.8s sample interval may miss very rapid moves
- Debounce 0.5s for 'Y' button may cause missed input

**Board Orientation Issues:**
- Assumes 8x8 grid in standard orientation
- Board flip logic in get_square_center() may have off-by-one errors
- Rank indexing inverted (rank_idx 0 = rank 8 for white)

---

## 5. PARTIALLY IMPLEMENTED OR INCOMPLETE FEATURES

**Incomplete:**
- Piece type recognition: Only color detected (white/black), not type (pawn/rook/etc) - BLOCKING for v1.0
- Multi-monitor support: Only scans primary monitor
- Promotion popup detection: Hardcoded click position, fails on site variations
- Side auto-detection: Corner brightness method unreliable - requires fixing
- Adaptive thresholds: All thresholds hardcoded, no learning/calibration

**Experimental Files:**
- `core/vision2.py`: Duplicate file with mixed/incomplete code at end (lines 244+)
  - Contains get_board_piece_map(), detect_move_from_maps(), detect_opponent_move_uci()
  - Appears to be abandoned experimental version
  - Should be deleted or consolidated before v1.0

**Dead Code:**
- None identified (code is clean)

**TODO Markers:**
- None found (no explicit TODO/FIXME comments)

---

## 6. ENVIRONMENTAL ASSUMPTIONS

**Resolution:**
- Primary monitor detected via mss.mss().monitors[1]
- No specific resolution requirement, auto-scales to board size
- Board detection scales square_size dynamically

**Board Size:**
- Always 8x8 grid
- Square size calculated: board_width / 8

**Platform Assumptions:**
- Windows only (uses pywin32 for overlay)
- Chess.com and Lichess board themes (primary tested platforms)
- Standard board rendering (not variants like Fischer Random)

**Side Detection Assumptions:**
- White: bottom-left corner brightness > 120 OR bottom-right > 120
- Black: both corners ≤ 120
- Assumption: light-squared corners differ based on orientation
- Known issue: unreliable on varied board themes

**Promotion Handling:**
- Hardcoded popup position offset in humanizer.py
- Clicks queen by default
- Offset likely fails on non-standard popup placements

---

## 7. WHAT CURRENTLY WORKS END-TO-END

**Successful Opponent Move Cycle:**

1. Game starts, user selects side (W/B/A)
2. Board detected via color grid or corner detection
3. Player makes move, move executed via humanizer
4. wait_for_opponent_move() called
5. Opponent makes move on screen
6. Pixel change detected within 20 attempts (one of attempts captures change > 1000 pixels)
7. Movement detected, sleep 1.0s for animation
8. wait_for_opponent_move() returns True
9. Main loop displays legal moves: "Opponent's turn. Movement detected! Enter the move."
10. User manually enters move (e.g., "e7e5")
11. board.push_uci("e7e5") succeeds
12. Board state updated in memory
13. Next turn begins, game continues
14. On next player turn: engine calculates best move, humanizer executes with realistic timing
15. Cycle repeats

**Fully Working:**
- Board finding and calibration
- Move execution with human-like timing
- Engine integration (Stockfish)
- Game loop state management
- Opening recognition (6 moves)
- Keyboard controls (S, Q, D, Y, Z)
- Colored logging output
- Configuration via .env file
- Yellow highlight detection (manual trigger via 'Y' button, smart from/to detection logic)

---

## v1.0 CRITICAL PATH

**PHASE 1: Piece Type Detection (3-4 hours) - PRIMARY BLOCKER**
- Implement hybrid piece type detection in vision.py
- Add size + position analysis
- Add shape detection as fallback
- Validate against board state using python-chess
- Test on chess.com + Lichess with various board themes

**PHASE 2: Side Detection Improvement (1-2 hours)**
- Enhance detect_player_side() using piece position analysis
- Improve fallback to corner brightness
- Add board preview in setup flow
- Test on both platforms

**PHASE 3: Yellow Highlight Validation (1-2 hours)**
- Run real games with yellow highlights enabled
- Test chess.com and Lichess
- Verify edge cases (captures, castling, promotion)
- Adjust HSV range if needed

**PHASE 4: Bug Fixes & Stabilization (1-2 hours)**
- Rapid move detection stability improvements
- Promotion popup coordinate fixes
- File structure cleanup (remove vision2.py)
- Rapid games (10+ min) focused testing

**PHASE 5: Documentation & Release (1-2 hours)**
- Complete README.md with supported platforms
- Create RELEASE_NOTES_v1.0.md
- Update .env.example with all config options
- Create CONFIG_GUIDE.md
- Tag v1.0.0 in git

**Total Estimated Time: 8-12 hours**
