# Piece Color Classification Threshold Fixes

## Problem Statement
The bot was misidentifying opponent moves because the piece color classification was returning '?' (uncertain) instead of 'w' (white) or 'b' (black) for most pieces. This prevented:
- ❌ Opponent move detection
- ❌ Move history display
- ❌ Evaluation bar display
- ❌ Opening recognition
- ❌ Legal moves display

## Root Cause
The `_classify_piece_color()` function in `core/vision.py` had thresholds that were too strict:
- Edge-vs-center delta required > 8 (too high)
- Calibrated background comparison used only 4% tolerance
- Last-resort absolute brightness required > 170 or < 90 (unrealistic)
- Highlighted square thresholds too rigid

## Solution Implemented
Relaxed all thresholds to be more lenient while maintaining accuracy:

### Threshold Changes:
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| COLOR_RATIO_THRESHOLD | 0.04 (4%) | 0.08 (8%) | More forgiving on borderline pieces |
| Edge-vs-center delta | > 8 | > 6 | Catches weaker contrast signals |
| Highlighted white threshold | > 160 | > 155 | More lenient on yellow backgrounds |
| Highlighted black threshold | < 100 | < 105 | More lenient on yellow backgrounds |
| Highlighted edge delta | > 7 | > 5 | Better detection on highlights |
| Last-resort white | > 170 | > 160 | Catches darker white pieces |
| Last-resort black | < 90 | < 100 | Catches brighter black pieces |

## How It Works
The classification uses a 4-strategy fallback approach:

1. **Edge-vs-center comparison** (PRIMARY - most robust)
   - Compares piece body to its surrounding square border
   - Works regardless of board background color
   - **NEW: Requires delta > 6 instead of > 8**

2. **Highlighted square handling** (for yellow-highlighted squares)
   - Uses absolute brightness + edge delta
   - **NEW: Relaxed thresholds for better detection**

3. **Calibrated background comparison** (uses learned square colors)
   - Compares to expected light/dark square brightness
   - **NEW: Uses 8% tolerance instead of 4%**

4. **Last-resort absolute brightness** (fallback)
   - Simple threshold on center pixel brightness
   - **NEW: Lowered to 160/100 for more catches**

## Testing Instructions

### Quick Test (Automatic Detection)
1. Run `runbot.bat`
2. Open a Lichess game with visible board
3. Press 'S' when prompted
4. Watch console output - should show piece detection like:
   ```
   Piece map: 32 pieces (w=16, b=16, ?=0)
   ```
   **Success:** All pieces are 'w' or 'b', zero '?' marks

### Move Detection Test
1. Start a game against the bot
2. Make a move on the board
3. Bot should auto-detect your move without '?' marks
4. Console should show:
   ```
   Detected move: e2e4
   ```

### Yellow Highlight Test
1. During opponent's turn, press 'Y' key
2. Bot should detect the move from yellow highlights instantly
3. No piece color classification needed (highlights are sufficient)

## What Changed in Code
**File:** `core/vision.py`
**Commit:** e2de5e7 (Relax piece color classification thresholds...)

The changes affect only the `_classify_piece_color()` method and the global threshold constants. The piece detection logic and move inference remain unchanged.

## Expected Results
After applying these fixes:
- ✅ Most pieces will be classified as 'w' or 'b' (rarely '?')
- ✅ Opponent moves will be detected automatically
- ✅ Move history will display properly
- ✅ Evaluation scores will show
- ✅ Opening detection will work
- ✅ The 'Y' highlight key will be a reliable fallback

## Rollback Instructions
If you need to revert these changes:
```bash
git revert e2de5e7
# or
git checkout HEAD~1 -- core/vision.py
```

## Notes
- These thresholds are calibrated for Lichess/chess.com
- If you're using a different chess platform, thresholds may need slight adjustment
- The 'Y' key (highlight detection) is now a more reliable fallback
- Consider adjusting screen zoom/lighting if still seeing many '?' marks
