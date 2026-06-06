"""
Proof of concept for CDP move injection.

Opens chess.com's free analysis board (no login), plays e2e4 then e7e5 by
clicking through Playwright, and checks:
  1. the pieces actually moved (DOM shows a piece on the destination square), and
  2. the real hardware cursor never moved (pyautogui position unchanged).

Run:  venv/Scripts/python.exe poc_browser.py
"""
import time
import pyautogui
from core.browser import GhostBrowser


def piece_on(page, square):
    """chess.com tags pieces with class 'square-FR' (file 1-8, rank 1-8). e4 -> square-54."""
    f = ord(square[0]) - ord("a") + 1
    r = int(square[1])
    return page.eval_on_selector_all(
        f".piece.square-{f}{r}", "els => els.length"
    ) > 0


def main():
    browser = GhostBrowser(start_url="https://www.chess.com/analysis")
    results = []
    try:
        browser.start()
        if not browser.wait_for_board(timeout_s=60):
            print("FAIL: board never appeared")
            return 1

        page = browser._require_page()
        box = browser.get_board_box()
        print(f"Board box (viewport px): {box}")

        cursor_before = pyautogui.position()
        print(f"Cursor before moves: {cursor_before}")

        # Play two moves on the analysis board (white perspective for both).
        browser.make_move("e2e4", is_white=True)
        time.sleep(0.6)
        browser.make_move("e7e5", is_white=True)
        time.sleep(0.6)

        cursor_after = pyautogui.position()
        print(f"Cursor after moves:  {cursor_after}")

        moved_e4 = piece_on(page, "e4")
        moved_e5 = piece_on(page, "e5")
        cursor_still = (cursor_before == cursor_after)

        print(f"\nPiece on e4? {moved_e4}   Piece on e5? {moved_e5}   Cursor unmoved? {cursor_still}")
        results = [
            ("e2e4 landed (piece on e4)", moved_e4),
            ("e7e5 landed (piece on e5)", moved_e5),
            ("hardware cursor did NOT move", cursor_still),
        ]

        if not (moved_e4 and moved_e5):
            page.screenshot(path="poc_failure.png")
            print("Saved poc_failure.png for inspection.")

        time.sleep(2)  # let you see the board before it closes
    finally:
        browser.close()

    print("\n=== RESULTS ===")
    ok = True
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
