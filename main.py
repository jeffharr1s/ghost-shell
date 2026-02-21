import time
import random
import cv2
import chess
import keyboard
import numpy as np
import ctypes
import sys
from core.vision import GhostVision
from core.engine import GhostEngine
from core.humanizer import Humanizer
from core.openings import detect_opening
from ui.overlay import GhostOverlay
from utils.logger import Logger
from utils.config import PLAYER_SIDE, THINK_TIME_MIN, THINK_TIME_MAX

class GhostShell:
    def __init__(self):
        self.logger = Logger("MAIN")
        self.vision = GhostVision()
        self.engine = GhostEngine()
        self.humanizer = Humanizer()
        self.overlay = GhostOverlay()
        self.board = chess.Board()
        self.user_side = chess.WHITE

    def get_square_center(self, square_name):
        """converts 'e4' to screen coords"""
        file_idx = chess.FILE_NAMES.index(square_name[0])
        rank_idx = int(square_name[1]) - 1

        bx, by, bw, bh = self.vision.board_location
        sq_size = self.vision.square_size

        if self.user_side == chess.WHITE:
            x = bx + (file_idx * sq_size) + (sq_size / 2)
            y = by + ((7 - rank_idx) * sq_size) + (sq_size / 2)
        else:
            # board is flipped for black
            x = bx + ((7 - file_idx) * sq_size) + (sq_size / 2)
            y = by + (rank_idx * sq_size) + (sq_size / 2)

        return int(x), int(y)

    def display_move_history(self):
        """Display all moves played so far in algebraic notation"""
        if len(self.board.move_stack) == 0:
            return

        # Format moves in pairs (White, Black) like: 1. e2e4 e7e5 2. g1f3...
        move_str = ""
        for i, move in enumerate(self.board.move_stack):
            if i % 2 == 0:  # White's move (or first player)
                move_num = (i // 2) + 1
                move_str += f"{move_num}. {move.uci()} "
            else:  # Black's move
                move_str += f"{move.uci()} "

        self.logger.log(f"Game: {move_str.strip()}")

        # Try to detect opening
        opening = detect_opening(self.board)
        if opening:
            self.logger.log(f"Opening: {opening}")

    def wait_for_opponent_move(self):
        """watches screen for pixel changes - waits for stable state"""
        self.logger.log("Waiting for opponent...")

        # wait a bit for our own animation to finish
        time.sleep(1.0)
        previous_state_img = self.vision.capture_screen()

        while True:
            if keyboard.is_pressed('q'):
                self.logger.warning("Quit detected.")
                return None

            time.sleep(0.8)
            current_state_img = self.vision.capture_screen()

            diff = cv2.absdiff(previous_state_img, current_state_img)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            non_zero = cv2.countNonZero(thresh)

            if non_zero > 1000:  # higher threshold
                self.logger.success("Movement detected!")
                time.sleep(1.0)  # wait for animation to fully finish
                return True

            # update baseline
            previous_state_img = current_state_img

    def retry_move(self, start_sq, end_sq):
        """retry making a move if it failed the first time"""
        self.logger.log(f"Retrying move {start_sq} -> {end_sq}")
        start_coords = self.get_square_center(start_sq)
        end_coords = self.get_square_center(end_sq)
        sq_size = int(self.vision.square_size)
        is_white = self.user_side == chess.WHITE

        self.overlay.draw_move_arrow(start_coords, end_coords)
        time.sleep(0.3)

        self.humanizer.make_move(
            start_coords, end_coords, None, sq_size, is_white
        )
        self.overlay.clear()
        self.logger.success("Move retry complete.")

    def run(self):
        try:
            self.logger.log("Initializing Ghost-Shell...")

            # ask user which side they're playing
            print("\n" + "=" * 50)
            print("Which side are you playing?")
            print("  [W] White (you move first)")
            print("  [B] Black (opponent moves first)")
            print("  [A] Auto-detect (may not be accurate)")
            print("=" * 50)

            # Bring console window to foreground on Windows
            if sys.platform == 'win32':
                try:
                    kernel32 = ctypes.windll.kernel32
                    user32 = ctypes.windll.user32
                    hwnd = kernel32.GetConsoleWindow()
                    if hwnd:
                        user32.SetForegroundWindow(hwnd)
                        user32.SetFocus(hwnd)
                except Exception as e:
                    self.logger.warning(f"Could not set window focus: {e}")

            # Make a beep sound to alert user
            print('\a')

            self.logger.debug("Waiting for user to select side...")
            side_input = input("Enter W/B/A: ").strip().upper()
            self.logger.debug(f"User selected: {side_input}")

            if side_input == "W":
                self.user_side = chess.WHITE
                self.logger.debug("White selected")
            elif side_input == "B":
                self.user_side = chess.BLACK
                self.logger.debug("Black selected")
            else:
                self.logger.debug("Auto-detect mode selected")

            self.logger.warning("Make sure the board is visible.")
            self.logger.warning("Press 'S' to start.")
            self.logger.debug("Waiting for user to press 'S'...")
            keyboard.wait('s')
            self.logger.debug("'S' pressed, starting board detection...")

            self.logger.debug("Calling vision.find_board()...")
            location = self.vision.find_board()
            if not location:
                self.logger.error("Couldnt find board. Exiting.")
                return

            self.logger.debug(f"Board found at: {location}")

            # snap overlay to board
            self.logger.debug("Updating overlay geometry...")
            self.overlay.update_geometry(*location)

            # auto-detect if user chose A
            if side_input == "A" or PLAYER_SIDE == "AUTO":
                self.logger.debug("Auto-detecting player side...")
                detected_side = self.vision.detect_player_side()
                if detected_side is not None:
                    self.user_side = detected_side
                    self.logger.debug(f"Auto-detected side: {detected_side}")

            self.logger.success(
                f"Board locked. Playing as {'White' if self.user_side == chess.WHITE else 'Black'}."
            )

            # Tell the vision system which way the board is oriented
            self.vision.set_orientation(self.user_side)

            # if playing black, wait for opponent's first move
            if self.user_side == chess.BLACK:
                self.logger.log("Playing as Black - waiting for White's first move...")
                print("\nEnter opponent's first move when ready.")
                while True:
                    move = input("Opponent's move (e.g. e2e4): ").strip().lower()
                    if move:
                        try:
                            self.logger.debug(f"Attempting to push move: {move}")
                            self.board.push_uci(move)
                            self.logger.debug("Getting initial board piece map...")
                            self.prev_map = self.vision.get_board_piece_map()
                            self.logger.debug(f"Initial piece map: {len(self.prev_map)} pieces")
                            break
                        except ValueError as e:
                            self.logger.error(f"Invalid move '{move}': {e}")

            # Initialize prev_map for board state tracking
            if not hasattr(self, 'prev_map'):
                self.logger.debug("Initializing prev_map for the first time...")
                self.prev_map = self.vision.get_board_piece_map()
                self.logger.debug(f"Initialized with {len(self.prev_map)} pieces")

            self.logger.debug("Checking if game is over...")
            self.logger.log("Press 'D' at any time to save a debug snapshot.")

            while not self.board.is_game_over():
                try:
                    # Debug snapshot hotkey
                    if keyboard.is_pressed('d'):
                        self.vision.debug_dump_board()
                        time.sleep(0.5)  # debounce

                    if self.board.turn == self.user_side:
                        self.logger.log("=== MY TURN ===")
                        think_time = random.uniform(THINK_TIME_MIN, THINK_TIME_MAX)
                        self.logger.log(f"Thinking for {think_time:.1f}s...")
                        time.sleep(think_time)

                        self.logger.debug("Getting FEN from board...")
                        fen = self.board.fen()
                        self.logger.debug(f"FEN: {fen}")

                        self.logger.debug("Requesting best move from engine...")
                        best_move_uci, eval_cp = self.engine.get_human_move(fen)
                        self.logger.debug(f"Engine returned: {best_move_uci} (eval: {eval_cp}cp)")

                        # Display evaluation
                        if eval_cp is not None:
                            eval_score = eval_cp / 100.0
                            if abs(eval_score) < 0.1:
                                self.logger.log(f"[Eval: {eval_score:+.1f}] Position is roughly equal")
                            elif eval_score > 0:
                                self.logger.log(f"[Eval: {eval_score:+.1f}] {'White' if self.user_side == chess.WHITE else 'Black'} is winning")
                            else:
                                self.logger.log(f"[Eval: {eval_score:+.1f}] {'Black' if self.user_side == chess.WHITE else 'White'} is winning")

                        # Display best legal moves
                        legal_moves = list(self.board.legal_moves)
                        self.logger.log(f"Legal moves ({len(legal_moves)} available): {' '.join([m.uci() for m in legal_moves[:15]])}" + ("..." if len(legal_moves) > 15 else ""))
                        self.logger.log(f"Playing: {best_move_uci}")

                        if best_move_uci:
                            start_sq = best_move_uci[:2]
                            end_sq = best_move_uci[2:4]
                            self.logger.debug(f"Move: {start_sq} -> {end_sq}")

                            self.logger.debug("Calculating screen coordinates...")
                            start_coords = self.get_square_center(start_sq)
                            end_coords = self.get_square_center(end_sq)
                            self.logger.debug(f"Screen coords: {start_coords} -> {end_coords}")

                            promotion_piece = None
                            if len(best_move_uci) > 4:
                                promotion_piece = best_move_uci[4]
                                self.logger.debug(f"Promotion piece: {promotion_piece}")

                            sq_size = int(self.vision.square_size)
                            is_white = self.user_side == chess.WHITE

                            # show the move on HUD
                            self.logger.debug("Drawing move arrow on overlay...")
                            self.overlay.draw_move_arrow(start_coords, end_coords)
                            time.sleep(0.3)

                            self.logger.debug("Making move with humanizer...")
                            self.humanizer.make_move(
                                start_coords, end_coords, promotion_piece, sq_size, is_white
                            )
                            self.logger.debug("Move made, clearing overlay...")
                            self.overlay.clear()

                            self.logger.debug(f"Pushing move to board: {best_move_uci}")
                            self.board.push_uci(best_move_uci)
                            self.logger.debug("Getting new board piece map...")
                            self.prev_map = self.vision.get_board_piece_map()
                            self.logger.debug(f"Board now has {len(self.prev_map)} pieces")

                            # log the move made
                            self.logger.success(f"Played: {best_move_uci}")

                            # Offer retry option if move didn't seem to register
                            self.logger.log("Press SPACE if piece didn't move, any other key to continue...")
                            if keyboard.is_pressed('space'):
                                self.logger.warning("SPACE pressed - retrying move")
                                self.retry_move(start_sq, end_sq)
                                self.logger.log("Retry complete. Continuing...")
                        else:
                            self.logger.error("Engine returned no move!")

                    else:
                        self.logger.log("=== OPPONENT'S TURN ===")
                        print(f"\n{self.board}")
                        print(
                            f"\nLegal moves: {', '.join([m.uci() for m in list(self.board.legal_moves)[:10]])}..."
                        )

                        self.logger.debug("Waiting for opponent to move...")
                        detected = self.wait_for_opponent_move()

                        if detected is None:
                            self.logger.warning("Quit detected during opponent's move.")
                            break

                        if detected:
                            # Try auto-detecting the move
                            uci_move = None

                            self.logger.debug("Attempting auto-detection of opponent's move...")
                            self.logger.debug(f"Previous board state had {len(self.prev_map)} pieces")
                            # Keep scanning a few times until a move is detected
                            for attempt in range(20):
                                self.logger.debug(f"Auto-detect attempt {attempt + 1}/20...")
                                uci_move = self.vision.detect_opponent_move_uci(self.prev_map)
                                if uci_move:
                                    self.logger.success(f"Auto-detected move: {uci_move}")
                                    break
                                time.sleep(0.25)

                            if not uci_move:
                                self.logger.error("=" * 60)
                                self.logger.error("AUTO-DETECTION FAILED AFTER 20 ATTEMPTS")
                                self.logger.error("Common causes:")
                                self.logger.error("  - Yellow highlight on last move interfering with detection")
                                self.logger.error("  - Board alignment or visibility issues")
                                self.logger.error("  - Color/shading of pieces not distinct enough")
                                self.logger.error("  - Try adjusting lighting or board zoom level")
                                self.logger.error("=" * 60)

                            time.sleep(1.5)  # Wait longer for move to complete

                            # Fallback to manual input if auto-detection fails
                            if not uci_move:
                                self.logger.log("Movement detected! Enter the move.")

                            while True:
                                move = uci_move or input("Opponent's move (e.g. e7e5, or 'z' to undo): ").strip().lower()
                                self.logger.debug(f"User entered move: {move}")

                                # Check for undo command
                                if move == 'z':
                                    if len(self.board.move_stack) > 0:
                                        self.logger.debug("Undo requested...")
                                        undone_move = self.board.pop()
                                        self.logger.warning(f"Undone opponent move: {undone_move.uci()}")
                                        self.logger.debug("Recapturing board state...")
                                        self.prev_map = self.vision.get_board_piece_map()
                                        # Ask for the correct move
                                        continue
                                    else:
                                        self.logger.error("No move to undo.")
                                        continue

                                try:
                                    self.logger.debug(f"Pushing move to board: {move}")
                                    self.board.push_uci(move)
                                    self.logger.debug("Getting updated board piece map...")
                                    self.prev_map = self.vision.get_board_piece_map()
                                    self.logger.debug(f"Board now has {len(self.prev_map)} pieces")
                                    self.logger.success(f"Opponent played: {move}")
                                    self.display_move_history()
                                    break
                                except ValueError as e:
                                    self.logger.error(f"Invalid move '{move}': {e}")
                                    print(
                                        f"Legal moves: {', '.join([m.uci() for m in list(self.board.legal_moves)[:10]])}..."
                                    )
                                    uci_move = None

                except Exception as e:
                    self.logger.error(f"Exception in game loop: {type(e).__name__}: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    self.logger.warning("Attempting to continue...")

            self.logger.success("Game Over.")

        except Exception as e:
            self.logger.error(f"Critical error in run(): {type(e).__name__}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

if __name__ == "__main__":
    bot = GhostShell()
    bot.run()
