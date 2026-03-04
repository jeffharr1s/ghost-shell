import time
import random
import sys
import itertools
import cv2
import chess
import keyboard
import numpy as np
from core.vision import GhostVision
from core.engine import GhostEngine
from core.humanizer import Humanizer
from ui.overlay import GhostOverlay
from utils.logger import Logger
from utils.config import PLAYER_SIDE, THINK_TIME_MIN, THINK_TIME_MAX, APP_VERSION

class GhostShell:
    def __init__(self):
        self.logger = Logger("MAIN")
        self.vision = GhostVision()
        self.engine = GhostEngine()
        self.humanizer = Humanizer()
        self.overlay = GhostOverlay()
        self.board = chess.Board()
        self.user_side = chess.WHITE
        self.last_move = None  # stores last bot move for redo: (uci, start_coords, end_coords, promo, sq_size, is_white)

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

    def _try_yellow_once(self):
        """tries yellow detection once, attempting all permutations of detected squares.
        filters out bot's own last-move squares to remove stale highlights.
        returns True if a valid move was pushed, False otherwise."""
        yellow_squares = self.vision.detect_yellow_highlights()
        if not yellow_squares:
            self.logger.warning("No yellow squares detected")
            return False

        self.logger.log(f"Yellow squares: {yellow_squares}")

        # filter out bot's own last-move squares - chess.com keeps them highlighted,
        # so after we play e2e4 and wait for the opponent, e2 and e4 stay yellow
        # until the opponent moves, causing 3+ squares and wrong move detection
        candidates = yellow_squares
        if self.last_move:
            bot_from = self.last_move[0][:2]
            bot_to   = self.last_move[0][2:4]
            filtered = [sq for sq in yellow_squares if sq not in (bot_from, bot_to)]
            if len(filtered) >= 2:
                candidates = filtered
                self.logger.log(f"Filtered bot squares ({bot_from},{bot_to}), using: {candidates}")

        # try every ordered pair - handles reversed direction AND 3-square stale highlight cases
        for from_sq, to_sq in itertools.permutations(candidates, 2):
            move_uci = from_sq + to_sq
            if self.last_move and move_uci == self.last_move[0]:
                continue  # skip our own move
            try:
                self.board.push_uci(move_uci)
                self.logger.success(f"Auto-detected: {move_uci}")
                return True
            except ValueError:
                pass

        self.logger.error(f"No valid move among yellow squares {candidates}")
        return False

    def _try_yellow_then_manual(self):
        """auto-scans yellow highlights with retries; falls back to manual input if all attempts fail"""
        max_attempts = 5
        retry_delay = 1.2  # seconds between scans

        for attempt in range(1, max_attempts + 1):
            wait = 1.0 if attempt == 1 else retry_delay
            self.logger.log(f"Yellow scan attempt {attempt}/{max_attempts} (waiting {wait}s)...")
            time.sleep(wait)
            if self._try_yellow_once():
                return
            self.logger.error(f"Attempt {attempt}: no valid move found, retrying...")

        self.logger.error(f"Yellow failed after {max_attempts} attempts - enter move manually")

        # fallback: manual input (only reached if yellow failed)
        while True:
            move = input("Opponent's move (e.g. e7e5), 'y'=yellow, 'r'=redo, 'd'=diag, 's'=I moved, 'q'=quit: ").strip().lower()
            if move == 'q':
                self.logger.warning("Quit.")
                sys.exit(0)
            elif move == 's':
                # user manually made the bot's failed move on screen - board state is already correct
                self.logger.success("Manual move confirmed - waiting for opponent...")
                return
            elif move == 'r':
                self.redo_last_move()
                return
            elif move == 'y':
                self.logger.log("Retrying yellow detection...")
                if self._try_yellow_once():
                    return
                self.logger.error("Yellow detection failed - try again or enter move manually")
            elif move == 'd':
                print(f"\n=== DIAGNOSTICS ===")
                print(f"Turn: {'White' if self.board.turn == chess.WHITE else 'Black'}")
                print(f"FEN: {self.board.fen()}")
                print(f"Last bot move: {self.last_move[0] if self.last_move else 'None'}")
                print(f"Board location: {self.vision.board_location}")
                print(f"All legal moves:")
                print(', '.join([m.uci() for m in self.board.legal_moves]))
                print(f"==================\n")
            else:
                try:
                    self.board.push_uci(move)
                    return
                except ValueError:
                    self.logger.error(f"Invalid move: {move}")
                    print(f"Legal moves: {', '.join([m.uci() for m in list(self.board.legal_moves)[:10]])}...")

    def redo_last_move(self):
        """undoes board state and re-executes the last bot move on screen"""
        if not self.last_move:
            self.logger.error("No previous move to redo.")
            return False
        uci, start_coords, end_coords, promo, sq_size, is_white = self.last_move
        self.logger.warning(f"Redoing move: {uci}")
        self.board.pop()
        self.overlay.draw_move_arrow(start_coords, end_coords)
        time.sleep(0.3)
        self.overlay.clear()  # clear before clicking
        self.humanizer.make_move(start_coords, end_coords, promo, sq_size, is_white)
        self.board.push_uci(uci)
        self.logger.success(f"Replayed: {uci}")
        return True

    def wait_for_opponent_move(self):
        """watches screen for pixel changes - waits for stable state"""
        self.logger.log("Waiting for opponent...")
        self.logger.log("Press 'R' to retry bot move | 'Y' for yellow detect | 'Q' to quit")

        # wait a bit for our own animation to finish
        time.sleep(1.0)
        previous_state_img = self.vision.capture_screen()

        attempt = 0
        max_attempts = 20

        while attempt < max_attempts:
            if keyboard.is_pressed('q'):
                self.logger.warning("Quit detected.")
                return None

            if keyboard.is_pressed('r'):
                self.logger.warning("Redo triggered - retrying bot move...")
                time.sleep(0.5)  # debounce
                return "redo"

            if keyboard.is_pressed('y'):
                self.logger.warning("Yellow highlight detection triggered!")
                time.sleep(0.5)  # debounce
                return "yellow"

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
            attempt += 1

        # After max attempts, fall through to yellow detect + manual input
        self.logger.warning(f"Move detection timed out after {max_attempts} attempts")
        self.logger.warning("Switching to manual input mode...")
        return "yellow"
                
    def run(self):
        self.logger.log(f"Initializing Ghost-Shell v{APP_VERSION}...")
        
        # ask user which side they're playing
        print("\n" + "="*50)
        print("Which side are you playing?")
        print("  [W] White (you move first)")
        print("  [B] Black (opponent moves first)")
        print("  [A] Auto-detect (may not be accurate)")
        print("  [Enter] Auto-detect (default)")
        print("="*50)

        side_input = input("Enter W/B/A (or just Enter for auto): ").strip().upper()
        if not side_input:
            side_input = "A"
        
        if side_input == "W":
            self.user_side = chess.WHITE
        elif side_input == "B":
            self.user_side = chess.BLACK
        else:
            # will try auto-detect after board found
            pass
        
        self.logger.warning("Make sure the board is fully visible, then press Enter.")
        input()
        
        location = self.vision.find_board()
        if not location:
            self.logger.error("Couldnt find board. Exiting.")
            return
        
        # snap overlay to board
        self.overlay.update_geometry(*location)

        # auto-detect if user chose A
        if side_input == "A" or PLAYER_SIDE == "AUTO":
            detected_side = self.vision.detect_player_side()
            if detected_side is not None:
                self.user_side = detected_side

        self.logger.success(f"Board locked. Playing as {'White' if self.user_side == chess.WHITE else 'Black'}.")

        # --- NEW/RESUME ---
        print("\n" + "="*50)
        print("  [N] New game  (default)")
        print("  [F] Resume from FEN  (paste FEN from chess.com)")
        print("="*50)
        mode = input("New game or Resume? [N/F]: ").strip().upper()
        if mode == "F":
            while True:
                print("\nHow to get FEN: chess.com game menu → Analysis → copy FEN from the box.")
                fen_str = input("Paste FEN here: ").strip()
                try:
                    self.board = chess.Board(fen_str)
                    turn_str = "White" if self.board.turn == chess.WHITE else "Black"
                    self.logger.success(f"Loaded FEN. It is {turn_str}'s turn.")
                    self.logger.success(f"Playing as {'White' if self.user_side == chess.WHITE else 'Black'}.")
                    break
                except ValueError:
                    self.logger.error("Invalid FEN string - try again.")
        # ------------------

        # if playing black AND starting a new game, wait for opponent's first move
        if mode != "F" and self.user_side == chess.BLACK:
            self.logger.log("Playing as Black - waiting for White's first move...")
            print("\nEnter opponent's first move when ready.")
            while True:
                move = input("Opponent's move (e.g. e2e4): ").strip().lower()
                if move:
                    try:
                        self.board.push_uci(move)
                        break
                    except ValueError:
                        self.logger.error(f"Invalid move: {move}")
                        print(f"Legal moves: {', '.join([m.uci() for m in list(self.board.legal_moves)[:10]])}...")
        
        while not self.board.is_game_over():
            
            if self.board.turn == self.user_side:
                think_time = random.uniform(THINK_TIME_MIN, THINK_TIME_MAX)
                self.logger.log(f"My turn. Thinking for {think_time:.1f}s...")
                time.sleep(think_time)
                
                fen = self.board.fen()
                best_move_uci = self.engine.get_human_move(fen)
                
                if best_move_uci:
                    start_sq = best_move_uci[:2]
                    end_sq = best_move_uci[2:4]
                    
                    start_coords = self.get_square_center(start_sq)
                    end_coords = self.get_square_center(end_sq)
                    
                    promotion_piece = None
                    if len(best_move_uci) > 4:
                        promotion_piece = best_move_uci[4]
                    
                    sq_size = int(self.vision.square_size)
                    is_white = self.user_side == chess.WHITE
                    
                    # show the move on HUD, then clear BEFORE clicking
                    self.overlay.draw_move_arrow(start_coords, end_coords)
                    time.sleep(0.3)
                    self.overlay.clear()  # clear first - overlay could eat clicks if click-through fails

                    self.humanizer.make_move(start_coords, end_coords, promotion_piece, sq_size, is_white)
                    self.board.push_uci(best_move_uci)
                    self.last_move = (best_move_uci, start_coords, end_coords, promotion_piece, sq_size, is_white)

                    # log the move made
                    self.logger.success(f"Played: {best_move_uci}")
                
            else:
                self.logger.warning("Opponent's turn.")
                print(f"\n{self.board}")
                print(f"\nLegal moves: {', '.join([m.uci() for m in list(self.board.legal_moves)[:10]])}...")

                detected = self.wait_for_opponent_move()

                if detected is None:
                    break
                elif detected == "redo":
                    # bot move didn't land - retry the click, then loop back to wait for opponent
                    success = self.redo_last_move()
                    if not success:
                        self.logger.error("Nothing to redo.")
                else:
                    # movement detected or manual Y - auto-scan yellow, fall back to manual if needed
                    self._try_yellow_then_manual()

        self.logger.success("Game Over.")

if __name__ == "__main__":
    bot = GhostShell()
    bot.run()