import time
import random
import sys
import itertools
import os
import cv2
import chess
import keyboard
import numpy as np
from core.vision import GhostVision
from core.engine import GhostEngine
from core.humanizer import Humanizer
from ui.overlay import GhostOverlay
from utils.logger import Logger
from utils.config import (
    PLAYER_SIDE,
    APP_VERSION,
    GAME_MODE_PRESETS,
    DEFAULT_GAME_MODE,
    QUICK_START,
    DEFAULT_START_MODE,
    PAUSE_ON_EXIT,
)


class RestartMainRequested(Exception):
    """Raised when the user wants to relaunch main.py."""


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
        self.baseline_yellow = set()  # yellow squares captured right after our move (stale highlights to ignore)
        self.timing = GAME_MODE_PRESETS[DEFAULT_GAME_MODE]  # default; may be overridden at startup

    def _apply_side_choice(self, side_input):
        choice = (side_input or "AUTO").upper()
        if choice == "W" or choice == "WHITE":
            self.user_side = chess.WHITE
            return "W"
        if choice == "B" or choice == "BLACK":
            self.user_side = chess.BLACK
            return "B"
        return "A"

    def _choose_start_mode(self):
        if QUICK_START:
            mode = DEFAULT_START_MODE
            self.logger.success(f"Quick start game default: {'Resume from FEN' if mode == 'F' else 'New game'}")
            return mode, None

        print("\n" + "="*50)
        print("  Paste FEN + Enter = resume")
        print("  Enter with nothing = new game")
        print("="*50)
        while True:
            choice = input("FEN or Enter for new game: ").strip()
            if not choice:
                return "N", None

            try:
                chess.Board(choice)
                return "F", choice
            except ValueError:
                self.logger.error("That did not look like a valid FEN. Paste a valid FEN, or press Enter for a new game.")

    def _apply_default_game_mode(self):
        mode_key = DEFAULT_GAME_MODE
        self.timing = GAME_MODE_PRESETS[mode_key]
        self.logger.success(f"Mode: {mode_key} - think {self.timing['think_min']}-{self.timing['think_max']}s")
        return mode_key

    def _detect_side_from_board_bottom(self):
        detected_side = self.vision.detect_player_side()
        if detected_side is not None:
            self.user_side = detected_side
            return

        fallback = self._apply_side_choice(PLAYER_SIDE)
        if fallback == "A":
            self.user_side = chess.WHITE
            self.logger.warning("Could not auto-detect bottom side; defaulting to White.")
            return

        self.logger.warning(f"Could not auto-detect bottom side; using PLAYER_SIDE={PLAYER_SIDE}.")

    def _capture_baseline(self):
        """Snapshots yellow squares right now so future scans can subtract stale highlights."""
        time.sleep(0.3)
        squares = self.vision.detect_yellow_highlights(is_white=(self.user_side == chess.WHITE))
        self.baseline_yellow = set(squares)
        if squares:
            self.logger.log(f"Baseline yellow: {squares}")

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
        yellow_squares = self.vision.detect_yellow_highlights(is_white=(self.user_side == chess.WHITE))
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

        # also subtract squares captured in our post-move baseline snapshot (persistent stale highlights)
        if self.baseline_yellow:
            fresh = [sq for sq in candidates if sq not in self.baseline_yellow]
            if len(fresh) >= 2:
                candidates = fresh
                self.logger.log(f"Filtered baseline {self.baseline_yellow}, using: {candidates}")

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
        retry_delay = self.timing["yellow_retry"]

        for attempt in range(1, max_attempts + 1):
            wait = self.timing["yellow_wait"] if attempt == 1 else retry_delay
            self.logger.log(f"Yellow scan attempt {attempt}/{max_attempts} (waiting {wait}s)...")
            time.sleep(wait)
            if self._try_yellow_once():
                return
            self.logger.error(f"Attempt {attempt}: no valid move found, retrying...")

        self.logger.error(f"Yellow failed after {max_attempts} attempts - enter move manually")

        # fallback: manual input (only reached if yellow failed)
        while True:
            move = input("Opponent's move (e.g. e2e4), 'y'=yellow, 'r'=redo, 'd'=diag, 'f'=FEN, 's'=I moved, 'new'=restart, 'q'=quit: ").strip().lower()
            if move in ('q', 'quit', 'exit') or move.startswith('q '):
                self.logger.warning("Quit.")
                sys.exit(0)
            elif move in ('new', 'restart'):
                self.logger.warning("Restart requested from manual move prompt.")
                raise RestartMainRequested
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
            elif move == 'f':
                fen_str = input("Paste FEN: ").strip()
                try:
                    self.board = chess.Board(fen_str)
                    self.baseline_yellow.clear()  # force re-capture at new position
                    turn_str = "White" if self.board.turn == chess.WHITE else "Black"
                    self.logger.success(f"FEN loaded. It is {turn_str}'s turn.")
                    return
                except ValueError:
                    self.logger.error("Invalid FEN - try again.")
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
        self.logger.log("Press 'R' to retry bot move | 'Y' for yellow detect | 'N' for new game/restart | 'Q' to quit")

        # wait a bit for our own animation to finish
        time.sleep(self.timing["settle"])
        previous_state_img = self.vision.capture_screen()

        attempt = 0
        max_attempts = 20

        while attempt < max_attempts:
            if keyboard.is_pressed('q'):
                self.logger.warning("Quit detected.")
                sys.exit(0)

            if keyboard.is_pressed('n'):
                self.logger.warning("New game/restart hotkey detected.")
                time.sleep(0.5)  # debounce
                raise RestartMainRequested

            if keyboard.is_pressed('r'):
                self.logger.warning("Redo triggered - retrying bot move...")
                time.sleep(0.5)  # debounce
                return "redo"

            if keyboard.is_pressed('y'):
                self.logger.warning("Yellow highlight detection triggered!")
                time.sleep(0.5)  # debounce
                return "yellow"

            time.sleep(self.timing["poll"])
            current_state_img = self.vision.capture_screen()

            diff = cv2.absdiff(previous_state_img, current_state_img)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            non_zero = cv2.countNonZero(thresh)

            if non_zero > 1000:  # higher threshold
                self.logger.success("Movement detected!")
                time.sleep(self.timing["settle"])  # wait for animation to fully finish
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
        self.logger.log(f"Writing copy/paste log to: {Logger.current_log_path()}")

        mode, fen_str = self._choose_start_mode()
        self._apply_default_game_mode()
        
        if QUICK_START:
            self.logger.warning("Quick start: looking for the board immediately.")
        else:
            self.logger.warning("Make sure the board is fully visible, then press Enter.")
            input()
        
        location = self.vision.find_board()
        if not location:
            self.logger.error("Couldnt find board. Exiting.")
            return
        
        # snap overlay to board
        self.overlay.update_geometry(*location)

        self._detect_side_from_board_bottom()

        self.logger.success(f"Board locked. Playing as {'White' if self.user_side == chess.WHITE else 'Black'}.")

        if mode == "F":
            self.board = chess.Board(fen_str)
            self.baseline_yellow.clear()
            turn_str = "White" if self.board.turn == chess.WHITE else "Black"
            self.logger.success(f"Loaded FEN. It is {turn_str}'s turn.")
            self.logger.success(f"Playing as {'White' if self.user_side == chess.WHITE else 'Black'}.")
        # ------------------

        # If playing black AND starting a new game, auto-detect White's first move.
        if mode != "F" and self.user_side == chess.BLACK:
            self.logger.log("Playing as Black - auto-detecting White's first move...")
            self._try_yellow_then_manual()
        
        while not self.board.is_game_over():
            
            if self.board.turn == self.user_side:
                think_time = random.uniform(self.timing["think_min"], self.timing["think_max"])
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
                    self._capture_baseline()  # snapshot our move's highlights so they're filtered next scan

                    # log the move made
                    self.logger.success(f"Played: {best_move_uci}")

            else:
                # capture baseline once at game start (before bot has played anything)
                if not self.baseline_yellow:
                    self._capture_baseline()
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
        self._prompt_after_game()


    def _prompt_after_game(self):
        if not sys.stdin or not sys.stdin.isatty():
            return

        print("\n" + "="*50)
        print("  [Enter] Start a new game / restart main.py")
        print("  [Q] Quit")
        print("="*50)
        choice = input("Next action: ").strip().lower()
        if choice in ("", "n", "new", "restart", "r"):
            self.logger.warning("Restart requested after game over.")
            raise RestartMainRequested
        if choice == "q":
            self.logger.warning("Quit after game over.")
            sys.exit(0)


def _pause_before_exit(logger, reason):
    if not PAUSE_ON_EXIT or "--no-pause" in sys.argv:
        return
    if not sys.stdin or not sys.stdin.isatty():
        return

    logger.warning(reason)
    try:
        input("Press Enter to close Ghost-Shell...")
    except EOFError:
        pass


def _restart_main_py(logger):
    script_path = os.path.abspath(sys.argv[0] or __file__)
    args = [arg for arg in sys.argv[1:] if arg != "--no-pause"]
    logger.warning(f"Restarting main.py: {script_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, script_path, *args])


def main():
    logger = Logger("BOOT")
    exit_code = 0
    should_pause = True
    try:
        bot = GhostShell()
        bot.run()
    except RestartMainRequested:
        try:
            should_pause = False
            _restart_main_py(logger)
        except Exception as exc:
            should_pause = True
            logger.exception(f"Restart failed: {exc}", exc)
            exit_code = 1
    except KeyboardInterrupt:
        logger.warning("Stopped by keyboard interrupt.")
        exit_code = 130
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
        if exit_code:
            logger.error(f"Ghost-Shell exited with code {exit_code}. Paste this log into chat: {Logger.current_log_path()}")
        else:
            logger.warning("Ghost-Shell exited.")
    except Exception as exc:
        logger.exception(f"Unhandled crash: {exc}", exc)
        logger.error(f"Crash details were saved here for copy/paste: {Logger.current_log_path()}")
        exit_code = 1
    finally:
        if should_pause:
            _pause_before_exit(logger, "Ghost-Shell finished. The console will stay open so you can read any errors.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
