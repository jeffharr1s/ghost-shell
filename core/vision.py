import cv2
import mss
import numpy as np
import time
import chess
from utils.logger import Logger

# Piece detection thresholds (tuned to require clear edges)
# Lowered thresholds to catch opponent pawns better
EDGE_MIN = 0.02  # reduced from 0.04 to catch smaller/fainter pieces
STD_MIN = 8.0    # reduced from 14.0 to catch opponent pawns
CONTRAST_MIN = 40.0  # reduced from 60.0 to be more lenient

# Color classification threshold
# Higher means more conservative. If you get too many '?' results, lower this a bit.
COLOR_DELTA_MIN = 8.0  # reduced from 12.0 for more lenient color detection


class GhostVision:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # Monitor 1 specifically (left 4K display)
        self.board_location = None
        self.square_size = 0
        self.logger = Logger("VISION")
        self.monitor1_width = 3840  # Width of Monitor 1 (left 4K display)

    def capture_screen(self):
        img = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def find_board(self):
        """tries multiple methods to find the board"""
        self.logger.log("Looking for board...")

        # try color-based detection first (works better with pieces)
        result = self._find_board_by_colors()
        if result:
            return result

        # fallback to corner detection (works on empty boards)
        result = self._find_board_by_corners()
        if result:
            return result

        self.logger.error("Couldnt find board. Try these:")
        self.logger.error("- Make sure the FULL board is visible")
        self.logger.error("- No overlays/popups blocking it")
        self.logger.error("- Try zooming browser to make board bigger")
        return None

    def _find_board_by_colors(self):
        """finds board by looking for grid pattern of light/dark squares"""
        frame = self.capture_screen()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10000:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)

                if 0.8 < aspect_ratio < 1.2 and w > 200:
                    candidates.append((x, y, w, h, area))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[4], reverse=True)

        for x, y, w, h, _ in candidates[:5]:
            if self._verify_chessboard(frame, x, y, w, h):
                self.board_location = (x, y, w, h)
                self.square_size = w / 8
                self.logger.success(f"Found board at ({x}, {y}) size {w}x{h}")
                return self.board_location

        return None

    def _verify_chessboard(self, frame, x, y, w, h):
        """checks if region actually looks like a chessboard"""
        try:
            roi = frame[y:y + h, x:x + w]
            if roi.size == 0:
                return False

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            sq_h = h // 8
            sq_w = w // 8

            samples = []
            for row in range(8):
                for col in range(8):
                    sq_y = row * sq_h + sq_h // 4
                    sq_x = col * sq_w + sq_w // 4
                    sq_y2 = row * sq_h + 3 * sq_h // 4
                    sq_x2 = col * sq_w + 3 * sq_w // 4

                    if sq_y2 <= h and sq_x2 <= w:
                        square_region = gray_roi[sq_y:sq_y2, sq_x:sq_x2]
                        if square_region.size > 0:
                            avg = np.mean(square_region)
                            expected_light = (row + col) % 2 == 0
                            samples.append((avg, expected_light))

            if len(samples) < 32:
                return False

            light_squares = [s[0] for s in samples if s[1]]
            dark_squares = [s[0] for s in samples if not s[1]]

            if not light_squares or not dark_squares:
                return False

            avg_light = np.mean(light_squares)
            avg_dark = np.mean(dark_squares)

            return abs(avg_light - avg_dark) > 20

        except Exception:
            return False

    def _find_board_by_corners(self):
        """opencv corner method, works on empty boards"""
        frame = self.capture_screen()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

        if ret:
            self.logger.log("Found via corner detection.")

            points = corners.reshape(-1, 2)
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)

            w_inner = x_max - x_min
            estimated_sq_size = w_inner / 6

            board_x = int(x_min - estimated_sq_size)
            board_y = int(y_min - estimated_sq_size)
            board_w = int(w_inner + (2 * estimated_sq_size))
            board_h = board_w

            self.board_location = (board_x, board_y, board_w, board_h)
            self.square_size = board_w / 8

            return self.board_location

        return None

    def detect_player_side(self):
        """figures out if youre white or black by checking corner brightness"""
        if not self.board_location:
            self.logger.error("Find the board first.")
            return None

        bottom_left = self.get_square_roi(7, 0)
        bottom_right = self.get_square_roi(7, 7)

        if bottom_left is None or bottom_right is None:
            return None

        h, w = bottom_left.shape[:2]
        center_left = bottom_left[h // 4:3 * h // 4, w // 4:3 * w // 4]
        center_right = bottom_right[h // 4:3 * h // 4, w // 4:3 * w // 4]

        gray_left = cv2.cvtColor(center_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(center_right, cv2.COLOR_BGR2GRAY)

        avg_left = np.mean(gray_left)
        avg_right = np.mean(gray_right)

        if avg_left > 120 or avg_right > 120:
            self.logger.success("Playing as WHITE")
            return chess.WHITE

        self.logger.success("Playing as BLACK")
        return chess.BLACK

    def get_square_roi(self, rank, file):
        """gets image of a specific square"""
        if not self.board_location:
            return None

        bx, by, bw, bh = self.board_location
        sq_s = self.square_size

        y_start = int(by + (rank * sq_s))
        y_end = int(y_start + sq_s)
        x_start = int(bx + (file * sq_s))
        x_end = int(x_start + sq_s)

        full_screen = self.capture_screen()
        return full_screen[y_start:y_end, x_start:x_end]

    def debug_draw_board(self):
        """saves debug image with grid overlay"""
        if not self.board_location:
            print("Find board first.")
            return

        img = self.capture_screen()
        x, y, w, h = self.board_location

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        sq = self.square_size
        for r in range(8):
            for c in range(8):
                sx = int(x + c * sq)
                sy = int(y + r * sq)
                cv2.rectangle(img, (sx, sy), (int(sx + sq), int(sy + sq)), (0, 0, 255), 1)

        cv2.imwrite("debug_vision.jpg", img)
        print("Saved debug_vision.jpg")

    def _detect_piece_in_square(self, cell_img, rank_idx, file_idx):
        """
        Advanced piece detection using multiple methods with debug logging.
        Returns True if a piece is likely present.
        """
        if cell_img.size == 0:
            return False

        try:
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.count_nonzero(edges) / edges.size

            std_dev = np.std(gray)

            min_val = np.min(gray)
            max_val = np.max(gray)
            contrast = max_val - min_val

            has_edges = edge_density >= EDGE_MIN
            has_variance = std_dev >= STD_MIN
            has_contrast = contrast >= CONTRAST_MIN

            result = has_edges and (has_variance or has_contrast)

            file = chess.FILE_NAMES[file_idx]
            rank = 8 - rank_idx
            self.logger.log(
                f"{file}{rank}: edge={edge_density:.3f}({'ok' if has_edges else 'no'}), "
                f"std={std_dev:.1f}({'ok' if has_variance else 'no'}), "
                f"contrast={contrast:.0f}({'ok' if has_contrast else 'no'}) "
                f"-> {'PIECE' if result else 'empty'}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Detection error at {file_idx},{rank_idx}: {e}")
            return False

    def _classify_piece_color(self, cell_img, rank_idx, file_idx):
        """
        Returns 'w' or 'b' for likely piece color, or '?' if uncertain.

        Strategy:
        - Compare mean brightness of center region vs corner regions of the same square.
        - If center is much brighter than corners, likely white piece.
        - If center is much darker than corners, likely black piece.
        """
        try:
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            if h < 10 or w < 10:
                return '?'

            cy1, cy2 = h // 3, 2 * h // 3
            cx1, cx2 = w // 3, 2 * w // 3
            center = gray[cy1:cy2, cx1:cx2]

            patch = max(2, min(h, w) // 8)
            corners = [
                gray[0:patch, 0:patch],
                gray[0:patch, w - patch:w],
                gray[h - patch:h, 0:patch],
                gray[h - patch:h, w - patch:w],
            ]

            center_mean = float(np.mean(center))
            bg_mean = float(np.mean([np.mean(c) for c in corners]))
            delta = center_mean - bg_mean

            if delta >= COLOR_DELTA_MIN:
                color = 'w'
            elif delta <= -COLOR_DELTA_MIN:
                color = 'b'
            else:
                color = '?'

            file = chess.FILE_NAMES[file_idx]
            rank = 8 - rank_idx
            self.logger.log(f"{file}{rank}: color delta={delta:.1f} -> {color}")

            return color

        except Exception as e:
            self.logger.error(f"Color classify error at {file_idx},{rank_idx}: {e}")
            return '?'

    def get_board_piece_map(self, sample_count: int = 3, sample_delay: float = 0.05):
        """
        Capture the current board region and return a mapping:
        {(file, rank): 'w' or 'b' or '?'}.
        Empty squares are omitted.
        """
        try:
            if not self.board_location:
                self.logger.error("Board not locked. Run find_board() first.")
                return {}

            sample_count = max(1, sample_count)
            self.logger.debug(f"Starting piece detection with {sample_count} sample(s)")

        bx, by, bw, bh = self.board_location
        sq_size = int(self.square_size)

        all_maps = []
        for idx in range(sample_count):
            frame = self.capture_screen()
            piece_map = {}

            for rank_idx in range(8):
                for file_idx in range(8):
                    x1 = bx + file_idx * sq_size
                    y1 = by + rank_idx * sq_size
                    x2 = x1 + sq_size
                    y2 = y1 + sq_size

                    cell_img = frame[y1:y2, x1:x2]
                    has_piece = self._detect_piece_in_square(cell_img, rank_idx, file_idx)
                    if not has_piece:
                        continue

                    color = self._classify_piece_color(cell_img, rank_idx, file_idx)

                    file = chess.FILE_NAMES[file_idx]
                    rank = 8 - rank_idx
                    piece_map[(file, rank)] = color

            all_maps.append(piece_map)
            if idx < sample_count - 1:
                time.sleep(sample_delay)

        if sample_count == 1:
            final_map = all_maps[0]
        else:
            votes = {}
            for pm in all_maps:
                for sq, col in pm.items():
                    if sq not in votes:
                        votes[sq] = {'w': 0, 'b': 0, '?': 0}
                    votes[sq][col] += 1

            majority = (sample_count // 2) + 1
            final_map = {}
            for sq, v in votes.items():
                best_col = max(v, key=lambda k: v[k])
                if v[best_col] >= majority:
                    final_map[sq] = best_col

            self.logger.debug(f"Detected {len(final_map)} pieces after voting")
            if not final_map:
                self.logger.warning("No pieces detected on board - this could indicate:")
                self.logger.warning("  1. Detection thresholds too strict (EDGE_MIN, STD_MIN, CONTRAST_MIN)")
                self.logger.warning("  2. Board not fully visible or misaligned")
                self.logger.warning("  3. Unusual lighting conditions or glare on the board")
                self.logger.warning("  4. Yellow highlighting on last move interfering with detection")

            # Log the final piece map for debugging
            if final_map:
                self.logger.debug("Final piece map:")
                for sq in sorted(final_map.keys()):
                    self.logger.debug(f"  {sq}: {final_map[sq]}")

            return final_map

        except Exception as e:
            self.logger.error(f"Exception in get_board_piece_map: {type(e).__name__}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}

    def detect_move_from_maps(self, prev_map, curr_map):
        """
        Compare two colored piece maps and infer the move.

        Returns tuple (from_square, to_square) like (('e', 2), ('e', 4))
        or None if no clear move detected.
        """
        removed = {}
        added = {}
        changed = {}

        for sq, p in prev_map.items():
            if sq not in curr_map:
                removed[sq] = p
            else:
                if curr_map[sq] != p:
                    changed[sq] = (p, curr_map[sq])

        for sq, p in curr_map.items():
            if sq not in prev_map:
                added[sq] = p

        # Log the differences found
        self.logger.debug(f"MOVE DETECTION ANALYSIS:")
        self.logger.debug(f"  Pieces removed (left board): {removed}")
        self.logger.debug(f"  Pieces added (new position):  {added}")
        self.logger.debug(f"  Pieces changed (color flip):  {changed}")

        # Normal move: from becomes empty, to becomes occupied
        if len(removed) == 1 and len(added) == 1:
            from_sq = list(removed.keys())[0]
            to_sq = list(added.keys())[0]
            self.logger.debug(f"  -> NORMAL MOVE: {from_sq} to {to_sq}")
            return (from_sq, to_sq)

        # Capture: from becomes empty, destination stays occupied but color flips
        if len(removed) == 1 and len(added) == 0 and len(changed) == 1:
            from_sq = list(removed.keys())[0]
            to_sq = list(changed.keys())[0]
            self.logger.debug(f"  -> CAPTURE MOVE: {from_sq} captures on {to_sq}")
            return (from_sq, to_sq)

        # En passant like pattern: from + captured pawn removed, to added
        if len(removed) == 2 and len(added) == 1:
            to_sq = list(added.keys())[0]
            for sq in removed.keys():
                if sq != to_sq:
                    self.logger.debug(f"  -> EN PASSANT MOVE: {sq} to {to_sq}")
                    return (sq, to_sq)

        # No move detected - log why
        self.logger.warning(f"  -> NO MOVE DETECTED")
        self.logger.warning(f"     Removed: {len(removed)}, Added: {len(added)}, Changed: {len(changed)}")
        if removed:
            self.logger.warning(f"     Removed squares: {list(removed.keys())}")
        if added:
            self.logger.warning(f"     Added squares: {list(added.keys())}")
        if changed:
            self.logger.warning(f"     Changed squares: {list(changed.keys())}")

        return None

    @staticmethod
    def square_to_uci(square):
        """Convert (file, rank) tuple to UCI string like 'e2'"""
        file, rank = square
        return f"{file}{rank}"

    def diagnose_piece_detection(self, prev_map, curr_map):
        """
        Compare piece detection across squares to find why detection failed.
        Helpful for debugging color/shading/highlighting issues.
        """
        self.logger.warning("=== PIECE DETECTION DIAGNOSIS ===")

        all_squares = set(prev_map.keys()) | set(curr_map.keys())

        for sq in sorted(all_squares):
            prev_color = prev_map.get(sq, None)
            curr_color = curr_map.get(sq, None)

            if prev_color is None and curr_color is None:
                continue  # Empty in both - skip

            if prev_color == curr_color:
                continue  # No change - skip

            # Something changed on this square
            file, rank = sq
            file_idx = chess.FILE_NAMES.index(file)
            rank_idx = 8 - rank

            self.logger.warning(f"  {sq}: {prev_color} -> {curr_color}")

            # Re-capture and analyze this specific square
            try:
                frame = self.capture_screen()
                bx, by, bw, bh = self.board_location
                sq_size = int(self.square_size)

                x1 = bx + file_idx * sq_size
                y1 = by + rank_idx * sq_size
                x2 = x1 + sq_size
                y2 = y1 + sq_size

                cell_img = frame[y1:y2, x1:x2]

                # Check if piece detection is working
                has_piece = self._detect_piece_in_square(cell_img, rank_idx, file_idx)
                detected_color = self._classify_piece_color(cell_img, rank_idx, file_idx) if has_piece else 'none'

                self.logger.warning(f"    Current detection: has_piece={has_piece}, color={detected_color}")

            except Exception as e:
                self.logger.error(f"    Error diagnosing {sq}: {e}")

    def detect_opponent_move_uci(self, prev_map, sample_count: int = 3):
        """
        Detect opponent move by comparing previous board state to current.
        Returns UCI move string like 'e2e4' or None.
        """
        try:
            self.logger.debug(f"detect_opponent_move_uci: prev_map has {len(prev_map)} pieces")
            curr_map = self.get_board_piece_map(sample_count=sample_count)
            if not curr_map:
                self.logger.warning("Could not detect ANY pieces on the current frame.")
                self.logger.warning("This might be due to: wrong threshold values, board not visible, bad lighting, or highlighting interference")
                return None

            self.logger.debug(f"curr_map has {len(curr_map)} pieces, prev_map has {len(prev_map)} pieces")
            move = self.detect_move_from_maps(prev_map, curr_map)
            if not move:
                self.logger.warning("No move pattern detected from piece comparison")
                # Run diagnosis to understand why
                self.diagnose_piece_detection(prev_map, curr_map)
                return None

            from_sq, to_sq = move
            uci_move = self.square_to_uci(from_sq) + self.square_to_uci(to_sq)
            self.logger.debug(f"Detected move: {uci_move}")
            return uci_move
        except Exception as e:
            self.logger.error(f"Exception in detect_opponent_move_uci: {type(e).__name__}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None


if __name__ == "__main__":
    vision = GhostVision()
    print("Open a chess board in 3 seconds...")
    time.sleep(3)

    loc = vision.find_board()
    if loc:
        print(f"Board at: {loc}")
        vision.debug_draw_board()
    else:
        print("Failed to detect.")
