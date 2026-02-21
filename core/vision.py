import cv2
import mss
import numpy as np
import time
import chess
from utils.logger import Logger

# Piece detection thresholds
EDGE_MIN = 0.02
STD_MIN = 8.0
CONTRAST_MIN = 40.0

# Color classification: how far a piece center must deviate from the
# known square background brightness to be called white or black.
# This is relative to the gap between light-square and dark-square averages.
COLOR_RATIO_THRESHOLD = 0.10  # 10% of the light-dark gap (was 15%, too strict)


class GhostVision:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.board_location = None
        self.square_size = 0
        self.logger = Logger("VISION")
        self.monitor1_width = 3840
        # Learned board square colors (set during find_board)
        self.light_square_brightness = None
        self.dark_square_brightness = None
        # Board orientation: True = white on bottom (rank 8 at top of screen)
        # False = black on bottom (board flipped)
        self.white_on_bottom = True

    def set_orientation(self, user_side):
        """
        Set board orientation based on which side the user is playing.
        Call this after detect_player_side() or manual side selection.
        user_side: chess.WHITE or chess.BLACK
        """
        self.white_on_bottom = (user_side == chess.WHITE)
        self.logger.log(
            f"Board orientation: {'white' if self.white_on_bottom else 'black'} on bottom"
        )
        # Recalibrate since corner identity depends on orientation
        self._calibrate_square_colors()

    def capture_screen(self):
        img = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def find_board(self):
        """tries multiple methods to find the board"""
        self.logger.log("Looking for board...")

        result = self._find_board_by_colors()
        if result:
            return result

        result = self._find_board_by_corners()
        if result:
            return result

        self.logger.error("Couldnt find board. Try these:")
        self.logger.error("- Make sure the FULL board is visible")
        self.logger.error("- No overlays/popups blocking it")
        self.logger.error("- Try zooming browser to make board bigger")
        return None

    def _screen_to_square(self, rank_idx, file_idx):
        """
        Convert screen grid position (rank_idx=row from top, file_idx=col from left)
        to chess square (file_letter, rank_number) respecting board orientation.

        When white_on_bottom=True:  rank_idx 0 = rank 8, file_idx 0 = a-file
        When white_on_bottom=False: rank_idx 0 = rank 1, file_idx 0 = h-file
        """
        if self.white_on_bottom:
            file_letter = chess.FILE_NAMES[file_idx]
            rank_number = 8 - rank_idx
        else:
            file_letter = chess.FILE_NAMES[7 - file_idx]
            rank_number = rank_idx + 1
        return (file_letter, rank_number)

    def _square_color_on_board(self, file_letter, rank_number):
        """
        Returns True if the square is a light square on a standard chess board.
        a1 is dark, a2 is light, etc.
        """
        file_idx = chess.FILE_NAMES.index(file_letter)
        return (file_idx + rank_number) % 2 == 1

    def _calibrate_square_colors(self, frame=None):
        """
        Learn what light and dark squares look like by sampling the 4 corner
        squares of the board, taking orientation into account.
        """
        if not self.board_location:
            return
        if frame is None:
            frame = self.capture_screen()

        bx, by, bw, bh = self.board_location
        sq = int(self.square_size)

        def sample_square_edge(rank_idx, file_idx):
            """Return mean gray brightness of the outer ring of a square."""
            x1 = bx + file_idx * sq
            y1 = by + rank_idx * sq
            cell = frame[y1:y1 + sq, x1:x1 + sq]
            if cell.size == 0:
                return None
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            margin = max(3, min(h, w) // 8)
            top = gray[0:margin, :]
            bot = gray[h - margin:h, :]
            lft = gray[:, 0:margin]
            rgt = gray[:, w - margin:w]
            return float(np.mean(np.concatenate([top.ravel(), bot.ravel(),
                                                  lft.ravel(), rgt.ravel()])))

        light_samples = []
        dark_samples = []

        # Sample all 4 corners and use orientation to determine which is light/dark
        corner_positions = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for ri, fi in corner_positions:
            file_letter, rank_number = self._screen_to_square(ri, fi)
            is_light = self._square_color_on_board(file_letter, rank_number)
            val = sample_square_edge(ri, fi)
            if val is not None:
                if is_light:
                    light_samples.append(val)
                else:
                    dark_samples.append(val)

        if light_samples and dark_samples:
            self.light_square_brightness = float(np.mean(light_samples))
            self.dark_square_brightness = float(np.mean(dark_samples))
            self.logger.log(
                f"Calibrated squares: light={self.light_square_brightness:.1f}, "
                f"dark={self.dark_square_brightness:.1f}"
            )
        else:
            self.logger.warning("Could not calibrate square colors, using defaults")
            self.light_square_brightness = 200.0
            self.dark_square_brightness = 130.0

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
                self._calibrate_square_colors(frame)
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
            self._calibrate_square_colors(frame)

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
        """gets image of a specific square (rank/file are screen grid indices)"""
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
                # Label with chess square name
                fl, rn = self._screen_to_square(r, c)
                cv2.putText(img, f"{fl}{rn}", (sx + 2, sy + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.imwrite("debug_vision.jpg", img)
        print("Saved debug_vision.jpg")

    def _detect_piece_in_square(self, cell_img, rank_idx, file_idx):
        """
        Returns True if a piece is likely present in this cell.
        Uses edge density + variance/contrast.
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

            return has_edges and (has_variance or has_contrast)

        except Exception as e:
            self.logger.error(f"Detection error at {file_idx},{rank_idx}: {e}")
            return False

    def _classify_piece_color(self, cell_img, rank_idx, file_idx):
        """
        Returns 'w' or 'b' for likely piece color, or '?' if uncertain.

        Uses a multi-strategy approach:
        1. Edge-vs-center comparison (most robust, works on any background)
        2. Absolute brightness thresholds (fallback for highlighted squares)
        3. Calibrated background comparison (normal squares)
        """
        try:
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            if h < 10 or w < 10:
                return '?'

            # --- Detect if this square is highlighted ---
            hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
            sat_mean = float(np.mean(hsv[:, :, 1]))
            # Both Lichess and chess.com highlights have elevated saturation
            is_highlighted = sat_mean > 45

            # --- Sample the piece body (center 50% of the cell) ---
            # Widened from 40% to 50% to better capture piece pixels
            margin_y = int(h * 0.25)
            margin_x = int(w * 0.25)
            center = gray[margin_y:h - margin_y, margin_x:w - margin_x]
            if center.size == 0:
                return '?'
            center_mean = float(np.mean(center))

            # --- Sample the edge of the cell (background) ---
            edge_strip = max(3, min(h, w) // 8)
            edge_pixels = np.concatenate([
                gray[0:edge_strip, :].ravel(),
                gray[h - edge_strip:h, :].ravel(),
                gray[:, 0:edge_strip].ravel(),
                gray[:, w - edge_strip:w].ravel(),
            ])
            edge_mean = float(np.mean(edge_pixels))

            # --- Strategy 1: Edge-vs-center comparison (primary) ---
            # This works regardless of background color because we're comparing
            # the piece body to its own surrounding square.
            delta_from_edge = center_mean - edge_mean

            if abs(delta_from_edge) > 15:
                # Strong signal: piece is clearly different from its background
                if delta_from_edge > 15:
                    return 'w'  # piece brighter than background = white piece
                else:
                    return 'b'  # piece darker than background = black piece

            # --- Strategy 2: Absolute brightness (for highlighted/ambiguous) ---
            if is_highlighted:
                if center_mean > 155:
                    return 'w'
                elif center_mean < 105:
                    return 'b'
                # Still ambiguous with highlight - use a slightly relaxed edge delta
                if delta_from_edge > 8:
                    return 'w'
                elif delta_from_edge < -8:
                    return 'b'
                return '?'

            # --- Strategy 3: Calibrated background comparison ---
            file_letter, rank_number = self._screen_to_square(rank_idx, file_idx)
            is_light_square = self._square_color_on_board(file_letter, rank_number)

            if self.light_square_brightness is not None and self.dark_square_brightness is not None:
                expected_bg = (self.light_square_brightness if is_light_square
                               else self.dark_square_brightness)
                bg_gap = abs(self.light_square_brightness - self.dark_square_brightness)
            else:
                expected_bg = 200.0 if is_light_square else 130.0
                bg_gap = 70.0

            bg_gap = max(bg_gap, 20.0)

            delta_from_bg = center_mean - expected_bg
            threshold = bg_gap * COLOR_RATIO_THRESHOLD

            if delta_from_bg > threshold:
                return 'w'
            elif delta_from_bg < -threshold:
                return 'b'

            # --- Last resort: absolute brightness ---
            if center_mean > 170:
                return 'w'
            elif center_mean < 90:
                return 'b'

            return '?'

        except Exception as e:
            self.logger.error(f"Color classify error at {file_idx},{rank_idx}: {e}")
            return '?'

    def get_board_piece_map(self, sample_count: int = 3, sample_delay: float = 0.05):
        """
        Capture the current board and return a mapping:
        {(file_letter, rank_number): 'w' or 'b' or '?'}.
        Empty squares are omitted.

        Respects board orientation (white_on_bottom flag).
        """
        if not self.board_location:
            self.logger.error("Board not locked. Run find_board() first.")
            return {}

        sample_count = max(1, sample_count)

        try:
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

                        # Convert screen position to chess square using orientation
                        file_letter, rank_number = self._screen_to_square(rank_idx, file_idx)
                        piece_map[(file_letter, rank_number)] = color

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

            # Summary log (not per-square spam)
            w_count = sum(1 for c in final_map.values() if c == 'w')
            b_count = sum(1 for c in final_map.values() if c == 'b')
            q_count = sum(1 for c in final_map.values() if c == '?')
            self.logger.debug(
                f"Piece map: {len(final_map)} pieces "
                f"(w={w_count}, b={b_count}, ?={q_count})"
            )

            if not final_map:
                self.logger.warning("No pieces detected on board!")

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

        Treats '?' as a wildcard: a ?-to-w or w-to-? change on the same square
        is NOT considered a color flip (it's the same piece, just uncertain read).
        """
        removed = {}
        added = {}
        changed = {}

        def colors_differ(c1, c2):
            """True only if both colors are known and they differ."""
            if c1 == '?' or c2 == '?':
                return False
            return c1 != c2

        for sq, p in prev_map.items():
            if sq not in curr_map:
                removed[sq] = p
            elif colors_differ(p, curr_map[sq]):
                changed[sq] = (p, curr_map[sq])

        for sq, p in curr_map.items():
            if sq not in prev_map:
                added[sq] = p

        self.logger.debug(f"Move diff: removed={list(removed.keys())}, "
                          f"added={list(added.keys())}, changed={list(changed.keys())}")

        # Normal move: one square emptied, one square newly occupied
        if len(removed) == 1 and len(added) == 1:
            from_sq = list(removed.keys())[0]
            to_sq = list(added.keys())[0]
            self.logger.debug(f"  -> NORMAL MOVE: {from_sq} to {to_sq}")
            return (from_sq, to_sq)

        # Capture: attacker's origin empties, defender's square changes color
        if len(removed) == 1 and len(added) == 0 and len(changed) == 1:
            from_sq = list(removed.keys())[0]
            to_sq = list(changed.keys())[0]
            self.logger.debug(f"  -> CAPTURE: {from_sq} captures on {to_sq}")
            return (from_sq, to_sq)

        # En passant: two pieces removed (pawn + captured pawn), one added
        if len(removed) == 2 and len(added) == 1:
            to_sq = list(added.keys())[0]
            for sq in removed.keys():
                if sq != to_sq:
                    self.logger.debug(f"  -> EN PASSANT: {sq} to {to_sq}")
                    return (sq, to_sq)

        # Castling: king + rook both move (2 removed, 2 added)
        if len(removed) == 2 and len(added) == 2:
            king_from = None
            for sq in removed:
                if sq[0] == 'e':
                    king_from = sq
                    break
            if king_from:
                for sq in added:
                    if sq[0] in ('c', 'g') and sq[1] == king_from[1]:
                        self.logger.debug(f"  -> CASTLING: {king_from} to {sq}")
                        return (king_from, sq)

        # Fuzzy fallback: best pair from removed/added
        if len(removed) >= 1 and len(added) >= 1 and len(removed) + len(added) <= 4:
            best_pair = None
            best_dist = -1
            for r_sq in removed:
                for a_sq in added:
                    dist = abs(ord(r_sq[0]) - ord(a_sq[0])) + abs(r_sq[1] - a_sq[1])
                    if dist > best_dist:
                        best_dist = dist
                        best_pair = (r_sq, a_sq)
            if best_pair:
                self.logger.debug(f"  -> FUZZY: {best_pair[0]} to {best_pair[1]}")
                return best_pair

        self.logger.warning(f"  -> NO MOVE DETECTED "
                            f"(rem={len(removed)}, add={len(added)}, chg={len(changed)})")
        return None

    @staticmethod
    def square_to_uci(square):
        """Convert (file, rank) tuple to UCI string like 'e2'"""
        file, rank = square
        return f"{file}{rank}"

    def diagnose_piece_detection(self, prev_map, curr_map):
        """Compare maps square-by-square for debugging."""
        self.logger.warning("=== PIECE DETECTION DIAGNOSIS ===")

        all_squares = set(prev_map.keys()) | set(curr_map.keys())

        for sq in sorted(all_squares):
            prev_color = prev_map.get(sq, None)
            curr_color = curr_map.get(sq, None)

            if prev_color == curr_color:
                continue

            self.logger.warning(f"  {sq[0]}{sq[1]}: {prev_color or 'empty'} -> {curr_color or 'empty'}")

    def detect_opponent_move_uci(self, prev_map, sample_count: int = 3):
        """
        Detect opponent move by comparing previous board state to current.
        Returns UCI move string like 'e2e4' or None.
        """
        try:
            curr_map = self.get_board_piece_map(sample_count=sample_count)
            if not curr_map:
                self.logger.warning("Could not detect ANY pieces on the current frame.")
                return None

            self.logger.debug(f"Comparing: prev={len(prev_map)} pieces, curr={len(curr_map)} pieces")
            move = self.detect_move_from_maps(prev_map, curr_map)
            if not move:
                self.logger.warning("No move pattern detected")
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

    def debug_dump_board(self, filename="debug_pieces.jpg"):
        """
        Save an annotated debug image showing what the vision system detects
        on each square: piece presence and color classification.
        """
        if not self.board_location:
            print("Find board first.")
            return

        frame = self.capture_screen()
        bx, by, bw, bh = self.board_location
        sq_size = int(self.square_size)
        img = frame.copy()

        for rank_idx in range(8):
            for file_idx in range(8):
                x1 = bx + file_idx * sq_size
                y1 = by + rank_idx * sq_size
                x2 = x1 + sq_size
                y2 = y1 + sq_size

                cell_img = frame[y1:y2, x1:x2]
                has_piece = self._detect_piece_in_square(cell_img, rank_idx, file_idx)
                fl, rn = self._screen_to_square(rank_idx, file_idx)

                if has_piece:
                    color = self._classify_piece_color(cell_img, rank_idx, file_idx)
                    if color == 'w':
                        box_color = (255, 255, 255)
                    elif color == 'b':
                        box_color = (0, 0, 0)
                    else:
                        box_color = (0, 0, 255)  # red for unknown
                    cv2.rectangle(img, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), box_color, 3)
                    label = f"{fl}{rn}:{color}"
                else:
                    label = f"{fl}{rn}"

                cv2.putText(img, label, (x1 + 4, y1 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.imwrite(filename, img)
        self.logger.log(f"Saved debug image: {filename}")


if __name__ == "__main__":
    vision = GhostVision()
    print("Open a chess board in 3 seconds...")
    time.sleep(3)

    loc = vision.find_board()
    if loc:
        print(f"Board at: {loc}")
        vision.debug_draw_board()
        print("\nDetecting pieces...")
        piece_map = vision.get_board_piece_map(sample_count=3)
        print(f"Found {len(piece_map)} pieces:")
        for sq in sorted(piece_map.keys()):
            print(f"  {sq[0]}{sq[1]}: {piece_map[sq]}")
        vision.debug_dump_board()
    else:
        print("Failed to detect.")
