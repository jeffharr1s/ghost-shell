import cv2
import mss
import numpy as np
import time
import chess
from utils.logger import Logger 

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
        
        # blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # look for large square-ish contours
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10000:  # too small
                continue
                
            # approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # looking for rectangles
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                
                # board should be roughly square
                if 0.8 < aspect_ratio < 1.2 and w > 200:
                    candidates.append((x, y, w, h, area))
        
        if not candidates:
            return None
        
        # pick the largest square candidate
        candidates.sort(key=lambda c: c[4], reverse=True)
        
        for x, y, w, h, _ in candidates[:5]:  # check top 5
            # verify its actually a chessboard by checking for alternating colors
            if self._verify_chessboard(frame, x, y, w, h):
                self.board_location = (x, y, w, h)
                self.square_size = w / 8
                self.logger.success(f"Found board at ({x}, {y}) size {w}x{h}")
                return self.board_location
        
        return None
    
    def _verify_chessboard(self, frame, x, y, w, h):
        """checks if region actually looks like a chessboard"""
        try:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                return False
                
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            sq_h = h // 8
            sq_w = w // 8
            
            # sample a few squares and check for alternating brightness
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
            
            # check if pattern roughly alternates
            light_squares = [s[0] for s in samples if s[1]]
            dark_squares = [s[0] for s in samples if not s[1]]
            
            if not light_squares or not dark_squares:
                return False
                
            avg_light = np.mean(light_squares)
            avg_dark = np.mean(dark_squares)
            
            # there should be a noticeable difference
            return abs(avg_light - avg_dark) > 20
            
        except Exception:
            return False

    def _find_board_by_corners(self):
        """original opencv corner method - works on empty boards"""
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
        center_left = bottom_left[h//4:3*h//4, w//4:3*w//4]
        center_right = bottom_right[h//4:3*h//4, w//4:3*w//4]
        
        gray_left = cv2.cvtColor(center_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(center_right, cv2.COLOR_BGR2GRAY)
        
        avg_left = np.mean(gray_left)
        avg_right = np.mean(gray_right)
        
        if avg_left > 120 or avg_right > 120:
            self.logger.success("Playing as WHITE")
            return chess.WHITE
        else:
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
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        sq = self.square_size
        for r in range(8):
            for c in range(8):
                sx = int(x + c * sq)
                sy = int(y + r * sq)
                cv2.rectangle(img, (sx, sy), (int(sx+sq), int(sy+sq)), (0, 0, 255), 1)

        cv2.imwrite("debug_vision.jpg", img)
        print("Saved debug_vision.jpg - check if grid aligns")

    def _detect_piece_in_square(self, cell_img, rank_idx, file_idx):
        """
        Advanced piece detection using multiple methods with DEBUG logging.
        Returns True if a piece is likely present.
        """
        if cell_img.size == 0:
            return False
        
        try:
            # Method 1: Edge detection
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.count_nonzero(edges) / edges.size
            
            # Method 2: Standard deviation (variance)
            std_dev = np.std(gray)
            
            # Method 3: Contrast analysis
            min_val = np.min(gray)
            max_val = np.max(gray)
            contrast = max_val - min_val
            
            # Thresholds (tune these based on your chess site)
            has_edges = edge_density > 0.04  # Pieces have edges
            has_variance = std_dev > 12  # Pieces have texture
            has_contrast = contrast > 30  # Pieces have depth/shading
            
            # Combine methods: need at least 2 of 3 indicators
            detection_score = sum([has_edges, has_variance, has_contrast])
            result = detection_score >= 2
            
            # DEBUG LOGGING - shows values for each square
            file = chess.FILE_NAMES[file_idx]
            rank = 8 - rank_idx
            self.logger.log(
                f"{file}{rank}: edge={edge_density:.3f}({'✓' if has_edges else '✗'}), "
                f"std={std_dev:.1f}({'✓' if has_variance else '✗'}), "
                f"contrast={contrast:.0f}({'✓' if has_contrast else '✗'}) "
                f"→ {'PIECE' if result else 'empty'}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Detection error at {file_idx},{rank_idx}: {e}")
            return False

    def get_board_piece_map(self, sample_count: int = 3, sample_delay: float = 0.05):
        """
        Capture the current board region and return a mapping
        {(file, rank): piece_code}.
        Uses advanced edge detection and variance analysis.
        """
        if not self.board_location:
            self.logger.error("Board not locked. Run find_board() first.")
            return {}
        
        sample_count = max(1, sample_count)
        self.logger.log(f"=== DETECTING PIECES ({sample_count} sample(s)) ===")

        bx, by, bw, bh = self.board_location
        sq_size = int(self.square_size)

        all_maps = []
        for idx in range(sample_count):
            frame = self.capture_screen()
            piece_map = {}

            # Sample each square and try to detect if there's a piece
            for rank_idx in range(8):  # 0..7 from top to bottom
                for file_idx in range(8):  # 0..7 from left to right
                    x1 = bx + file_idx * sq_size
                    y1 = by + rank_idx * sq_size
                    x2 = x1 + sq_size
                    y2 = y1 + sq_size
                    
                    cell_img = frame[y1:y2, x1:x2]
                    
                    # Use advanced piece detection
                    has_piece = self._detect_piece_in_square(cell_img, rank_idx, file_idx)
                    
                    if has_piece:
                        file = chess.FILE_NAMES[file_idx]
                        rank = 8 - rank_idx  # flip: rank 8 at top, 1 at bottom for white
                        piece_map[(file, rank)] = 'piece'

            all_maps.append(piece_map)
            if idx < sample_count - 1:
                time.sleep(sample_delay)

        # Majority vote across samples to reduce noise
        if sample_count == 1:
            final_map = all_maps[0]
        else:
            vote_counts = {}
            for piece_map in all_maps:
                for square in piece_map.keys():
                    vote_counts[square] = vote_counts.get(square, 0) + 1

            majority = (sample_count // 2) + 1
            final_map = {sq: 'piece' for sq, count in vote_counts.items() if count >= majority}

        self.logger.log(f"=== DETECTED {len(final_map)} PIECES AFTER VOTING ===")
        if not final_map:
            self.logger.warning("No pieces detected; check thresholds or board alignment.")

        return final_map

    def detect_move_from_maps(self, prev_map, curr_map):
        """
        Compare two piece maps and infer the move that was made.
        Returns tuple (from_square, to_square) like (('e', 2), ('e', 4))
        or None if no clear move detected.
        """
        # Find removed and added pieces
        removed = {}
        added = {}
        
        for sq, p in prev_map.items():
            if sq not in curr_map:
                removed[sq] = p
        
        for sq, p in curr_map.items():
            if sq not in prev_map:
                added[sq] = p
        
        # Simple case: one from, one to
        if len(removed) == 1 and len(added) == 1:
            from_sq = list(removed.keys())[0]
            to_sq = list(added.keys())[0]
            return (from_sq, to_sq)
        
        # Capture: one piece disappears, one moves to target
        if len(removed) == 2 and len(added) == 1:
            to_sq = list(added.keys())[0]
            # The from_sq is whichever removed square isn't the to_sq
            for sq in removed.keys():
                if sq != to_sq:
                    return (sq, to_sq)
        
        return None

    @staticmethod
    def square_to_uci(square):
        """Convert (file, rank) tuple to UCI string like 'e2'"""
        file, rank = square
        return f"{file}{rank}"

    def detect_opponent_move_uci(self, prev_map, sample_count: int = 3):
        """
        Detect opponent's move by comparing previous board state to current.
        Returns UCI move string like 'e2e4' or None if no move detected.
        """
        curr_map = self.get_board_piece_map(sample_count=sample_count)
        if not curr_map:
            self.logger.error("Could not detect pieces on the current frame.")
            return None

        move = self.detect_move_from_maps(prev_map, curr_map)
        
        if not move:
            return None
        
        from_sq, to_sq = move
        return self.square_to_uci(from_sq) + self.square_to_uci(to_sq)


if __name__ == "__main__":
    vision = GhostVision()
    print("Open a chess board in 3 seconds...")
    time.sleep(3)
    
    loc = vision.find_board()
    if loc:
        print(f"Board at: {loc}")
        vision.debug_draw_board()
    else:
        print("Failed to detect. Run debug to see what it sees.")
