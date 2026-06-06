import os
import sys
import random
from stockfish import Stockfish
from utils.logger import Logger
from utils.config import (
    ENGINE_DEPTH,
    ENGINE_CONTEMPT,
    MISTAKE_CHANCE,
    MISTAKE_MAX_DROP,
    MISTAKE_FLOOR,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class GhostEngine:
    def __init__(self):
        self.logger = Logger("ENGINE")
        
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        engine_path = os.path.join(base_path, "assets", "stockfish.exe")
        
        if not os.path.exists(engine_path):
            self.logger.error(f"Stockfish not found at: {engine_path}")
            sys.exit(1)

        self.stockfish = Stockfish(path=engine_path)
        self.stockfish.set_depth(ENGINE_DEPTH) 
        self.stockfish.update_engine_parameters({"Contempt": ENGINE_CONTEMPT}) 

        self.logger.success(f"Brain loaded. Depth: {ENGINE_DEPTH}, Contempt: {ENGINE_CONTEMPT}")

    def get_human_move(self, fen) -> "str | None":
        """Usually returns Stockfish's best move, but every now and then plays a small
        deliberate inaccuracy to look human - bounded so it never blunders into a loss."""
        try:
            self.stockfish.set_fen_position(fen)
            top_moves = self.stockfish.get_top_moves(5)
        except Exception as e:
            self.logger.error(f"Stockfish error: {e}")
            return None

        if not top_moves:
            return None

        best = top_moves[0]

        # Never gamble when a forced mate is on the board - just play the best move,
        # whether we're delivering mate or avoiding getting mated.
        if best.get('Mate') is not None:
            self.logger.log(f"Best move (mate line): {best['Move']}")
            return str(best['Move'])

        best_cp = int(best.get('Centipawn') or 0)

        # Roll for a human mistake: pick a slightly-worse move, but only one that stays
        # within MISTAKE_MAX_DROP of best AND keeps the eval at/above MISTAKE_FLOOR.
        if MISTAKE_CHANCE > 0 and len(top_moves) > 1 and random.random() < MISTAKE_CHANCE:
            safe = []
            for m in top_moves[1:]:
                if m.get('Mate') is not None:
                    continue  # never touch mate-related alternatives
                cp = int(m.get('Centipawn') or 0)
                drop = best_cp - cp
                if 0 < drop <= MISTAKE_MAX_DROP and cp >= MISTAKE_FLOOR:
                    safe.append((m, drop, cp))
            if safe:
                choice, drop, cp = random.choice(safe)
                self.logger.log(f"Human mistake: {choice['Move']} (-{drop}cp from best, eval {cp}cp)")
                return str(choice['Move'])

        self.logger.log(f"Best move: {best['Move']}")
        return str(best['Move'])

if __name__ == "__main__":
    brain = GhostEngine()
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print("Thinking...")
    move = brain.get_human_move(start_fen)
    print(f"Move: {move}")