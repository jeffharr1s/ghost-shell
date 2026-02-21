"""
Simple opening recognition using hardcoded opening patterns.
Matches the first 4-6 moves of the game against known openings.
"""

# Nested dictionary of openings by move sequence
# Format: {move1: {move2: {move3: {move4: "Opening Name"}}}}
# Moves in UCI notation (e.g., 'e2e4')

OPENINGS_DB = {
    # 1. e2e4 (Open Game)
    'e2e4': {
        # 1...e7e5 (Open Game)
        'e7e5': {
            # 2. Nf3 (Most common)
            'g1f3': {
                # 2...Nc6 (Italian/Spanish)
                'b8c6': {
                    # 3. Bb5 (Ruy Lopez/Spanish Opening)
                    'f1b5': 'Ruy Lopez (Spanish Opening)',
                    # 3. Bc4 (Italian Game)
                    'f1c4': 'Italian Game',
                },
                # 2...Nf6 (Scandinavia / Caro-Kann ideas)
                'g8f6': 'Petrov Defence',
            },
        },
        # 1...c7c5 (Sicilian Defence)
        'c7c5': {
            'g1f3': {
                'd7d6': 'Sicilian Defence (Najdorf/Classical)',
                'b8c6': 'Sicilian Defence (Open)',
            },
        },
        # 1...c7c6 (Caro-Kann Defence)
        'c7c6': 'Caro-Kann Defence',

        # 1...d7d5 (French or Scandinavian)
        'd7d5': {
            'e4d5': 'Scandinavian Defence',
            'd2d4': 'French Defence',
        },
    },

    # 1. d2d4 (Closed Games)
    'd2d4': {
        # 1...d7d5 (Queen's Gambit)
        'd7d5': {
            'c2c4': {
                'e7e6': 'Queen\'s Gambit Declined',
                'dxc4': 'Queen\'s Gambit Accepted',
            },
        },
        # 1...Nf6 (Indian Defences)
        'g8f6': {
            'c2c4': {
                'e7e6': 'Nimzo-Indian Defence',
                'g7g6': 'King\'s Indian Defence',
            },
        },
        # 1...c7c5 (Benoni)
        'c7c5': 'd2d4 c7c5 (Benoni)',
    },

    # 1. c2c4 (English Opening)
    'c2c4': {
        'e7e5': 'English Opening (Symmetric)',
        'c7c5': 'English Opening (with Sicilian structure)',
    },

    # 1. Nf3 (Reti Opening)
    'g1f3': {
        'd7d5': 'Reti Opening',
    },
}


def detect_opening(board) -> str:
    """
    Detect opening name from current board position.

    Args:
        board: python-chess Board object

    Returns:
        str: Opening name if matched, None otherwise
    """
    if len(board.move_stack) == 0:
        return None

    # Build the move sequence
    moves = [move.uci() for move in board.move_stack]

    # Try to match against openings database
    current = OPENINGS_DB
    matched_opening = None

    for move in moves[:6]:  # Check first 6 moves
        if move in current:
            current = current[move]
            # Check if this is a terminal string (opening name)
            if isinstance(current, str):
                matched_opening = current
                break
        else:
            # No further matches
            break

    return matched_opening


if __name__ == "__main__":
    # Test the opening detection
    import chess

    # Test 1: Ruy Lopez
    board = chess.Board()
    board.push_uci('e2e4')
    board.push_uci('e7e5')
    board.push_uci('g1f3')
    board.push_uci('b8c6')
    board.push_uci('f1b5')
    opening = detect_opening(board)
    print(f"Test 1 - Expected Ruy Lopez, got: {opening}")

    # Test 2: Sicilian
    board = chess.Board()
    board.push_uci('e2e4')
    board.push_uci('c7c5')
    board.push_uci('g1f3')
    opening = detect_opening(board)
    print(f"Test 2 - Expected Sicilian, got: {opening}")

    # Test 3: Queen's Gambit
    board = chess.Board()
    board.push_uci('d2d4')
    board.push_uci('d7d5')
    board.push_uci('c2c4')
    board.push_uci('e7e6')
    opening = detect_opening(board)
    print(f"Test 3 - Expected Queen's Gambit Declined, got: {opening}")
