import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Version (Semantic Versioning)
APP_VERSION = "0.6.0"  # v0.6.0 - CDP browser mode (detached Chrome, clicks via DevTools, no hardware mouse), patient yellow scan, colored log states, bounded human mistakes


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")

# Engine Settings
ENGINE_DEPTH = int(os.getenv("ENGINE_DEPTH", 15))
ENGINE_CONTEMPT = int(os.getenv("ENGINE_CONTEMPT", 20))

# Human-like move selection: every now and then play a small inaccuracy instead of
# the engine's best move, so it doesn't look like a perfect machine - but never a
# blunder that loses. Tune to taste:
#   MISTAKE_CHANCE   - probability per move of considering a mistake (0 = always best)
#   MISTAKE_MAX_DROP - how much worse than best the move may be, in centipawns (100 = 1 pawn)
#   MISTAKE_FLOOR    - never pick a move whose resulting eval is below this (cp); keeps
#                      you off a losing position, and means when you're already losing it
#                      just plays the best move.
MISTAKE_CHANCE = float(os.getenv("MISTAKE_CHANCE", 0.15))
MISTAKE_MAX_DROP = int(os.getenv("MISTAKE_MAX_DROP", 110))
MISTAKE_FLOOR = int(os.getenv("MISTAKE_FLOOR", -100))

# Humanization  
MOUSE_SPEED = float(os.getenv("MOUSE_SPEED", 0.4))
TARGET_JITTER = int(os.getenv("TARGET_JITTER", 6))

# Thinking Time (simulates human deliberation)
THINK_TIME_MIN = float(os.getenv("THINK_TIME_MIN", 1.5))
THINK_TIME_MAX = float(os.getenv("THINK_TIME_MAX", 6.0))

# Game mode timing presets
GAME_MODE_PRESETS = {
    "BLITZ":   {"think_min": 0.2, "think_max": 0.8,  "poll": 0.25, "settle": 0.4, "yellow_wait": 0.4, "yellow_retry": 0.4},
    "RAPID":   {"think_min": 1.0, "think_max": 3.5,  "poll": 0.5,  "settle": 0.7, "yellow_wait": 0.7, "yellow_retry": 0.8},
    "CLASSIC": {"think_min": 2.0, "think_max": 7.0,  "poll": 0.8,  "settle": 1.0, "yellow_wait": 1.0, "yellow_retry": 1.2},
}

# Startup defaults. QUICK_START skips the FEN/new-game question and uses these values.
DEFAULT_GAME_MODE = os.getenv("DEFAULT_GAME_MODE", "RAPID").upper()
if DEFAULT_GAME_MODE not in GAME_MODE_PRESETS:
    DEFAULT_GAME_MODE = "RAPID"

QUICK_START = _env_bool("QUICK_START", False)
DEFAULT_START_MODE = os.getenv("DEFAULT_START_MODE", "N").upper()
if DEFAULT_START_MODE not in ("N", "F"):
    DEFAULT_START_MODE = "N"

PAUSE_ON_EXIT = _env_bool("PAUSE_ON_EXIT", True)

# Player Side: "AUTO", "WHITE", or "BLACK"
PLAYER_SIDE = os.getenv("PLAYER_SIDE", "AUTO").upper()

# Move execution: when USE_BROWSER is true, moves are injected into a
# Playwright-launched browser via CDP (no hardware cursor, terminal keeps focus).
# When false, the legacy pyautogui path drives the real mouse. Reading the board
# is done by screen-vision in both modes.
USE_BROWSER = _env_bool("USE_BROWSER", False)
CHESS_URL = os.getenv("CHESS_URL", "https://www.chess.com")
# Browser to launch in browser mode. "chrome" uses your real Google Chrome, which
# is required for chess.com's in-browser WASM bot engines to load. "msedge" also
# works. Leave blank to use Playwright's bundled Chromium (bots won't load there).
BROWSER_CHANNEL = os.getenv("BROWSER_CHANNEL", "chrome")
# Chrome is launched DETACHED with remote debugging on this port and the bot connects
# over CDP, so Chrome outlives the bot - you close it yourself, and restarts reconnect.
CDP_PORT = int(os.getenv("CDP_PORT", 9222))
# Optional explicit path to the browser .exe (leave blank to auto-detect Chrome/Edge).
CHROME_PATH = os.getenv("CHROME_PATH", "")

# How many times to scan for the opponent's move before falling back to asking you.
# ~30 (default) gives the opponent plenty of time to move (~25s+ on RAPID) before it
# ever turns red and asks. 0 = wait indefinitely. Pressing 'M' always jumps to manual.
YELLOW_MAX_ATTEMPTS = int(os.getenv("YELLOW_MAX_ATTEMPTS", 30))

# Legacy config (for backwards compatibility)
BOT_PERSONA = "AGGRESSIVE"
JITTER_AMOUNT = TARGET_JITTER
