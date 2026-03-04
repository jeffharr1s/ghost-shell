import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Version (Semantic Versioning)
APP_VERSION = "0.2.0"  # v0.2.0 - permutation yellow detection, filter stale highlights, fix timeout trap

# Engine Settings
ENGINE_DEPTH = int(os.getenv("ENGINE_DEPTH", 15))
ENGINE_CONTEMPT = int(os.getenv("ENGINE_CONTEMPT", 20))

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

# Player Side: "AUTO", "WHITE", or "BLACK"
PLAYER_SIDE = os.getenv("PLAYER_SIDE", "AUTO").upper()

# Legacy config (for backwards compatibility)
BOT_PERSONA = "AGGRESSIVE"
JITTER_AMOUNT = TARGET_JITTER