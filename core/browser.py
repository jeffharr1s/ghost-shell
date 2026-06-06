"""
GhostBrowser - launches Chrome as an INDEPENDENT process with remote debugging, then
connects to it over CDP. Moves are injected into the page via page.mouse, so the real
hardware cursor never moves and the terminal never loses focus.

Because Chrome is launched detached (not owned by Playwright), it OUTLIVES the bot:
closing Ghost-Shell only disconnects - you close Chrome yourself. On the next run the
bot reconnects to the still-open Chrome (same session, no re-login, no profile lock).

Reading the board is still done by the screen-vision pipeline (vision.py). This module
only needs the board element's bounding box in *viewport* coordinates (a geometry-only
DOM query) to know where to click - it does NOT read piece positions from the DOM.
"""
import os
import time
import random
import subprocess
import urllib.request

from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext

from utils.logger import Logger

# Candidate selectors for the live board element, most-specific first.
# chess.com renders the board as a <wc-chess-board> web component; the others
# are fallbacks in case of layout/version differences.
BOARD_SELECTORS = [
    "wc-chess-board",
    "chess-board",
    "#board-single",
    "#board-layout-chessboard .board",
    ".board",
]


class GhostBrowser:
    def __init__(self, profile_dir=None, start_url="https://www.chess.com", channel="chrome",
                 cdp_port=9222, chrome_path=""):
        self.logger = Logger("BROWSER")
        self.start_url = start_url
        # channel="chrome" finds your real Google Chrome (full feature set, so chess.com's
        # in-browser bot engines can load). "msedge" finds Edge. "" / "chromium" uses
        # Playwright's bundled Chromium (online play works; chess.com bots won't load).
        self.channel = channel
        self.cdp_port = cdp_port
        self.chrome_path = chrome_path  # optional explicit path to the browser .exe
        # Persist login/cookies between runs so you don't re-auth every time.
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.profile_dir = profile_dir or os.path.join(base, "browser_profile")

        self._pw = None
        self._browser: "Browser | None" = None
        self.context: "BrowserContext | None" = None
        self.page: "Page | None" = None
        self._board_selector: "str | None" = None

    def _require_page(self) -> Page:
        if self.page is None:
            raise RuntimeError("Browser not started - call start() first.")
        return self.page

    def _find_browser_executable(self, pw):
        """Path to the browser exe to launch. An explicit chrome_path wins; otherwise
        look up real Chrome/Edge by channel, finally falling back to bundled Chromium."""
        if self.chrome_path and os.path.exists(self.chrome_path):
            return self.chrome_path

        local = os.environ.get("LOCALAPPDATA", "")
        if self.channel == "msedge":
            candidates = [
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            ]
        elif self.channel in ("", None, "chromium"):
            candidates = []
        else:  # chrome
            candidates = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                os.path.join(local, r"Google\Chrome\Application\chrome.exe"),
            ]
        for path in candidates:
            if path and os.path.exists(path):
                return path

        self.logger.warning("Real Chrome not found - using Playwright's bundled Chromium "
                            "(chess.com bot engines won't load there, but online play works).")
        return pw.chromium.executable_path

    def _debugger_ready(self, port):
        """True if a Chrome DevTools endpoint is already answering on this port."""
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _wait_for_debugger(self, port, timeout_s=40):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._debugger_ready(port):
                return True
            time.sleep(0.5)
        return False

    def _launch_detached(self, exe, port):
        """Starts the browser as an independent process so it survives bot exit."""
        args = [
            exe,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={self.profile_dir}",
            "--start-maximized",
            "--no-first-run",
            "--no-default-browser-check",
            self.start_url,
        ]
        creationflags = 0
        if os.name == "nt":
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP - fully independent of the bot
            creationflags = 0x00000008 | 0x00000200
        subprocess.Popen(args, creationflags=creationflags, close_fds=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _maximize_window(self, context, page):
        """Maximizes the browser window. --start-maximized is unreliable with a
        persistent context, so we drive it directly via CDP after launch."""
        try:
            session = context.new_cdp_session(page)
            win = session.send("Browser.getWindowForTarget")
            session.send("Browser.setWindowBounds", {
                "windowId": win["windowId"],
                "bounds": {"windowState": "maximized"},
            })
        except Exception as e:
            self.logger.warning(f"Could not maximize window: {e}")

    # ----- lifecycle -----------------------------------------------------
    def start(self):
        """Connects to a debuggable browser - reusing one that's already open, or
        launching a fresh detached one - and opens chess.com. The browser is NOT owned
        by Playwright, so it keeps running after the bot exits."""
        self.logger.log("Connecting to browser...")
        pw = sync_playwright().start()
        self._pw = pw
        port = self.cdp_port

        if self._debugger_ready(port):
            self.logger.success(f"Reconnecting to the browser already open on port {port} "
                                f"(your session and game stay put).")
        else:
            exe = self._find_browser_executable(pw)
            self.logger.log(f"Launching browser (detached) on debug port {port}.")
            self._launch_detached(exe, port)
            if not self._wait_for_debugger(port, timeout_s=40):
                self.logger.error(f"Browser debug port {port} never came up.")
                raise RuntimeError("CDP endpoint not reachable")

        browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
        self._browser = browser
        context = browser.contexts[0] if browser.contexts else browser.new_context()
        self.context = context

        # Belt-and-suspenders: hide navigator.webdriver on future navigations.
        try:
            context.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")
        except Exception:
            pass

        # Prefer an existing chess.com tab; otherwise use/navigate the first tab.
        pages = context.pages
        page = next((p for p in pages if "chess.com" in (p.url or "")), None)
        if page is None:
            page = pages[0] if pages else context.new_page()
            try:
                page.goto(self.start_url, wait_until="domcontentloaded", timeout=60000)
            except Exception as e:
                self.logger.warning(f"Auto-load of {self.start_url} didn't finish ({e}). "
                                    f"Just type the address into the window yourself.")
        self.page = page
        self._maximize_window(context, page)
        self.logger.success("Browser ready. Log in and open a game in THIS window, then return to the terminal.")
        return self.page

    def close(self):
        """Disconnect only - the detached browser keeps running so you close it yourself.
        Idempotent: safe to call more than once (restart path + finally)."""
        try:
            if self._pw:
                self._pw.stop()  # drops our CDP connection; the detached browser lives on
        except Exception:
            pass
        self._browser = None
        self.context = None
        self.page = None
        self._pw = None

    # ----- board geometry ------------------------------------------------
    def wait_for_board(self, timeout_s=120):
        """Blocks until a board element is present, then remembers its selector."""
        page = self._require_page()
        self.logger.log("Waiting for a chess board to appear in the page...")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            for sel in BOARD_SELECTORS:
                try:
                    el = page.query_selector(sel)
                    if el and el.is_visible():
                        box = el.bounding_box()
                        # sanity: must be a sizeable, roughly-square region
                        if box and box["width"] > 200 and 0.8 < box["width"] / box["height"] < 1.2:
                            self._board_selector = sel
                            self.logger.success(f"Board found via selector '{sel}' "
                                                f"({int(box['width'])}x{int(box['height'])}).")
                            return True
                except Exception:
                    pass
            time.sleep(0.5)
        self.logger.error("No board element found in the page within timeout.")
        return False

    def get_board_box(self):
        """Returns the board's current bounding box in viewport CSS pixels."""
        if not self._board_selector:
            if not self.wait_for_board():
                return None
        page = self._require_page()
        el = page.query_selector(self._board_selector) if self._board_selector else None
        if not el:
            self.logger.error("Board element disappeared - re-scanning.")
            self._board_selector = None
            return self.get_board_box() if self.wait_for_board() else None
        return el.bounding_box()

    def _square_to_xy(self, square, is_white, box):
        """Maps an algebraic square ('e2') to a viewport (x, y) click point.
        Mirrors main.get_square_center, but in viewport space instead of screen space."""
        file_idx = ord(square[0]) - ord("a")
        rank_idx = int(square[1]) - 1
        sq = box["width"] / 8.0

        if is_white:
            x = box["x"] + file_idx * sq + sq / 2
            y = box["y"] + (7 - rank_idx) * sq + sq / 2
        else:
            # board is flipped 180 for black
            x = box["x"] + (7 - file_idx) * sq + sq / 2
            y = box["y"] + rank_idx * sq + sq / 2
        return x, y

    # ----- making moves --------------------------------------------------
    def _click_xy(self, x, y):
        """A single click injected via CDP - does NOT move the OS cursor."""
        page = self._require_page()
        page.mouse.move(x, y)
        page.mouse.down()
        time.sleep(random.uniform(0.04, 0.10))
        page.mouse.up()

    def make_move(self, uci, is_white):
        """Executes a UCI move ('e2e4', 'e7e8q') with click-source, click-target.
        Returns True on success. Never touches the hardware cursor."""
        box = self.get_board_box()
        if not box:
            self.logger.error("Cannot make move - no board box.")
            return False

        from_sq, to_sq = uci[:2], uci[2:4]
        promo = uci[4] if len(uci) > 4 else None

        fx, fy = self._square_to_xy(from_sq, is_white, box)
        tx, ty = self._square_to_xy(to_sq, is_white, box)

        self.logger.log(f"Clicking {from_sq} -> {to_sq}"
                        f"{(' =' + promo.upper()) if promo else ''}")
        self._click_xy(fx, fy)            # select the piece
        time.sleep(random.uniform(0.10, 0.25))
        self._click_xy(tx, ty)            # move it

        if promo:
            self._click_promotion(to_sq, promo, is_white, box)

        return True

    def _click_promotion(self, to_sq, piece, is_white, box):
        """Clicks the piece in chess.com's promotion popup.
        The popup stacks Q,R,B,N starting at the promotion square: it drops DOWN
        for White (promotion on rank 8) and UP for Black (rank 1)."""
        order = {"q": 0, "r": 1, "b": 2, "n": 3}
        idx = order.get(piece.lower())
        if idx is None:
            self.logger.error(f"Bad promotion piece: {piece}")
            return
        sq = box["width"] / 8.0
        x, y = self._square_to_xy(to_sq, is_white, box)
        # On screen the popup always grows downward from the top-most cell,
        # which is the promotion square in both orientations as drawn.
        promo_y = y + idx * sq if is_white else y - idx * sq
        time.sleep(random.uniform(0.15, 0.3))
        self.logger.log(f"Clicking promotion {piece.upper()}")
        self._click_xy(x, promo_y)
