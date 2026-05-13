import datetime
import os
import re
import threading
import traceback

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class Logger:
    _lock = threading.Lock()
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _log_path = os.path.join(_repo_root, "COPY_THIS_LOG.txt")
    _ansi_pattern = re.compile(r"\033\[[0-9;]*m")
    _initialized = False

    def __init__(self, module_name="CORE"):
        self.module_name = module_name.upper()
        self._initialize_log_file()

    @classmethod
    def _initialize_log_file(cls):
        with cls._lock:
            if cls._initialized:
                return
            header = [
                "Ghost-Shell copy/paste log",
                f"Run started: {datetime.datetime.now().isoformat(timespec='seconds')}",
                f"Log file: {cls._log_path}",
                "-" * 72,
            ]
            with open(cls._log_path, "w", encoding="utf-8") as log_file:
                log_file.write("\n".join(header) + "\n")
            cls._initialized = True

    @classmethod
    def current_log_path(cls):
        cls._initialize_log_file()
        return cls._log_path

    def _get_timestamp(self):
        return datetime.datetime.now().strftime("%H:%M:%S")

    def _write_file(self, line):
        clean_line = self._ansi_pattern.sub("", line)
        self.current_log_path()
        with self._lock:
            with open(self._log_path, "a", encoding="utf-8") as log_file:
                log_file.write(clean_line + "\n")

    def _emit(self, color, message):
        t = self._get_timestamp()
        line = f"[{t}] [{self.module_name}] {message}"
        print(f"{color}{line}{Colors.ENDC}")
        self._write_file(line)

    def log(self, message):
        self._emit(Colors.CYAN, message)

    def success(self, message):
        self._emit(Colors.GREEN, message)

    def warning(self, message):
        self._emit(Colors.WARNING, message)

    def error(self, message):
        self._emit(Colors.FAIL, message)

    def debug(self, message):
        self._emit(Colors.BLUE, message)

    def exception(self, message, exc=None):
        self.error(message)
        if exc is None:
            trace = traceback.format_exc()
        else:
            trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

        for line in trace.rstrip().splitlines():
            self._write_file(f"[{self._get_timestamp()}] [{self.module_name}] {line}")

if __name__ == "__main__":
    log = Logger("TEST")
    log.log("Starting up...")
    log.success("Connected.")
    log.warning("Slow response.")
    log.error("Something broke.")
    log.debug("Debug info.")
