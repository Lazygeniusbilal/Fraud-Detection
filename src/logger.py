# Copies everything a script prints (stdout and stderr) into a log file as well,
# so each run leaves a record in logs/<name>_<timestamp>.log.

import os
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")


class _Tee:
    """Writes to the terminal stream and the log file at the same time."""

    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False


def start_logging(name: str) -> Path:
    """
        Starts copying all printed output to logs/<name>_<YYYYmmdd_HHMMSS>_<pid>.log
        and returns the log file path.
    """
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}.log"
    file = open(path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, file)
    sys.stderr = _Tee(sys.__stderr__, file)
    print(f"Logging to {path}")
    return path
