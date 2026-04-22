"""
Unified logging system.

- All modules call get_logger(__name__) — returns a standard logging.Logger.
- Every log goes to the terminal via the console handler (same as before).
- When a Celery task is running, set_task_id(task_id) is called once.
  From that point, every log produced anywhere in that worker process is also
  written in real-time to logs/YYYY-MM-DD/{task_id}.log.
- clear_task_id() is called in the finally block to stop file writing.
- Multiple concurrent tasks run in separate Celery worker processes, so their
  ContextVars are completely isolated — no cross-contamination.
- To disable file logging entirely: comment out the TaskFileHandler.install() line.
"""

import copy
import logging
import logging.handlers
import os
import sys
from contextvars import ContextVar
from datetime import date
from typing import Optional

# ---------------------------------------------------------------------------
# ContextVar — holds the active task ID for the current worker process/context
# ---------------------------------------------------------------------------
_task_id_var: ContextVar[Optional[str]] = ContextVar("task_id", default=None)

# Open file handles keyed by task_id so we don't re-open on every log record
_file_handles: dict = {}


def set_task_id(task_id: str) -> None:
    """Call at the start of a Celery task's run_task(). Opens the log file."""
    TaskFileHandler.install()  # Re-attach if Celery wiped root logger handlers
    _task_id_var.set(task_id)
    if task_id not in _file_handles:
        today = date.today().strftime("%Y-%m-%d")
        log_dir = os.path.join("logs", today)
        os.makedirs(log_dir, exist_ok=True)
        _file_handles[task_id] = open(
            os.path.join(log_dir, f"{task_id}.log"), "a", encoding="utf-8"
        )


def clear_task_id(task_id: str) -> None:
    """Call in the finally block of a Celery task. Flushes and closes the file."""
    _task_id_var.set(None)
    handle = _file_handles.pop(task_id, None)
    if handle:
        handle.flush()
        handle.close()


# ---------------------------------------------------------------------------
# Custom handler — writes to the active task's log file
# ---------------------------------------------------------------------------
_FILE_FORMAT = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class TaskFileHandler(logging.Handler):
    """
    Reads the current task_id from ContextVar on every emit.
    Writes to that task's open file handle.
    No-ops if no task is active.
    """

    def emit(self, record: logging.LogRecord) -> None:
        task_id = _task_id_var.get()
        if task_id is None:
            return
        handle = _file_handles.get(task_id)
        if handle is None:
            return
        try:
            handle.write(_FILE_FORMAT.format(record) + "\n")
            handle.flush()
        except Exception:
            self.handleError(record)

    @staticmethod
    def install() -> None:
        """Attach to the root logger once at startup. Re-run after Celery resets handlers."""
        root = logging.getLogger()

        # Re-attach colored console handler if Celery wiped it.
        # Use stderr — Celery only redirects stdout, so using stdout causes
        # colored ANSI output to be re-logged via celery.redirected into task files.
        if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
                   for h in root.handlers):
            console = logging.StreamHandler(sys.stderr)
            console.setLevel(root.level or logging.INFO)
            console.setFormatter(_CONSOLE_FORMAT)
            root.addHandler(console)

        # Silence celery.redirected — it re-logs anything written to stdout,
        # which would dump ANSI codes into task log files.
        logging.getLogger("celery.redirected").propagate = False

        if not any(isinstance(h, TaskFileHandler) for h in root.handlers):
            root.addHandler(TaskFileHandler())


# ---------------------------------------------------------------------------
# Console format — colored to match old Loguru terminal style
# ---------------------------------------------------------------------------
_ANSI_RESET = "\033[0m"
_LEVEL_COLORS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[1;31m", # bold red
}


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record = copy.copy(record)  # don't mutate the shared record object
        color = _LEVEL_COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname:<8}{_ANSI_RESET}"
        record.msg = f"{color}{record.getMessage()}{_ANSI_RESET}"
        record.args = None  # already interpolated above
        return super().format(record)


_CONSOLE_FORMAT = _ColorFormatter(
    "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Setup — called once when this module is imported
# ---------------------------------------------------------------------------
def _setup() -> None:
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)

    # Avoid adding duplicate handlers if module is re-imported
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
               for h in root.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(log_level)
        console.setFormatter(_CONSOLE_FORMAT)
        root.addHandler(console)

    # Optional: global app.log file (separate from per-task files)
    log_file = os.getenv("LOG_FILE", "logs/app.log")
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
            fh = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            fh.setLevel(log_level)
            fh.setFormatter(_CONSOLE_FORMAT)
            root.addHandler(fh)

    # Reduce noise from chatty third-party libraries
    for noisy in ("urllib3", "requests", "aiohttp", "openai", "google", "asyncio",
                  "httpcore", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("celery.worker").setLevel(logging.INFO)
    logging.getLogger("celery.task").setLevel(logging.INFO)
    logging.getLogger("celery.beat").setLevel(logging.WARNING)

    # Install the per-task file handler
    # To disable file logging entirely, comment out the next line:
    TaskFileHandler.install()


_setup()


# ---------------------------------------------------------------------------
# Public API — same signature as before, zero changes needed in callers
# ---------------------------------------------------------------------------
def get_logger(name: str = None) -> logging.Logger:
    """Drop-in replacement for the old Loguru get_logger."""
    return logging.getLogger(name or "app")


# Legacy alias used by app/utils/logger_factory.py
app_logger = logging.getLogger("app")
