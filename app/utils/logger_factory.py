"""
Logger factory — thin wrapper around the unified logging system in app.logging_config.
All callers use get_logger(__name__) which returns a standard logging.Logger.
"""
import logging
from app.logging_config import get_logger, set_task_id, clear_task_id

__all__ = ["get_logger", "set_task_id", "clear_task_id"]
