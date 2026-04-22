"""
Centralized logging configuration for the insurance AI service.
Provides consistent logging setup, formatters, and utilities across all modules.
"""
import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from app.config import ENV_PROJECT


class LoggingConfig:
    """Centralized logging configuration manager."""
    
    # Log levels mapping
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    # Default log format
    DEFAULT_FORMAT = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
    )
    
    # Simplified format for console output
    CONSOLE_FORMAT = (
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    
    # JSON format for structured logging (if needed)
    JSON_FORMAT = (
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
        '"logger": "%(name)s", "file": "%(filename)s", "line": %(lineno)d, '
        '"function": "%(funcName)s", "message": "%(message)s"}'
    )
    
    @classmethod
    def get_logging_config(
        cls, 
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        enable_console: bool = True,
        enable_file: bool = True,
        json_format: bool = False
    ) -> Dict[str, Any]:
        """
        Generate logging configuration dictionary.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            enable_console: Enable console logging
            enable_file: Enable file logging
            json_format: Use JSON format for structured logging
            
        Returns:
            Logging configuration dictionary
        """
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'detailed': {
                    'format': cls.DEFAULT_FORMAT,
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'console': {
                    'format': cls.CONSOLE_FORMAT,
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'json': {
                    'format': cls.JSON_FORMAT,
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
            'handlers': {},
            'loggers': {
                'app': {
                    'level': log_level,
                    'handlers': [],
                    'propagate': False
                },
                'uvicorn': {
                    'level': 'INFO',
                    'handlers': [],
                    'propagate': False
                },
                'celery': {
                    'level': 'INFO',
                    'handlers': [],
                    'propagate': False
                }
            },
            'root': {
                'level': log_level,
                'handlers': []
            }
        }
        
        handlers = []
        
        # Console handler
        if enable_console:
            config['handlers']['console'] = {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'json' if json_format else 'console',
                'stream': 'ext://sys.stdout'
            }
            handlers.append('console')
        
        # File handler
        if enable_file and log_file:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            config['handlers']['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': log_level,
                'formatter': 'json' if json_format else 'detailed',
                'filename': log_file,
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf8'
            }
            handlers.append('file')
        
        # Error file handler (always uses detailed format)
        if enable_file and log_file:
            error_log_file = str(log_path.parent / f"{log_path.stem}_errors{log_path.suffix}")
            config['handlers']['error_file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': error_log_file,
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf8'
            }
            handlers.append('error_file')
        
        # Assign handlers to loggers
        for logger_name in config['loggers']:
            config['loggers'][logger_name]['handlers'] = handlers
        config['root']['handlers'] = handlers
        
        return config
    
    @classmethod
    def setup_logging(
        cls,
        log_level: str = "INFO",
        log_file: Optional[str] = "logs/insurance_ai.log",
        enable_console: bool = True,
        enable_file: bool = True,
        json_format: bool = False
    ) -> None:
        """
        Setup centralized logging configuration.
        
        Args:
            log_level: Logging level
            log_file: Path to log file
            enable_console: Enable console logging
            enable_file: Enable file logging
            json_format: Use JSON format for structured logging
        """
        config = cls.get_logging_config(
            log_level=log_level,
            log_file=log_file,
            enable_console=enable_console,
            enable_file=enable_file,
            json_format=json_format
        )
        
        logging.config.dictConfig(config)
        
        # Log the configuration setup
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured - Level: {log_level}, File: {log_file}")
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger instance with consistent configuration.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)
    
    @classmethod
    def configure_third_party_loggers(cls) -> None:
        """Configure third-party library loggers to reduce noise."""
        # Reduce noise from third-party libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('aiohttp').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('google').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        
        # Keep important Celery logs but reduce verbosity
        logging.getLogger('celery.worker').setLevel(logging.INFO)
        logging.getLogger('celery.task').setLevel(logging.INFO)
        logging.getLogger('celery.beat').setLevel(logging.WARNING)


# Convenience functions for common logging patterns
def log_function_entry(logger: logging.Logger, func_name: str, **kwargs) -> None:
    """Log function entry with parameters."""
    if kwargs:
        logger.debug(f"Entering {func_name} with parameters: {kwargs}")
    else:
        logger.debug(f"Entering {func_name}")


def log_function_exit(logger: logging.Logger, func_name: str, result: Any = None) -> None:
    """Log function exit with optional result."""
    if result is not None:
        logger.debug(f"Exiting {func_name} with result type: {type(result).__name__}")
    else:
        logger.debug(f"Exiting {func_name}")


def log_performance(logger: logging.Logger, operation: str, duration: float) -> None:
    """Log performance metrics."""
    logger.info(f"Performance - {operation}: {duration:.3f}s")


def log_business_event(logger: logging.Logger, event: str, details: Dict[str, Any] = None) -> None:
    """Log important business events."""
    if details:
        logger.info(f"Business Event - {event}: {details}")
    else:
        logger.info(f"Business Event - {event}")


def log_security_event(logger: logging.Logger, event: str, details: Dict[str, Any] = None) -> None:
    """Log security-related events."""
    if details:
        logger.warning(f"Security Event - {event}: {details}")
    else:
        logger.warning(f"Security Event - {event}")


# Global logging configuration instance
logging_config = LoggingConfig()
