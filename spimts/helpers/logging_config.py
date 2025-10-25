# spimts/helpers/logging_config.py
"""
Unified logging configuration for compute.py and visualize.py

Usage:
    from spimts.helpers.logging_config import setup_logging
    
    # In main():
    logger = setup_logging(
        script_name='visualize',  # or 'compute'
        profile='dev++',
        log_file=args.log_file  # None, 'auto', or explicit path
    )
    
    logger.info("Starting visualization...")
    logger.warning("Found NaN values in MPI matrix")
    logger.error("Failed to load artifact")
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to terminal output (but not file output)"""
    
    # ANSI color codes (compatible with Windows 10+ PowerShell)
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, fmt=None, use_color=True):
        super().__init__(fmt)
        self.use_color = use_color
    
    def format(self, record):
        if self.use_color and record.levelname in self.COLORS:
            # Color only the level name, not the entire message
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(script_name: str, profile: str, log_file: str | None = None) -> logging.Logger:
    """Setup logging with dual output (terminal + optional file).
    
    Args:
        script_name: 'visualize' or 'compute'
        profile: dev/dev+/dev++/paper
        log_file: None (terminal only), 'auto' (auto-named in ./logs/), or explicit path
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # Remove any existing handlers
    
    # Format: [HH:MM:SS] LEVEL: message
    # (Timestamps in file, levels in terminal)
    console_format = '%(levelname)-8s %(message)s'
    file_format = '[%(asctime)s] %(levelname)-8s %(message)s'
    date_format = '%H:%M:%S'
    
    # 1) Console handler (ALWAYS active, colored output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredFormatter(console_format, use_color=True))
    logger.addHandler(console_handler)
    
    # 2) File handler (OPTIONAL, based on --log-file argument)
    if log_file:
        if log_file.lower() == 'auto':
            # Auto-generate filename: logs/<script>_<profile>_YYYYMMDD-HHMMSS.log
            logs_dir = Path('./logs')
            logs_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            log_file = logs_dir / f"{script_name}_{profile}_{timestamp}.log"
        
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # File gets ALL messages (including DEBUG)
        file_handler.setFormatter(logging.Formatter(file_format, datefmt=date_format))
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_path.absolute()}")
    
    return logger


def log_progress_bar(logger, current: int, total: int, prefix: str = '', width: int = 30):
    """Log a simple ASCII progress bar (no special characters).
    
    Args:
        logger: Logger instance
        current: Current item number (1-indexed)
        total: Total number of items
        prefix: Text to show before the progress bar
        width: Width of the progress bar in characters
    
    Example:
        [VIZ] Processing models [=======>          ] 7/16 (44%)
    """
    filled = int(width * current / total)
    bar = '=' * filled + '>' + ' ' * (width - filled - 1)
    pct = int(100 * current / total)
    logger.info(f"{prefix} [{bar}] {current}/{total} ({pct}%)")
