"""
utils/logger.py
───────────────
Structured logger using Rich for colourful console output.
"""

import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
)

logger = logging.getLogger("nl2sql")


def get_logger(name: str = "nl2sql") -> logging.Logger:
    return logging.getLogger(name)
