from loguru import logger
from pathlib import Path
import sys


def _get_log_dir() -> Path:
    project_log_dir = Path(__file__).parent.parent / "logs"
    try:
        project_log_dir.mkdir(parents=True, exist_ok=True)
        return project_log_dir
    except (PermissionError, OSError):
        fallback_path = Path.home() / ".skincare_ai" / "logs"
        fallback_path.mkdir(parents=True, exist_ok=True)
        return fallback_path


# Remove default handler
logger.remove()

# Console: Only show WARNING and above (less noise)
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="WARNING",  # Only warnings and errors in console
    colorize=True,
)

# File: Log everything with details
log_dir = _get_log_dir()
logger.add(
    log_dir / "app_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="INFO",  # Log everything to file
)
