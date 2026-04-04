import logging
import sys
from pathlib import Path
from datetime import datetime


def _get_log_dir() -> Path:
    project_log_dir = Path(__file__).parent.parent / "logs"
    try:
        project_log_dir.mkdir(parents=True, exist_ok=True)
        # Verify we can actually write to it
        test_file = project_log_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        return project_log_dir
    except (PermissionError, OSError):
        fallback_path = Path.home() / ".skincare_ai" / "logs"
        fallback_path.mkdir(parents=True, exist_ok=True)
        print(
            f"Warning: Project 'logs' directory is not writable. Logging to: {fallback_path}"
        )
        return fallback_path


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = _get_log_dir()
    current_date = datetime.now().strftime("%Y-%m-%d")
    try:
        file_handler = logging.FileHandler(
            filename=log_dir / f"app_{current_date}.log", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (PermissionError, OSError) as e:
        logger.warning(f"Could not create log file: {e}. Logging to console only.")

    return logger


logger = setup_logger("skin_app")
