import logging
import sys
from pathlib import Path
from datetime import datetime

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Centralized Logger Configuration
def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Date-Stamped File Handler
    # This approach is Windows-safe because it doesn't require renaming an active file.
    # It creates a new file for each day the application is run.
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_handler = logging.FileHandler(
        filename=log_dir / f"app_{current_date}.log",
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
    return logger

logger = setup_logger("skin_app")
