import logging
import os
from pathlib import Path
from datetime import datetime


def setup_logging(experiment_name: str = "experiment", outputs_dir: str = "outputs") -> logging.Logger:
    """
    Set up logging configuration to write to outputs/main.log
    
    Args:
        experiment_name: Name of the experiment for the logger
        outputs_dir: Directory where logs should be stored
    
    Returns:
        Configured logger instance
    """
    # Create outputs directory if it doesn't exist
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(exist_ok=True)
    
    # Create log file path
    log_file = outputs_path / "main.log"
    
    # Create logger
    logger = logging.getLogger("vlforge")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(levelname)-8s | %(message)s'
    )
    
    # File handler - logs to outputs/main.log
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - minimal output to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Log session start
    logger.info("="*80)
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info(f"SESSION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    return logger


def get_logger() -> logging.Logger:
    """Get the configured logger instance"""
    return logging.getLogger("vlforge") 