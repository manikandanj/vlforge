import logging
import os
from pathlib import Path
from datetime import datetime
from hydra.core.hydra_config import HydraConfig


def setup_logging(experiment_name: str = "experiment", 
                  outputs_dir: str = None, 
                  log_filename: str = None,
                  auto_filename: bool = False) -> logging.Logger:
    """
    Set up logging configuration that works with Hydra's automatic directory structure
    
    Args:
        experiment_name: Name of the experiment for the logger
        outputs_dir: Directory where logs should be stored (if None, uses Hydra's working dir)
        log_filename: Specific log filename (if None, uses "main.log")
        auto_filename: If True, generates filename based on experiment name and timestamp
    
    Returns:
        Configured logger instance
    """
    # Try to get Hydra's working directory first
    if outputs_dir is None:
        try:
            # Use Hydra's current working directory if available
            hydra_cfg = HydraConfig.get()
            outputs_path = Path(hydra_cfg.runtime.output_dir)
        except Exception:
            # Fallback to a default outputs directory if Hydra is not available
            outputs_path = Path("outputs")
    else:
        # Use the specified outputs directory
        outputs_path = Path(outputs_dir)
    
    # Create outputs directory if it doesn't exist
    outputs_path.mkdir(parents=True, exist_ok=True)
    
    # Determine log filename
    if auto_filename:
        # Generate filename based on experiment name and timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_exp_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in experiment_name)
        log_filename = f"{timestamp}_{safe_exp_name}.log"
    elif log_filename is None:
        log_filename = "main.log"
    
    # Create log file path
    log_file = outputs_path / log_filename
    
    # Get or create logger
    logger = logging.getLogger("vlforge")
    
    # Clear any existing handlers to prevent duplicates
    logger.handlers.clear()
    
    # Set logger level
    logger.setLevel(logging.INFO)
    
    # Prevent propagation to avoid duplicate messages from parent loggers
    logger.propagate = False
    
    # Create formatters
    file_formatter = logging.Formatter(
        '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )
    
    # File handler - logs to the determined log file path
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
    logger.info(f"LOG FILE: {log_file}")
    logger.info("="*80)
    
    return logger


def get_logger() -> logging.Logger:
    """Get the configured logger instance"""
    logger = logging.getLogger("vlforge")
    # If logger has no handlers, it hasn't been configured yet
    if not logger.handlers:
        # Create a basic logger configuration as fallback
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger 