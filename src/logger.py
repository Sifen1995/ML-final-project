"""
Logging configuration for Student Grade Prediction System.

Provides structured logging for all pipeline stages including
data loading, preprocessing, training, and prediction.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Union


def setup_logger(
    name: str = "student_grade_prediction",
    log_file: Optional[str] = None,
    log_level: Union[int, str] = logging.INFO,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure a logger for the student grade prediction system.
    
    Args:
        name: Logger name
        log_file: Optional file path for log output
        log_level: Logging level (default: INFO)
        log_format: Optional custom log format string
        
    Returns:
        Configured logger instance
    """
    # Handle string log levels
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if log_format is None:
        log_format = (
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "%(funcName)s:%(lineno)d | %(message)s"
        )
    
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "student_grade_prediction") -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set up default configuration
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


class PipelineLogger:
    """
    Specialized logger for ML pipeline stages.
    
    Provides structured logging methods for different stages
    of the student grade prediction pipeline.
    """
    
    def __init__(self, name: str = "student_grade_prediction"):
        """Initialize the pipeline logger."""
        self.logger = get_logger(name)
        self.start_times = {}
    
    def log_stage_start(self, stage: str) -> None:
        """Log the start of a pipeline stage."""
        self.start_times[stage] = datetime.now()
        self.logger.info(f"Starting stage: {stage}")
    
    def log_stage_end(self, stage: str, success: bool = True) -> None:
        """Log the end of a pipeline stage with duration."""
        status = "completed" if success else "failed"
        
        if stage in self.start_times:
            duration = datetime.now() - self.start_times[stage]
            self.logger.info(
                f"Stage {stage} {status} in {duration.total_seconds():.2f}s"
            )
        else:
            self.logger.info(f"Stage {stage} {status}")
    
    def log_data_info(
        self, 
        n_samples: int, 
        n_features: int,
        stage: str = "data"
    ) -> None:
        """Log dataset information."""
        self.logger.info(
            f"[{stage}] Dataset: {n_samples} samples, {n_features} features"
        )
    
    def log_model_info(
        self,
        model_name: str,
        params: Optional[dict] = None
    ) -> None:
        """Log model information."""
        msg = f"Model: {model_name}"
        if params:
            msg += f" | Params: {params}"
        self.logger.info(msg)
    
    def log_metrics(self, metrics: dict, stage: str = "evaluation") -> None:
        """Log evaluation metrics."""
        metrics_str = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        )
        self.logger.info(f"[{stage}] Metrics: {metrics_str}")
    
    def log_prediction(
        self,
        n_predictions: int,
        mean_grade: float,
        stage: str = "prediction"
    ) -> None:
        """Log prediction information."""
        self.logger.info(
            f"[{stage}] Made {n_predictions} predictions | "
            f"Mean predicted grade: {mean_grade:.2f}"
        )
    
    def log_error(self, error: Exception, stage: str = "pipeline") -> None:
        """Log an error with stage context."""
        self.logger.error(f"[{stage}] Error: {str(error)}", exc_info=True)
    
    def log_warning(self, message: str, stage: str = "pipeline") -> None:
        """Log a warning message."""
        self.logger.warning(f"[{stage}] {message}")
    
    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)


# Create a default logger instance
default_logger = get_logger()


def log_info(message: str) -> None:
    """Convenience function for info logging."""
    default_logger.info(message)


def log_error(message: str) -> None:
    """Convenience function for error logging."""
    default_logger.error(message)


def log_warning(message: str) -> None:
    """Convenience function for warning logging."""
    default_logger.warning(message)


def log_debug(message: str) -> None:
    """Convenience function for debug logging."""
    default_logger.debug(message)
