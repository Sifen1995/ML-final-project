"""
Student Grade Prediction System

A machine learning system for predicting student academic performance
using the UCI Student Performance dataset.

Modules:
    - config: Configuration settings and constants
    - data_loader: Data loading and validation utilities
    - feature_engineer: Feature engineering for student data
    - preprocessor: Data preprocessing and scaling
    - model_trainer: Model training and hyperparameter tuning
    - model_evaluator: Model evaluation and metrics
    - predictor: Grade prediction and performance classification
    - logger: Logging utilities
"""

__version__ = "1.0.0"
__author__ = "Student Grade Prediction Team"

from .config import (
    DataConfig, 
    ModelConfig, 
    StudentFeatures,
    data_config,
    model_config,
    logging_config,
    api_config
)
from .data_loader import DataLoader, load_train_test_split
from .feature_engineer import StudentFeatureEngineer, prepare_features
from .preprocessor import DataPreprocessor
from .model_trainer import ModelTrainer, train_all_models
from .model_evaluator import ModelEvaluator
from .predictor import GradePredictor, load_predictor
from .logger import setup_logger, get_logger, PipelineLogger

__all__ = [
    # Config
    "DataConfig",
    "ModelConfig", 
    "StudentFeatures",
    "data_config",
    "model_config",
    "logging_config",
    "api_config",
    # Data Loading
    "DataLoader",
    "load_train_test_split",
    # Feature Engineering
    "StudentFeatureEngineer",
    "prepare_features",
    # Preprocessing
    "DataPreprocessor",
    # Model Training
    "ModelTrainer",
    "train_all_models",
    # Evaluation
    "ModelEvaluator",
    # Prediction
    "GradePredictor",
    "load_predictor",
    # Logging
    "setup_logger",
    "get_logger",
    "PipelineLogger"
]
