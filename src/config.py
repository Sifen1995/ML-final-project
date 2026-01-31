from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import os


# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # File paths for UCI Student Performance datasets
    student_math_path: Path = RAW_DATA_DIR / "student-mat.csv"
    student_por_path: Path = RAW_DATA_DIR / "student-por.csv"
    
    # Processed data paths
    math_processed_path: Path = PROCESSED_DATA_DIR / "student_math_processed.csv"
    por_processed_path: Path = PROCESSED_DATA_DIR / "student_por_processed.csv"
    
    # Feature columns to drop (intermediate grades - optional to include)
    math_drop_columns: List[str] = field(default_factory=lambda: [])
    por_drop_columns: List[str] = field(default_factory=lambda: [])
    
    # Target column (Final Grade)
    target_column: str = "G3"
    
    # Binary classification threshold (for pass/fail classification)
    passing_grade: int = 10  # Grades >= 10 are considered passing
    
    # Test split ratio
    test_size: float = 0.2
    random_state: int = 42
    
    # Feature categories for the UCI Student Performance dataset
    demographic_features: List[str] = field(default_factory=lambda: [
        "school", "sex", "age", "address", "famsize", "Pstatus"
    ])
    
    family_features: List[str] = field(default_factory=lambda: [
        "Medu", "Fedu", "Mjob", "Fjob", "guardian"
    ])
    
    academic_features: List[str] = field(default_factory=lambda: [
        "traveltime", "studytime", "failures", "schoolsup", 
        "famsup", "paid", "activities", "nursery", "higher", "internet"
    ])
    
    social_features: List[str] = field(default_factory=lambda: [
        "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"
    ])


@dataclass
class ModelConfig:
    """Configuration for model training."""
    
    # Model types
    model_types: List[str] = field(default_factory=lambda: [
        "linear_regression", "random_forest", "xgboost"
    ])
    
    # Random Forest parameters (for regression)
    rf_params: Dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 3,
        "random_state": 42,
        "n_jobs": -1
    })
    
    # XGBoost parameters (for regression)
    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "objective": "reg:squarederror"
    })
    
    # Linear Regression parameters
    lr_params: Dict = field(default_factory=lambda: {
        "fit_intercept": True,
        "n_jobs": -1
    })
    
    # Ridge Regression parameters
    ridge_params: Dict = field(default_factory=lambda: {
        "alpha": 1.0,
        "random_state": 42
    })
    
    # Model save paths
    model_save_dir: Path = MODELS_DIR
    
    # Cross-validation
    cv_folds: int = 5


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    log_dir: Path = LOGS_DIR
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True
    log_to_console: bool = True


@dataclass
class APIConfig:
    """Configuration for API service."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    title: str = "Student Grade Prediction API"
    description: str = "API for predicting student academic performance based on demographic, social, and academic features"
    version: str = "1.0.0"


class StudentFeatures:
    """Feature groupings for student performance prediction."""
    
    DEMOGRAPHIC = [
        "school", "sex", "age", "address", "famsize", "Pstatus"
    ]
    
    FAMILY = [
        "Medu", "Fedu", "Mjob", "Fjob", "guardian"
    ]
    
    ACADEMIC = [
        "traveltime", "studytime", "failures", "schoolsup", 
        "famsup", "paid", "activities", "nursery", "higher", "internet"
    ]
    
    SOCIAL = [
        "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"
    ]
    
    GRADES = ["G1", "G2", "G3"]
    
    @classmethod
    def get_all_features(cls) -> list:
        """Get all feature names."""
        return cls.DEMOGRAPHIC + cls.FAMILY + cls.ACADEMIC + cls.SOCIAL + cls.GRADES
    
    @classmethod
    def get_categorical_features(cls) -> list:
        """Get categorical feature names."""
        return [
            "school", "sex", "address", "famsize", "Pstatus",
            "Mjob", "Fjob", "guardian", "schoolsup", "famsup",
            "paid", "activities", "nursery", "higher", "internet", "romantic"
        ]
    
    @classmethod
    def get_numeric_features(cls) -> list:
        """Get numeric feature names."""
        return [
            "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
            "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
            "G1", "G2"
        ]


# Create config instances
data_config = DataConfig()
model_config = ModelConfig()
logging_config = LoggingConfig()
api_config = APIConfig()


# Aliases for backwards compatibility
DataConfig.STUDENT_DATA_PATH = DATA_DIR / "student_mat.csv"
DataConfig.TARGET_COLUMN = "G3"
DataConfig.TEST_SIZE = 0.2
DataConfig.RANDOM_STATE = 42
ModelConfig.MODELS = {
    'linear_regression': 'LinearRegression',
    'ridge': 'Ridge',
    'random_forest': 'RandomForestRegressor',
    'xgboost': 'XGBRegressor'
}
ModelConfig.CV_FOLDS = 5
