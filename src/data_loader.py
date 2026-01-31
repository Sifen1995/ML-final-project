import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

from src.logger import setup_logger
from src.config import data_config

logger = setup_logger(__name__, "data_loader.log")


class DataLoader:
    """Handles loading and basic validation of UCI Student Performance datasets."""
    
    def __init__(self, config=None):
        self.config = config or data_config
        
    def load_math_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """Load the Mathematics course student dataset."""
        path = file_path or self.config.student_math_path
        
        try:
            logger.info(f"Loading Mathematics student data from {path}")
            df = pd.read_csv(path, sep=';')
            logger.info(f"Successfully loaded {len(df)} student records (Mathematics)")
            
            # Basic validation
            self._validate_dataframe(df, "math")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Math student data file not found: {path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Math student data file is empty: {path}")
            raise
        except Exception as e:
            logger.error(f"Error loading math student data: {str(e)}")
            raise
    
    def load_portuguese_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """Load the Portuguese course student dataset."""
        path = file_path or self.config.student_por_path
        
        try:
            logger.info(f"Loading Portuguese student data from {path}")
            df = pd.read_csv(path, sep=';')
            logger.info(f"Successfully loaded {len(df)} student records (Portuguese)")
            
            # Basic validation
            self._validate_dataframe(df, "portuguese")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Portuguese student data file not found: {path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Portuguese student data file is empty: {path}")
            raise
        except Exception as e:
            logger.error(f"Error loading Portuguese student data: {str(e)}")
            raise
    
    def load_merged_data(self) -> pd.DataFrame:
        """
        Load and merge both datasets for students taking both courses.
        Uses matching on common demographic features.
        """
        try:
            df_math = self.load_math_data()
            df_por = self.load_portuguese_data()
            
            # Merge on common student attributes
            merge_cols = [
                "school", "sex", "age", "address", "famsize", "Pstatus",
                "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"
            ]
            
            df_merged = pd.merge(
                df_math, df_por,
                on=merge_cols,
                suffixes=('_math', '_por')
            )
            
            logger.info(f"Merged dataset: {len(df_merged)} students taking both courses")
            
            return df_merged
            
        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}")
            raise
    
    def save_processed_data(
        self,
        df: pd.DataFrame,
        dataset_type: str,
        file_path: Optional[Path] = None
    ) -> None:
        """Save processed data to CSV."""
        if dataset_type == "math":
            path = file_path or self.config.math_processed_path
        elif dataset_type == "portuguese":
            path = file_path or self.config.por_processed_path
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        try:
            logger.info(f"Saving processed {dataset_type} data to {path}")
            df.to_csv(path, index=False)
            logger.info(f"Successfully saved {len(df)} records")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def load_processed_data(self, dataset_type: str) -> pd.DataFrame:
        """Load processed data from CSV."""
        if dataset_type == "math":
            path = self.config.math_processed_path
        elif dataset_type == "portuguese":
            path = self.config.por_processed_path
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        try:
            logger.info(f"Loading processed {dataset_type} data from {path}")
            df = pd.read_csv(path)
            logger.info(f"Successfully loaded {len(df)} processed records")
            return df
            
        except FileNotFoundError:
            logger.error(f"Processed data file not found: {path}")
            raise
    
    def _validate_dataframe(self, df: pd.DataFrame, dataset_type: str) -> None:
        """Validate the student dataframe."""
        if df.empty:
            raise ValueError(f"{dataset_type} dataset is empty")
        
        # Check for target column (G3 - final grade)
        if self.config.target_column not in df.columns:
            logger.warning(
                f"Target column '{self.config.target_column}' not found in {dataset_type} data"
            )
        else:
            # Log grade distribution
            grade_stats = df[self.config.target_column].describe()
            logger.info(f"{dataset_type} grade statistics:\n{grade_stats}")
            
            # Check for pass/fail distribution
            passing = (df[self.config.target_column] >= self.config.passing_grade).sum()
            failing = (df[self.config.target_column] < self.config.passing_grade).sum()
            logger.info(f"{dataset_type} pass/fail: {passing} passing, {failing} failing")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values in {dataset_type}:\n{missing[missing > 0]}")
        
        # Validate expected columns for UCI Student Performance dataset
        expected_cols = [
            'school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
            'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
            'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
            'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
            'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
            'G1', 'G2', 'G3'
        ]
        
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing expected columns in {dataset_type}: {missing_cols}")


def load_train_test_split(
    dataset_type: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load processed data and split into train/test sets."""
    from sklearn.model_selection import train_test_split
    
    loader = DataLoader()
    df = loader.load_processed_data(dataset_type)
    
    # Separate features and target
    y = df[data_config.target_column]
    X = df.drop(columns=[data_config.target_column])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_config.test_size,
        random_state=data_config.random_state
    )
    
    logger.info(f"Split {dataset_type} data: train={len(X_train)}, test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test
