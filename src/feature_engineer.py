import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.logger import setup_logger

logger = setup_logger(__name__, "feature_engineer.log")


class StudentFeatureEngineer:
    """Feature engineering for UCI Student Performance dataset."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.fitted = False
        
        # Define categorical columns
        self.binary_cols = [
            'school', 'sex', 'address', 'famsize', 'Pstatus',
            'schoolsup', 'famsup', 'paid', 'activities', 
            'nursery', 'higher', 'internet', 'romantic'
        ]
        
        self.nominal_cols = ['Mjob', 'Fjob', 'reason', 'guardian']
        
        self.ordinal_cols = [
            'Medu', 'Fedu', 'traveltime', 'studytime', 
            'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health'
        ]
        
        self.numeric_cols = ['age', 'failures', 'absences', 'G1', 'G2']
    
    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Engineer features from the student dataset.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training, False for inference)
        
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        logger.info("Engineering features for student performance data")
        
        # Encode binary categorical variables
        df = self._encode_binary_features(df, fit)
        
        # Encode nominal categorical variables (one-hot encoding)
        df = self._encode_nominal_features(df, fit)
        
        # Create derived features
        df = self._create_derived_features(df)
        
        # Create interaction features
        df = self._create_interaction_features(df)
        
        # Create risk indicators
        df = self._create_risk_indicators(df)
        
        logger.info(f"Feature engineering complete: {len(df.columns)} features")
        
        self.fitted = True
        return df
    
    def _encode_binary_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Encode binary categorical features."""
        binary_mapping = {
            'school': {'GP': 0, 'MS': 1},
            'sex': {'F': 0, 'M': 1},
            'address': {'R': 0, 'U': 1},
            'famsize': {'LE3': 0, 'GT3': 1},
            'Pstatus': {'A': 0, 'T': 1},
            'schoolsup': {'no': 0, 'yes': 1},
            'famsup': {'no': 0, 'yes': 1},
            'paid': {'no': 0, 'yes': 1},
            'activities': {'no': 0, 'yes': 1},
            'nursery': {'no': 0, 'yes': 1},
            'higher': {'no': 0, 'yes': 1},
            'internet': {'no': 0, 'yes': 1},
            'romantic': {'no': 0, 'yes': 1}
        }
        
        for col, mapping in binary_mapping.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        logger.info("Encoded binary categorical features")
        return df
    
    def _encode_nominal_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """One-hot encode nominal categorical features."""
        for col in self.nominal_cols:
            if col in df.columns:
                if fit:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    self.label_encoders[col] = list(dummies.columns)
                else:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    # Ensure same columns as training
                    if col in self.label_encoders:
                        for expected_col in self.label_encoders[col]:
                            if expected_col not in dummies.columns:
                                dummies[expected_col] = 0
                        dummies = dummies[[c for c in self.label_encoders[col] if c in dummies.columns]]
                
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        
        logger.info("One-hot encoded nominal categorical features")
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing ones."""
        
        # Parent education average
        if 'Medu' in df.columns and 'Fedu' in df.columns:
            df['parent_edu_avg'] = (df['Medu'] + df['Fedu']) / 2
            df['parent_edu_max'] = df[['Medu', 'Fedu']].max(axis=1)
            df['parent_edu_diff'] = abs(df['Medu'] - df['Fedu'])
            logger.info("Created parent education features")
        
        # Study-leisure balance
        if 'studytime' in df.columns and 'freetime' in df.columns:
            df['study_leisure_ratio'] = df['studytime'] / (df['freetime'] + 1)
            df['study_leisure_diff'] = df['studytime'] - df['freetime']
            logger.info("Created study-leisure balance features")
        
        # Social engagement score
        if 'goout' in df.columns and 'romantic' in df.columns:
            social_cols = ['goout', 'romantic', 'activities'] if 'activities' in df.columns else ['goout', 'romantic']
            existing_social = [c for c in social_cols if c in df.columns]
            df['social_engagement'] = df[existing_social].sum(axis=1)
            logger.info("Created social engagement feature")
        
        # Alcohol consumption (combined)
        if 'Dalc' in df.columns and 'Walc' in df.columns:
            df['total_alcohol'] = df['Dalc'] + df['Walc']
            df['alcohol_weekly_avg'] = (df['Dalc'] * 5 + df['Walc'] * 2) / 7
            logger.info("Created alcohol consumption features")
        
        # Support system score
        support_cols = ['schoolsup', 'famsup', 'paid']
        existing_support = [c for c in support_cols if c in df.columns]
        if existing_support:
            df['support_score'] = df[existing_support].sum(axis=1)
            logger.info("Created support system score")
        
        # Grade progression (if G1 and G2 are available)
        if 'G1' in df.columns and 'G2' in df.columns:
            df['grade_progression'] = df['G2'] - df['G1']
            df['grade_avg_g1g2'] = (df['G1'] + df['G2']) / 2
            df['grade_trend'] = np.where(df['G2'] > df['G1'], 1, 
                                         np.where(df['G2'] < df['G1'], -1, 0))
            logger.info("Created grade progression features")
        
        # Age-related features
        if 'age' in df.columns:
            df['age_above_avg'] = (df['age'] > df['age'].mean()).astype(int)
            logger.info("Created age-related features")
        
        # Absence rate category
        if 'absences' in df.columns:
            df['absence_category'] = pd.cut(
                df['absences'],
                bins=[-1, 0, 5, 10, 20, float('inf')],
                labels=[0, 1, 2, 3, 4]
            ).astype(int)
            df['log_absences'] = np.log1p(df['absences'])
            logger.info("Created absence features")
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables."""
        
        # Study time × Parent education
        if 'studytime' in df.columns and 'parent_edu_avg' in df.columns:
            df['studytime_x_parent_edu'] = df['studytime'] * df['parent_edu_avg']
        
        # Failures × Absences
        if 'failures' in df.columns and 'absences' in df.columns:
            df['failures_x_absences'] = df['failures'] * df['absences']
        
        # Internet × Higher education aspiration
        if 'internet' in df.columns and 'higher' in df.columns:
            df['internet_x_higher'] = df['internet'] * df['higher']
        
        # Family support × Study time
        if 'famsup' in df.columns and 'studytime' in df.columns:
            df['famsup_x_studytime'] = df['famsup'] * df['studytime']
        
        # Alcohol × Study time (negative interaction)
        if 'total_alcohol' in df.columns and 'studytime' in df.columns:
            df['alcohol_x_studytime'] = df['total_alcohol'] * df['studytime']
        
        logger.info("Created interaction features")
        return df
    
    def _create_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk indicators for academic failure."""
        
        # High absence risk
        if 'absences' in df.columns:
            df['high_absence_risk'] = (df['absences'] > df['absences'].quantile(0.75)).astype(int)
        
        # Previous failure risk
        if 'failures' in df.columns:
            df['has_failures'] = (df['failures'] > 0).astype(int)
            df['multiple_failures'] = (df['failures'] > 1).astype(int)
        
        # Low study time risk
        if 'studytime' in df.columns:
            df['low_study_risk'] = (df['studytime'] <= 1).astype(int)
        
        # High alcohol risk
        if 'total_alcohol' in df.columns:
            df['high_alcohol_risk'] = (df['total_alcohol'] > 6).astype(int)
        
        # Combined risk score
        risk_cols = ['high_absence_risk', 'has_failures', 'low_study_risk', 'high_alcohol_risk']
        existing_risk = [c for c in risk_cols if c in df.columns]
        if existing_risk:
            df['combined_risk_score'] = df[existing_risk].sum(axis=1)
        
        logger.info("Created risk indicator features")
        return df


def prepare_features(
    df: pd.DataFrame,
    dataset_type: str,
    drop_columns: Optional[List[str]] = None,
    fit: bool = True
) -> pd.DataFrame:
    """
    Apply feature engineering based on dataset type.
    
    Args:
        df: Input DataFrame
        dataset_type: Type of dataset ('math' or 'portuguese')
        drop_columns: Columns to drop after feature engineering
        fit: Whether to fit the feature engineer
    
    Returns:
        DataFrame with engineered features
    """
    engineer = StudentFeatureEngineer()
    df = engineer.engineer_features(df, fit=fit)
    
    # Drop specified columns
    if drop_columns:
        cols_to_drop = [col for col in drop_columns if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped columns: {cols_to_drop}")
    
    # Select only numeric columns for modeling
    numeric_df = df.select_dtypes(include=['number', 'bool'])
    logger.info(f"Final feature set: {len(numeric_df.columns)} columns")
    
    return numeric_df
