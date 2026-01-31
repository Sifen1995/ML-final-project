"""
Tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np

from src.feature_engineer import StudentFeatureEngineer


class TestStudentFeatureEngineer:
    """Test cases for StudentFeatureEngineer."""
    
    def test_engineer_initialization(self):
        """Test StudentFeatureEngineer initialization."""
        engineer = StudentFeatureEngineer()
        assert engineer is not None
    
    def test_create_academic_ratio(self, sample_student_data):
        """Test academic performance ratio creation."""
        engineer = StudentFeatureEngineer()
        df = sample_student_data.copy()
        
        result = engineer.create_academic_ratio(df)
        
        assert 'academic_ratio' in result.columns
        assert result['academic_ratio'].isna().sum() == 0
    
    def test_create_study_efficiency(self, sample_student_data):
        """Test study efficiency feature creation."""
        engineer = StudentFeatureEngineer()
        df = sample_student_data.copy()
        
        result = engineer.create_study_efficiency(df)
        
        assert 'study_efficiency' in result.columns
    
    def test_create_family_education_score(self, sample_student_data):
        """Test family education score creation."""
        engineer = StudentFeatureEngineer()
        df = sample_student_data.copy()
        
        result = engineer.create_family_education_score(df)
        
        assert 'family_education_score' in result.columns
        # Score should be average of Medu and Fedu
        expected = (df['Medu'] + df['Fedu']) / 2
        pd.testing.assert_series_equal(
            result['family_education_score'], 
            expected.rename('family_education_score'),
            check_exact=False
        )
    
    def test_create_risk_indicator(self, sample_student_data):
        """Test at-risk student indicator."""
        engineer = StudentFeatureEngineer()
        df = sample_student_data.copy()
        
        result = engineer.create_risk_indicator(df)
        
        assert 'at_risk' in result.columns
        assert result['at_risk'].dtype in [np.int64, np.int32, bool]
    
    def test_create_grade_trend(self, sample_student_data):
        """Test grade trend feature."""
        engineer = StudentFeatureEngineer()
        df = sample_student_data.copy()
        
        result = engineer.create_grade_trend(df)
        
        assert 'grade_trend' in result.columns
        # Trend should be G2 - G1
        expected = df['G2'] - df['G1']
        pd.testing.assert_series_equal(
            result['grade_trend'],
            expected.rename('grade_trend'),
            check_exact=False
        )
    
    def test_create_all_features(self, sample_student_data):
        """Test creating all engineered features."""
        engineer = StudentFeatureEngineer()
        df = sample_student_data.copy()
        
        result = engineer.create_all_features(df)
        
        # Should have more columns than original
        assert len(result.columns) > len(df.columns)
    
    def test_features_no_nan_introduced(self, sample_student_data):
        """Test that feature engineering doesn't introduce NaN."""
        engineer = StudentFeatureEngineer()
        df = sample_student_data.copy()
        
        initial_nan = df.isna().sum().sum()
        result = engineer.create_all_features(df)
        
        # New features should not have more NaN than handled
        new_cols = [c for c in result.columns if c not in df.columns]
        for col in new_cols:
            assert result[col].isna().sum() == 0, f"NaN introduced in {col}"


class TestFeatureCalculations:
    """Test cases for specific feature calculations."""
    
    def test_support_score_calculation(self, sample_student_data):
        """Test support score calculation."""
        engineer = StudentFeatureEngineer()
        df = sample_student_data.copy()
        
        result = engineer.create_support_score(df)
        
        assert 'support_score' in result.columns
        # Score should be between 0 and 1
        assert result['support_score'].min() >= 0
        assert result['support_score'].max() <= 1
    
    def test_social_activity_score(self, sample_student_data):
        """Test social activity score calculation."""
        engineer = StudentFeatureEngineer()
        df = sample_student_data.copy()
        
        result = engineer.create_social_score(df)
        
        assert 'social_score' in result.columns
    
    def test_alcohol_consumption_score(self, sample_student_data):
        """Test alcohol consumption combined score."""
        engineer = StudentFeatureEngineer()
        df = sample_student_data.copy()
        
        result = engineer.create_alcohol_score(df)
        
        assert 'alcohol_score' in result.columns
        # Weighted average of Dalc and Walc
        expected = (df['Dalc'] * 5 + df['Walc'] * 2) / 7
        pd.testing.assert_series_equal(
            result['alcohol_score'],
            expected.rename('alcohol_score'),
            check_exact=False,
            atol=0.01
        )
