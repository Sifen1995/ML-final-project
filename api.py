"""
FastAPI application for student grade prediction service.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.predictor import load_predictor
from src.logger import setup_logger
from src.config import api_config, model_config

logger = setup_logger(__name__, "api.log")

app = FastAPI(
    title=api_config.title,
    description=api_config.description,
    version=api_config.version
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store loaded predictors in memory
predictors_cache = {}


# Pydantic models for request/response
class StudentData(BaseModel):
    """Student information for grade prediction."""
    school: str = Field(..., description="Student's school (GP or MS)")
    sex: str = Field(..., description="Student's sex (F or M)")
    age: int = Field(..., ge=15, le=22, description="Student's age")
    address: str = Field(..., description="Home address type (U or R)")
    famsize: str = Field(..., description="Family size (LE3 or GT3)")
    Pstatus: str = Field(..., description="Parent's cohabitation status (T or A)")
    Medu: int = Field(..., ge=0, le=4, description="Mother's education (0-4)")
    Fedu: int = Field(..., ge=0, le=4, description="Father's education (0-4)")
    Mjob: str = Field(..., description="Mother's job")
    Fjob: str = Field(..., description="Father's job")
    reason: str = Field(..., description="Reason to choose this school")
    guardian: str = Field(..., description="Student's guardian")
    traveltime: int = Field(..., ge=1, le=4, description="Home to school travel time")
    studytime: int = Field(..., ge=1, le=4, description="Weekly study time")
    failures: int = Field(..., ge=0, le=4, description="Number of past class failures")
    schoolsup: str = Field(..., description="Extra educational support")
    famsup: str = Field(..., description="Family educational support")
    paid: str = Field(..., description="Extra paid classes")
    activities: str = Field(..., description="Extra-curricular activities")
    nursery: str = Field(..., description="Attended nursery school")
    higher: str = Field(..., description="Wants to take higher education")
    internet: str = Field(..., description="Internet access at home")
    romantic: str = Field(..., description="In a romantic relationship")
    famrel: int = Field(..., ge=1, le=5, description="Quality of family relationships")
    freetime: int = Field(..., ge=1, le=5, description="Free time after school")
    goout: int = Field(..., ge=1, le=5, description="Going out with friends")
    Dalc: int = Field(..., ge=1, le=5, description="Workday alcohol consumption")
    Walc: int = Field(..., ge=1, le=5, description="Weekend alcohol consumption")
    health: int = Field(..., ge=1, le=5, description="Current health status")
    absences: int = Field(..., ge=0, le=93, description="Number of school absences")
    G1: Optional[int] = Field(None, ge=0, le=20, description="First period grade")
    G2: Optional[int] = Field(None, ge=0, le=20, description="Second period grade")
    
    class Config:
        json_schema_extra = {
            "example": {
                "school": "GP",
                "sex": "F",
                "age": 18,
                "address": "U",
                "famsize": "GT3",
                "Pstatus": "T",
                "Medu": 4,
                "Fedu": 4,
                "Mjob": "teacher",
                "Fjob": "health",
                "reason": "course",
                "guardian": "mother",
                "traveltime": 2,
                "studytime": 3,
                "failures": 0,
                "schoolsup": "no",
                "famsup": "yes",
                "paid": "no",
                "activities": "yes",
                "nursery": "yes",
                "higher": "yes",
                "internet": "yes",
                "romantic": "no",
                "famrel": 4,
                "freetime": 3,
                "goout": 3,
                "Dalc": 1,
                "Walc": 2,
                "health": 4,
                "absences": 4,
                "G1": 14,
                "G2": 15
            }
        }


class PredictionRequest(BaseModel):
    """Request for batch predictions."""
    students: List[Dict[str, Any]]
    include_details: bool = True


class PredictionResponse(BaseModel):
    """Response for prediction."""
    predicted_grade: float
    predicted_grade_int: int
    performance_level: str
    pass_status: str
    confidence_range: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[PredictionResponse]
    total_students: int
    average_predicted_grade: float
    pass_rate: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    models_available: Dict[str, bool]


def get_predictor(model_type: str, dataset_type: str):
    """Get or load predictor from cache."""
    cache_key = f"{model_type}_{dataset_type}"
    
    if cache_key not in predictors_cache:
        try:
            logger.info(f"Loading predictor: {cache_key}")
            predictor = load_predictor(model_type, dataset_type)
            predictors_cache[cache_key] = predictor
            logger.info(f"Predictor loaded successfully: {cache_key}")
        except Exception as e:
            logger.error(f"Failed to load predictor {cache_key}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Model not available: {model_type} for {dataset_type}"
            )
    
    return predictors_cache[cache_key]


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Student Grade Prediction API",
        "version": api_config.version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    models_available = {}
    
    for model_type in model_config.model_types:
        for dataset_type in ['math', 'portuguese']:
            model_key = f"{model_type}_{dataset_type}"
            model_path = model_config.model_save_dir / f"{model_key}_model.joblib"
            models_available[model_key] = model_path.exists()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_available=models_available
    )


@app.post("/predict/math", response_model=PredictionResponse)
async def predict_math_grade(
    student: StudentData,
    model_type: str = "random_forest"
):
    """
    Predict final grade for a Mathematics student.
    
    Args:
        student: Student data
        model_type: Model to use (linear_regression, random_forest, xgboost)
    
    Returns:
        Prediction result with grade and performance level
    """
    try:
        predictor = get_predictor(model_type, "math")
        student_dict = student.model_dump()
        result = predictor.predict_single(student_dict)
        
        return PredictionResponse(
            predicted_grade=result['predicted_grade'],
            predicted_grade_int=result['predicted_grade_int'],
            performance_level=result['performance_level'],
            pass_status=result['pass_status'],
            confidence_range=result['confidence_range'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/portuguese", response_model=PredictionResponse)
async def predict_portuguese_grade(
    student: StudentData,
    model_type: str = "random_forest"
):
    """
    Predict final grade for a Portuguese student.
    
    Args:
        student: Student data
        model_type: Model to use
    
    Returns:
        Prediction result
    """
    try:
        predictor = get_predictor(model_type, "portuguese")
        student_dict = student.model_dump()
        result = predictor.predict_single(student_dict)
        
        return PredictionResponse(
            predicted_grade=result['predicted_grade'],
            predicted_grade_int=result['predicted_grade_int'],
            performance_level=result['performance_level'],
            pass_status=result['pass_status'],
            confidence_range=result['confidence_range'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: PredictionRequest,
    dataset_type: str = "math",
    model_type: str = "random_forest"
):
    """
    Predict grades for multiple students.
    
    Args:
        request: Batch prediction request with students
        dataset_type: Dataset type (math or portuguese)
        model_type: Model to use
    
    Returns:
        Batch prediction results
    """
    try:
        predictor = get_predictor(model_type, dataset_type)
        df = pd.DataFrame(request.students)
        
        results = predictor.predict_batch(df, include_details=request.include_details)
        
        predictions = []
        for idx, row in results.iterrows():
            predictions.append(PredictionResponse(
                predicted_grade=float(row['predicted_grade']),
                predicted_grade_int=int(row['predicted_grade_int']),
                performance_level=row.get('performance_level', 'Unknown'),
                pass_status=row.get('pass_status', 'Unknown'),
                confidence_range=row.get('confidence_range', 'N/A'),
                timestamp=datetime.now().isoformat()
            ))
        
        avg_grade = results['predicted_grade'].mean()
        pass_rate = (results['predicted_grade'] >= 10).mean() * 100
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_students=len(results),
            average_predicted_grade=avg_grade,
            pass_rate=pass_rate
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models."""
    available_models = {}
    
    for model_type in model_config.model_types:
        for dataset_type in ['math', 'portuguese']:
            model_key = f"{model_type}_{dataset_type}"
            model_path = model_config.model_save_dir / f"{model_key}_model.joblib"
            
            if model_path.exists():
                available_models[model_key] = {
                    "model_type": model_type,
                    "dataset": dataset_type,
                    "path": str(model_path),
                    "loaded": model_key in predictors_cache
                }
    
    return {
        "available_models": available_models,
        "total": len(available_models)
    }


@app.get("/features")
async def get_features():
    """Get list of features used for prediction."""
    from src.config import data_config
    
    return {
        "demographic_features": data_config.demographic_features,
        "family_features": data_config.family_features,
        "academic_features": data_config.academic_features,
        "social_features": data_config.social_features,
        "target": data_config.target_column,
        "passing_grade": data_config.passing_grade
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=api_config.host,
        port=api_config.port,
        reload=api_config.reload
    )
