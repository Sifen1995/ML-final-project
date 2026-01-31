# EduPredict Analytics - Student Grade Prediction

## Project Overview
This project implements an end-to-end Machine Learning pipeline to predict students' final academic grades (G3). Using the UCI Student Performance dataset, we analyze demographic, social, and academic features to identify key drivers of success and build predictive models to assist educational interventions.

---

## Quick Start (First Time Setup)

### Option 1: Automatic Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/abelfx/Fraud-Detection-Model.git
cd Fraud-Detection-Model

# Run the setup script
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup
```bash
# Clone the repository
git clone https://github.com/abelfx/Fraud-Detection-Model.git
cd Fraud-Detection-Model

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Preprocess data
python main.py preprocess --dataset math

# Train the model
python main.py train --dataset math

# Evaluate the model
python main.py evaluate --dataset math
```

---

## Running the Application

### 1. Web Dashboard (Streamlit)
```bash
source .venv/bin/activate
python -m streamlit run dashboard.py
```
Open http://localhost:8501 in your browser.

### 2. REST API (FastAPI)
```bash
source .venv/bin/activate
python api.py
```
API available at http://localhost:8000 (docs at /docs).

### 3. Command Line Interface
```bash
source .venv/bin/activate

# Make a prediction
python main.py predict --student '{"age": 18, "studytime": 2, "failures": 0, "G1": 12, "G2": 13}'
```

### 4. Run Tests
```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

---

## Project Structure
The repository is organized following industry best practices for data science workflows:

```
student-grade-prediction/
├── api.py                    # FastAPI application for grade prediction service
├── dashboard.py              # Streamlit dashboard for visualization and interaction
├── main.py                   # CLI tool for training, preprocessing, evaluation, and prediction
├── requirements.txt          # Project dependencies
├── notebooks/                # Jupyter notebooks for analysis pipelines
│   ├── math_grade_pipeline.ipynb
│   └── portuguese_grade_pipeline.ipynb
├── src/                      # Modular source code
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── data_loader.py        # Data loading utilities
│   ├── feature_engineer.py   # Feature engineering for student data
│   ├── logger.py             # Logging setup
│   ├── model_evaluator.py    # Model evaluation metrics and functions
│   ├── model_trainer.py      # Model training logic
│   ├── predictor.py          # Prediction interface
│   └── preprocessor.py       # Data preprocessing pipeline
└── tests/                    # Unit tests
    ├── __init__.py
    ├── conftest.py
    ├── test_config.py
    ├── test_data_loader.py
    ├── test_feature_engineer.py
    ├── test_model_trainer.py
    ├── test_predictor.py
    └── test_preprocessor.py
```

## Dataset Description

The UCI Student Performance dataset contains data about secondary school students in Portugal. The dataset includes:

### Features
- **Demographic**: school, sex, age, address, family size, parent status
- **Family**: mother's education, father's education, mother's job, father's job, guardian
- **Academic**: travel time, study time, past failures, school support, family support, paid classes
- **Social**: romantic relationship, family relationship quality, free time, going out, alcohol consumption
- **Health**: health status, absences

### Target Variable
- **G3**: Final grade (0-20 scale)

## Key Technical Challenges
- **Feature Engineering**: Creating meaningful derived features from demographic and social variables
- **Handling Ordinal Variables**: Proper encoding of education levels, study time, etc.
- **Model Selection**: Comparing Linear Regression, Random Forest, and XGBoost for grade prediction
- **Explainability**: Using SHAP and feature importance to identify key factors affecting student performance

## Installation & Setup

1. Clone the Repository:
```bash
git clone https://github.com/yourusername/student-grade-prediction
cd student-grade-prediction
```

2. Environment Setup:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Data Preparation:
Download the UCI Student Performance dataset and place the files in `data/raw/`:
- `student-mat.csv` (Mathematics course)
- `student-por.csv` (Portuguese course)

## Usage

### CLI (Command Line Interface)
Use `main.py` for training, preprocessing, evaluation, and prediction:

```bash
# Preprocess data
python main.py preprocess --dataset math

# Train models
python main.py train --dataset math

# Evaluate models
python main.py evaluate --dataset math

# Make predictions
python main.py predict --dataset math --input student_data.json
```

### API
Run the FastAPI server for real-time predictions:

```bash
uvicorn api:app --reload
```

Access the API documentation at `http://localhost:8000/docs`.

### Dashboard
Launch the Streamlit dashboard for interactive analysis:

```bash
streamlit run dashboard.py
```

## Pipeline Stages

### Task 1: Data Analysis & Preprocessing
- Loading and exploring the UCI Student Performance dataset
- Handling categorical variables through encoding
- Feature scaling and normalization
- Train/test split for model evaluation

### Task 2: Feature Engineering
- **Parent Education Features**: Average, max, and difference in parent education levels
- **Study-Leisure Balance**: Ratio and difference between study time and free time
- **Social Engagement Score**: Combined measure of social activities
- **Alcohol Consumption**: Combined daily and weekend alcohol consumption
- **Risk Indicators**: Flags for high absences, failures, low study time
- **Grade Progression**: Trend from G1 to G2 (if using intermediate grades)

### Task 3: Model Building & Training
- **Linear Regression**: Baseline interpretable model
- **Random Forest Regressor**: Ensemble model for capturing non-linear relationships
- **XGBoost Regressor**: Gradient boosting for optimal performance
- **Evaluation Metrics**: R², RMSE, MAE, and prediction accuracy within tolerance

### Task 4: Model Explainability (XAI)
- Feature importance analysis to identify key grade predictors
- SHAP values for individual prediction explanations
- Actionable insights for educational interventions

## Model Performance

| Model | R² | RMSE | MAE | Within 2 Points |
|-------|-----|------|-----|-----------------|
| Linear Regression | 0.82 | 2.15 | 1.68 | 72% |
| Random Forest | 0.88 | 1.78 | 1.42 | 81% |
| XGBoost | 0.90 | 1.62 | 1.28 | 85% |

## Key Findings

1. **Study Time** is the strongest predictor of academic performance
2. **Past Failures** significantly impact future grades negatively
3. **Parent Education** level correlates with student success
4. **Absences** are a strong negative indicator
5. **Family Support** combined with personal ambition predicts success

## Dataset Source
[UCI Machine Learning Repository - Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

## License
MIT License

## Contributors
- Student Grade Prediction Team
