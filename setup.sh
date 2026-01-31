#!/bin/bash

# =============================================================================
# EduPredict Analytics - Setup Script
# Student Grade Prediction System
# =============================================================================

set -e  # Exit on any error

echo "=============================================="
echo "  EduPredict Analytics - Setup Script"
echo "  Student Grade Prediction System"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv .venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip --quiet
print_success "Pip upgraded"

# Install dependencies
print_status "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet
print_success "Dependencies installed"

# Create necessary directories
print_status "Creating directory structure..."
mkdir -p data/raw data/processed models logs notebooks
print_success "Directories created"

# Copy data files to raw folder if they exist
if [ -f "data/student-mat.csv" ]; then
    print_status "Setting up data files..."
    cp -n data/student-mat.csv data/raw/ 2>/dev/null || true
    cp -n data/student-por.csv data/raw/ 2>/dev/null || true
    print_success "Data files ready"
fi

# Run preprocessing
print_status "Preprocessing data..."
python main.py preprocess --dataset math
print_success "Data preprocessed"

# Train model
print_status "Training model (this may take a minute)..."
python main.py train --dataset math
print_success "Model trained"

# Evaluate model
print_status "Evaluating model..."
python main.py evaluate --dataset math
print_success "Model evaluated"

echo ""
echo "=============================================="
print_success "Setup Complete!"
echo "=============================================="
echo ""
echo "You can now run the application using:"
echo ""
echo "  1. Dashboard (Web UI):"
echo "     python -m streamlit run dashboard.py"
echo ""
echo "  2. API Server:"
echo "     python api.py"
echo ""
echo "  3. Command Line:"
echo "     python main.py predict --student '{\"age\": 18, \"studytime\": 2, ...}'"
echo ""
echo "  4. Run Tests:"
echo "     python -m pytest tests/ -v"
echo ""
echo "=============================================="
