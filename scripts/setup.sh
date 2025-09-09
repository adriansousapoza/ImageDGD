#!/bin/bash
# Setup script for ImageDGD

echo "Setting up ImageDGD environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p outputs
mkdir -p logs
mkdir -p figures
mkdir -p models
mkdir -p optuna_studies

# Initialize MLflow
echo "Initializing MLflow..."
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:./mlruns --default-artifact-root ./mlruns &

echo "Setup complete!"
echo "To activate the environment: source venv/bin/activate"
echo "To run training: python scripts/cli.py train"
echo "To run optimization: python scripts/cli.py optimize"
echo "To view MLflow UI: python scripts/cli.py mlflow-ui"
