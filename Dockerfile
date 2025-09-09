FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data outputs logs figures models optuna_studies

# Make scripts executable
RUN chmod +x scripts/*.py scripts/*.sh

# Set environment variables
ENV PYTHONPATH=/app/src
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

# Expose ports
EXPOSE 5000 8080

# Default command
CMD ["python", "scripts/cli.py", "train"]
