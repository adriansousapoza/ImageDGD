#!/bin/bash
# RAPIDS Environment Activation Script
# Usage: source activate_rapids.sh
# Or add to ~/.zshrc: alias activate_rapids='source /path/to/activate_rapids.sh'

echo "========================================================================"
echo "  RAPIDS Environment Activation"
echo "========================================================================"
echo ""

# Try to load CUDA 13 module if available
echo "Checking for CUDA 13..."
if module load cuda/13.0 2>/dev/null; then
    echo "  ✓ CUDA 13.0 module loaded"
elif module load cuda/13 2>/dev/null; then
    echo "  ✓ CUDA 13 module loaded"
else
    echo "  ✗ CUDA module not available (will use locally installed CUDA)"
fi

# Activate RAPIDS virtual environment
echo ""
echo "Activating RAPIDS environment..."

# Use global environment location
VENV_PATH="$HOME/.venvs/rapids_cuda13"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "  ✓ Environment activated successfully"
else
    echo "  ✗ Virtual environment not found at $VENV_PATH"
    echo "    Please run install_rapids.sh first"
    return 1
fi

# Display environment info
echo ""
echo "------------------------------------------------------------------------"
echo "Environment Information:"
echo "------------------------------------------------------------------------"
echo "  Python: $(which python)"
echo "  Python version: $(python --version 2>&1)"
echo "  CUDA version: $(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)"
echo "  Virtual environment: $VIRTUAL_ENV"
echo "  GPU devices: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'unknown')"

# Quick functionality test
echo ""
echo "Running quick tests..."
if command -v python &> /dev/null; then
    # Test RAPIDS
    python -c "import cudf; print('  ✓ cuDF (RAPIDS) working')" 2>/dev/null || echo "  ✗ cuDF test failed"
    python -c "import cuml; print('  ✓ cuML (RAPIDS) working')" 2>/dev/null || echo "  ✗ cuML test failed"
    
    # Test PyTorch
    python -c "import torch; print('  ✓ PyTorch working')" 2>/dev/null || echo "  ✗ PyTorch test failed"
    python -c "import torch; print('  ✓ CUDA available:', torch.cuda.is_available())" 2>/dev/null || echo "  ✗ CUDA test failed"
    
    # Test other packages
    python -c "import umap; print('  ✓ UMAP working')" 2>/dev/null || echo "  ✗ UMAP test failed"
    
    echo ""
    echo "------------------------------------------------------------------------"
    echo "All systems ready!"
else
    echo "  ✗ Python not found. Something went wrong with activation."
    return 1
fi

echo ""
echo "Available commands:"
echo "  jupyter lab          - Start JupyterLab"
echo "  python script.py     - Run Python scripts"
echo "  uv pip install PKG   - Install packages with UV"
echo "  deactivate           - Exit environment"
echo ""
echo "========================================================================"
