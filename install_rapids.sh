#!/bin/bash
# RAPIDS Environment Installation Script with CUDA 13
# This script sets up a RAPIDS environment optimized for ImageDGD with CUDA 13 support
# Using UV for fast package installation
# Author: Generated for ImageDGD project
# Date: 2025-10-27

set -e  # Exit on error

echo "========================================================================"
echo "  RAPIDS + CUDA 13 Installation"
echo "========================================================================"
echo ""

# Try to load CUDA 13 module if available
echo "Checking for CUDA 13..."
if module load cuda/13.0 2>/dev/null; then
    echo "  ✓ CUDA 13 module loaded from system"
    CUDA_LOADED=true
elif module load cuda/13 2>/dev/null; then
    echo "  ✓ CUDA 13 module loaded from system"
    CUDA_LOADED=true
else
    echo "  ✗ CUDA 13 module not available on this system"
    CUDA_LOADED=false
fi

# Check if CUDA is available via nvcc
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "  ✓ CUDA detected: version $CUDA_VERSION"
    CUDA_AVAILABLE=true
else
    echo "  ✗ nvcc not found in PATH"
    CUDA_AVAILABLE=false
fi

# If CUDA is not available, install CUDA Toolkit via conda-forge
if [ "$CUDA_AVAILABLE" = false ]; then
    echo ""
    echo "CUDA 13 not found. Installing CUDA Toolkit locally..."
    echo "This will install CUDA 13 in the virtual environment for PyTorch compatibility"
    INSTALL_CUDA=true
else
    INSTALL_CUDA=false
fi

# Confirm modules are loaded
if [ "$CUDA_LOADED" = true ]; then
    echo ""
    echo "Loaded modules:"
    module list
fi

# Install UV if not already installed
echo ""
echo "------------------------------------------------------------------------"
echo "Checking for UV package manager..."
if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "  ✓ UV installed: $(uv --version)"
else
    echo "  ✓ UV version: $(uv --version)"
fi

# Create Python virtual environment with UV
echo ""
echo "------------------------------------------------------------------------"
echo "Creating Python virtual environment with Python 3.13..."
echo "This may take several minutes..."
echo ""

# Define global environment location
VENV_DIR="$HOME/.venvs/rapids_cuda13"

# Create global venv directory if it doesn't exist
mkdir -p "$HOME/.venvs"

# Remove old environment if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing environment at $VENV_DIR..."
    rm -rf "$VENV_DIR"
fi

# Create virtual environment with Python 3.13 in global location
uv venv "$VENV_DIR" --python 3.13

echo ""
echo "  ✓ Virtual environment created at $VENV_DIR"

# Activate the environment
echo ""
echo "Activating environment..."
source "$VENV_DIR/bin/activate"

# Verify Python version
PYTHON_VERSION=$(python --version)
echo "  ✓ Python version: $PYTHON_VERSION"

# Install CUDA Toolkit locally if needed (BEFORE RAPIDS)
if [ "$INSTALL_CUDA" = true ]; then
    echo ""
    echo "------------------------------------------------------------------------"
    echo "Installing CUDA Toolkit 13.0 in virtual environment..."
    echo "This provides CUDA runtime libraries for PyTorch and RAPIDS"
    
    # Install CUDA toolkit via pip (provides cudatoolkit for the environment)
    uv pip install nvidia-cuda-runtime-cu13 nvidia-cuda-nvcc-cu13 nvidia-cudnn-cu13
    
    echo "  ✓ CUDA Toolkit installed in virtual environment"
fi

# Install RAPIDS packages FIRST (version 25.10 with CUDA 13)
echo ""
echo "========================================================================"
echo "Installing RAPIDS 25.10 packages (CUDA 13)..."
echo "IMPORTANT: Installing RAPIDS before other packages for proper dependency resolution"
echo ""

# Install RAPIDS with exact version specifications
uv pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu13==25.10.*" \
    "dask-cudf-cu13==25.10.*" \
    "cuml-cu13==25.10.*" \
    "cugraph-cu13==25.10.*" \
    "nx-cugraph-cu13==25.10.*" \
    "cuxfilter-cu13==25.10.*" \
    "cucim-cu13==25.10.*" \
    "pylibraft-cu13==25.10.*" \
    "raft-dask-cu13==25.10.*" \
    "cuvs-cu13==25.10.*"

echo ""
echo "  ✓ RAPIDS 25.10 installed successfully"

# Verify RAPIDS installation
echo ""
echo "Testing RAPIDS installation..."
python -c "import cudf; print('  ✓ RAPIDS cuDF working! Version:', cudf.__version__)" || echo "  ✗ cuDF failed"
python -c "import cuml; print('  ✓ cuML working! Version:', cuml.__version__)" || echo "  ✗ cuML failed"
python -c "import cupy; print('  ✓ CuPy working!')" || echo "  ✗ CuPy failed"

# Install PyTorch with CUDA 13 support (AFTER RAPIDS)
echo ""
echo "------------------------------------------------------------------------"
echo "Installing PyTorch with CUDA 13.0 support..."
echo "Using official PyTorch CUDA 13.0 build"
echo ""

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

echo ""
echo "  ✓ PyTorch installed with CUDA 13.0 support"

# Test PyTorch installation
echo ""
echo "Testing PyTorch installation..."
python -c 'import torch; print(f"  ✓ PyTorch version: {torch.__version__}")'
python -c 'import torch; print(f"  ✓ CUDA available: {torch.cuda.is_available()}")'
python -c 'import torch; print(f"  ✓ CUDA device count: {torch.cuda.device_count()}")'
python -c 'import torch; print(f"  ✓ CUDA version (PyTorch): {torch.version.cuda}")'

# Install ImageDGD specific dependencies from requirements.txt if available
echo ""
echo "------------------------------------------------------------------------"
echo "Installing ImageDGD project dependencies..."
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    uv pip install -r "$PROJECT_DIR/requirements.txt"
    echo "  ✓ Installed requirements from requirements.txt"
else
    echo "  ✗ requirements.txt not found"
    echo "    You can install them later with: uv pip install -r requirements.txt"
fi

echo ""
echo "========================================================================"
echo "  Installation Complete!"
echo "========================================================================"
echo ""

# Get installed versions
PYTHON_VER=$(python --version 2>&1)
RAPIDS_VER=$(python -c 'import cudf; print(cudf.__version__)' 2>/dev/null || echo 'N/A')
PYTORCH_VER=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')
CUDA_PYTORCH=$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')
GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')

echo "Installation Summary:"
echo "------------------------------------------------------------------------"
echo "  Environment location: $VENV_DIR"
echo "  Python: $PYTHON_VER"
echo "  RAPIDS: $RAPIDS_VER (CUDA 13)"
echo "  PyTorch: $PYTORCH_VER (CUDA $CUDA_PYTORCH)"
echo "  GPUs detected: $GPU_COUNT"
echo "  CUDA Toolkit: $([ "$INSTALL_CUDA" = true ] && echo 'Installed locally' || echo 'Using system CUDA')"
echo ""
echo "Installed components:"
echo "  - RAPIDS 25.10 (cuDF, cuML, cuGraph, cuPy) - GPU data science"
echo "  - PyTorch + torchvision (CUDA 13.0) - Deep learning"
echo "  - ImageDGD dependencies"
echo ""
echo "To activate the environment from any directory:"
echo "------------------------------------------------------------------------"
echo "  Option 1: Add an alias (recommended)"
echo "     echo 'alias rapids=\"source $VENV_DIR/bin/activate\"' >> ~/.zshrc"
echo "     source ~/.zshrc"
echo "     rapids"
echo ""
echo "  Option 2: Activate manually"
echo "     source $VENV_DIR/bin/activate"
echo ""
echo "  To deactivate:"
echo "     deactivate"
echo ""
echo "========================================================================"
echo "  Setup alias now? (recommended)"
echo "========================================================================"
echo ""
read -p "Add 'rapids' alias to ~/.zshrc? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Remove old aliases if they exist
    sed -i '/alias rapids=/d' ~/.zshrc 2>/dev/null || true
    sed -i '/alias activate_rapids=/d' ~/.zshrc 2>/dev/null || true
    
    # Add new alias
    echo "" >> ~/.zshrc
    echo "# RAPIDS environment activation" >> ~/.zshrc
    echo "alias rapids='source $VENV_DIR/bin/activate'" >> ~/.zshrc
    
    echo "  ✓ Alias added to ~/.zshrc"
    echo "  ✓ Run 'source ~/.zshrc' or open a new terminal"
    echo "  ✓ Then simply type 'rapids' to activate"
else
    echo "  Skipped. You can manually activate with:"
    echo "    source $VENV_DIR/bin/activate"
fi
echo ""
