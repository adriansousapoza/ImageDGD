#!/bin/bash
# RAPIDS Environment Installation Script with CUDA 13/12.8
# This script sets up a RAPIDS environment optimized for ImageDGD with CUDA support
# Prefers CUDA 13, falls back to CUDA 12.8 if unavailable
# Using UV for fast package installation
# Author: Generated for ImageDGD project
# Date: 2025-10-30

set -e  # Exit on error

echo "========================================================================"
echo "  RAPIDS + CUDA Installation"
echo "========================================================================"
echo ""

# Check for CUDA 13 first (preferred)
echo "Checking for CUDA 13..."
CUDA_VERSION=""
CUDA_SUFFIX=""
RAPIDS_SUFFIX=""
CUDA_LOADED=false
INSTALL_CUDA=false

if module load cuda/13.0 2>/dev/null; then
    echo "  ✓ CUDA 13.0 module loaded from system"
    CUDA_VERSION="13"
    CUDA_SUFFIX="cu130"
    RAPIDS_SUFFIX="cu13"
    CUDA_LOADED=true
elif module load cuda/13 2>/dev/null; then
    echo "  ✓ CUDA 13 module loaded from system"
    CUDA_VERSION="13"
    CUDA_SUFFIX="cu130"
    RAPIDS_SUFFIX="cu13"
    CUDA_LOADED=true
else
    echo "  ✗ CUDA 13 module not available"
    echo ""
    echo "Checking for CUDA 12.8 as fallback..."
    
    if module load cuda/12.8 2>/dev/null; then
        echo "  ✓ CUDA 12.8 module loaded from system"
        CUDA_VERSION="12"
        CUDA_SUFFIX="cu121"
        RAPIDS_SUFFIX="cu12"
        CUDA_LOADED=true
    elif module load cuda/12.5 2>/dev/null; then
        echo "  ✓ CUDA 12.5 module loaded from system"
        CUDA_VERSION="12"
        CUDA_SUFFIX="cu121"
        RAPIDS_SUFFIX="cu12"
        CUDA_LOADED=true
    elif module load cuda/12.2 2>/dev/null; then
        echo "  ✓ CUDA 12.2 module loaded from system"
        CUDA_VERSION="12"
        CUDA_SUFFIX="cu121"
        RAPIDS_SUFFIX="cu12"
        CUDA_LOADED=true
    else
        echo "  ✗ No CUDA 12.x module available"
        CUDA_LOADED=false
    fi
fi

# Check if CUDA is available via nvcc
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "  ✓ nvcc detected: version $NVCC_VERSION"
    CUDA_AVAILABLE=true
    
    # If no module was loaded but nvcc exists, determine version
    if [ "$CUDA_LOADED" = false ]; then
        if [[ "$NVCC_VERSION" == 13* ]]; then
            CUDA_VERSION="13"
            CUDA_SUFFIX="cu130"
            RAPIDS_SUFFIX="cu13"
        else
            CUDA_VERSION="12"
            CUDA_SUFFIX="cu121"
            RAPIDS_SUFFIX="cu12"
        fi
    fi
else
    echo "  ✗ nvcc not found in PATH"
    CUDA_AVAILABLE=false
fi

# If CUDA is not available, we'll install it locally
if [ "$CUDA_AVAILABLE" = false ] && [ "$CUDA_LOADED" = false ]; then
    echo ""
    echo "No CUDA found. Installing CUDA Toolkit locally..."
    echo "Preference: CUDA 13, fallback to CUDA 12 if unavailable"
    INSTALL_CUDA=true
    # Default to CUDA 13 for local installation
    CUDA_VERSION="13"
    CUDA_SUFFIX="cu130"
    RAPIDS_SUFFIX="cu13"
fi


# Confirm modules are loaded and display final CUDA version choice
echo ""
echo "------------------------------------------------------------------------"
echo "Using CUDA $CUDA_VERSION (suffix: $CUDA_SUFFIX for PyTorch, $RAPIDS_SUFFIX for RAPIDS)"
echo "------------------------------------------------------------------------"

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

# Define global environment location based on CUDA version
if [ "$CUDA_VERSION" = "13" ]; then
    VENV_DIR="$HOME/.venvs/rapids_cuda13"
else
    VENV_DIR="$HOME/.venvs/rapids_cuda12"
fi

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
    echo "Installing CUDA Toolkit $CUDA_VERSION in virtual environment..."
    echo "This provides CUDA runtime libraries for PyTorch and RAPIDS"
    
    # Install CUDA toolkit via pip based on version
    if [ "$CUDA_VERSION" = "13" ]; then
        # Try CUDA 13 first, fallback to 12 if it fails
        if ! uv pip install nvidia-cuda-runtime-cu13 nvidia-cuda-nvcc-cu13 nvidia-cudnn-cu13 2>/dev/null; then
            echo "  ⚠ CUDA 13 packages not available, falling back to CUDA 12..."
            CUDA_VERSION="12"
            CUDA_SUFFIX="cu121"
            RAPIDS_SUFFIX="cu12"
            uv pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
        fi
    else
        uv pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
    fi
    
    echo "  ✓ CUDA Toolkit $CUDA_VERSION installed in virtual environment"
fi

# Install RAPIDS packages FIRST (version 25.10)
echo ""
echo "========================================================================"
echo "Installing RAPIDS 25.10 packages (CUDA $CUDA_VERSION)..."
echo "IMPORTANT: Installing RAPIDS before other packages for proper dependency resolution"
echo ""

# Install RAPIDS with version-specific CUDA suffix
uv pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-${RAPIDS_SUFFIX}==25.10.*" \
    "dask-cudf-${RAPIDS_SUFFIX}==25.10.*" \
    "cuml-${RAPIDS_SUFFIX}==25.10.*" \
    "cugraph-${RAPIDS_SUFFIX}==25.10.*" \
    "nx-cugraph-${RAPIDS_SUFFIX}==25.10.*" \
    "cuxfilter-${RAPIDS_SUFFIX}==25.10.*" \
    "cucim-${RAPIDS_SUFFIX}==25.10.*" \
    "pylibraft-${RAPIDS_SUFFIX}==25.10.*" \
    "raft-dask-${RAPIDS_SUFFIX}==25.10.*" \
    "cuvs-${RAPIDS_SUFFIX}==25.10.*"

echo ""
echo "  ✓ RAPIDS 25.10 installed successfully"

# Verify RAPIDS installation
echo ""
echo "Testing RAPIDS installation..."
python -c "import cudf; print('  ✓ RAPIDS cuDF working! Version:', cudf.__version__)" || echo "  ✗ cuDF failed"
python -c "import cuml; print('  ✓ cuML working! Version:', cuml.__version__)" || echo "  ✗ cuML failed"
python -c "import cupy; print('  ✓ CuPy working!')" || echo "  ✗ CuPy failed"

# Install PyTorch with appropriate CUDA support (AFTER RAPIDS)
echo ""
echo "------------------------------------------------------------------------"
echo "Installing PyTorch with CUDA $CUDA_VERSION support..."
echo ""

if [ "$CUDA_VERSION" = "13" ]; then
    echo "Using PyTorch with CUDA 13.0..."
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
else
    echo "Using PyTorch with CUDA 12.x..."
    uv pip install torch torchvision
fi

echo ""
echo "  ✓ PyTorch installed with CUDA $CUDA_VERSION support"

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
echo "  RAPIDS: $RAPIDS_VER (CUDA $CUDA_VERSION)"
echo "  PyTorch: $PYTORCH_VER (CUDA $CUDA_PYTORCH)"
echo "  GPUs detected: $GPU_COUNT"
echo "  CUDA Toolkit: $([ "$INSTALL_CUDA" = true ] && echo "Installed locally (CUDA $CUDA_VERSION)" || echo "Using system CUDA $CUDA_VERSION")"
echo ""
echo "Installed components:"
echo "  - RAPIDS 25.10 (cuDF, cuML, cuGraph, cuPy) - GPU data science"
echo "  - PyTorch + torchvision (CUDA $CUDA_VERSION) - Deep learning"
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
