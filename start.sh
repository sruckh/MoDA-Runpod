#!/bin/bash
set -e

echo "Starting MoDA container initialization..."

# Set environment variables for non-interactive installation
export DEBIAN_FRONTEND=noninteractive
export TZ=UTC

# Configure timezone non-interactively
ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install pip if not available
echo "Ensuring pip is available..."
python3 -m ensurepip --upgrade || curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Install PyTorch with CUDA 12.1 support
echo "Installing PyTorch..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo "PyTorch installation complete"

# Clone MoDA repository
echo "Cloning MoDA repository..."
if [ ! -d "MoDA" ]; then
    git clone https://github.com/lixinyyang/MoDA.git
fi
cd MoDA

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install flash attention with dynamic version detection
echo "Installing flash attention..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
echo "Detected Python version: $PYTHON_VERSION"

# Install flash attention based on Python version
if [ "$PYTHON_VERSION" = "312" ]; then
    echo "Installing flash_attn for Python 3.12..."
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
elif [ "$PYTHON_VERSION" = "311" ]; then
    echo "Installing flash_attn for Python 3.11..."
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
elif [ "$PYTHON_VERSION" = "310" ]; then
    echo "Installing flash_attn for Python 3.10..."
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
else
    echo "Unknown Python version $PYTHON_VERSION, trying Python 3.12 wheel..."
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
fi

echo "Installation complete. Starting application..."

# Start the application
python app.py
