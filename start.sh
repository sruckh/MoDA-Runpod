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

# Install PyTorch with CUDA 12.4 support
echo "Installing PyTorch..."
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

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

# Install flash attention
echo "Installing flash attention..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/flash_attn-2.7.0.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

echo "Installation complete. Starting application..."

# Start the application
python app.py
