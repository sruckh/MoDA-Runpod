#!/bin/bash
set -e

echo "Starting MoDA container initialization..."

# Set environment variables for non-interactive installation
export DEBIAN_FRONTEND=noninteractive
export TZ=UTC

# Configure timezone non-interactively
ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Python 3.11 using deadsnakes PPA
echo "Installing Python 3.11..."
apt-get update && \
apt-get install -y --no-install-recommends software-properties-common git && \
add-apt-repository ppa:deadsnakes/ppa -y && \
apt-get update && \
apt-get install -y --no-install-recommends python3.11 python3-pip ffmpeg

# Set Python 3.11 as default
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

echo "Python installation complete"

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
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/flash_attn-2.7.0.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

echo "Installation complete. Starting application..."

# Start the application
python app.py
