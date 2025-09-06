# MoDA Runpod Container

A containerized version of the MoDA (Motion-guided Diffusion Avatar) project optimized for deployment on the Runpod platform.

## Overview

This Docker container provides a ready-to-run environment for the MoDA application with all necessary dependencies pre-configured for GPU-accelerated inference on Runpod.

## Features

- **CUDA 12.1 Support**: Built on `nvidia/cuda:12.1.0-cudnn8-devel-ubi8` for optimal GPU performance
- **PyTorch 2.5.1**: Pre-configured with CUDA 12.1 support for deep learning workloads
- **Flash Attention**: Optimized attention mechanism for improved performance
- **Runtime Installation**: All dependencies installed at container startup for faster builds
- **Runpod Optimized**: Designed specifically for Runpod's container execution environment

## Quick Start

### Runpod Deployment

1. **Create a new Pod/Serverless endpoint** on Runpod
2. **Use container image**: `gemneye/moda-runpod:latest`
3. **Select GPU**: Any CUDA-compatible GPU (RTX 3090, RTX 4090, A100, etc.)
4. **Configure environment variables** as needed for your specific use case

### Manual Docker Run

```bash
docker run --gpus all -p 8000:8000 gemneye/moda-runpod:latest
```

## Container Details

### Base Image
- `nvidia/cuda:12.1.0-cudnn8-devel-ubi8`

### Runtime Installation Process
1. PyTorch 2.5.1 installation with CUDA 12.1 support
2. MoDA repository cloning from GitHub
3. Requirements installation from requirements.txt
4. Flash Attention wheel installation (dynamically selected based on Python version)
5. Application startup

## Architecture Support

- **AMD64 only** (as required by Runpod platform)
- GPU acceleration via NVIDIA CUDA 12.1
- Optimized for cloud deployment scenarios

## Environment Variables

Configure these on the Runpod platform as needed:

- Standard Runpod environment variables are automatically available
- Additional application-specific variables can be configured in the Runpod interface

## Source Repository

Original MoDA project: [https://github.com/lixinyyang/MoDA.git](https://github.com/lixinyyang/MoDA.git)

Container repository: [https://github.com/sruckh/MoDA-Runpod](https://github.com/sruckh/MoDA-Runpod)

## License

This container follows the licensing terms of the original MoDA project.

## Support

For container-specific issues, please open an issue in the [MoDA-Runpod repository](https://github.com/sruckh/MoDA-Runpod/issues).

For MoDA application issues, refer to the [original repository](https://github.com/lixinyyang/MoDA.git).