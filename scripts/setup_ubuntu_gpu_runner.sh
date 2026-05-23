#!/bin/bash
# ------------------------------------------------------------------------------
# AdaptiveGEMM - Self-Hosted Runner Provisioning Script
# OS: Ubuntu 22.04 LTS
# Target: NVIDIA GPU (Ada / RTX 40-series recommended)
# ------------------------------------------------------------------------------
# WARNING: This script installs NVIDIA drivers and Docker/Runner services.
# It is recommended to review the script and run it step-by-step or on a
# dedicated CI machine. Do not run blindly on your daily workstation.
# ------------------------------------------------------------------------------

set -e

echo "=== [1/5] Updating system and installing prerequisites ==="
sudo apt-get update
sudo apt-get install -y \
    curl wget git build-essential dkms ninja-build cmake python3 python3-pip jq

echo "=== [2/5] Installing NVIDIA Driver & CUDA Toolkit 12.6 ==="
# Note: You can skip this step if drivers are already installed.
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found. Installing NVIDIA CUDA repository..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-12-6 nvidia-driver-550
    echo "NVIDIA Driver installed. A reboot is highly recommended after this script finishes."
else
    echo "NVIDIA Driver is already installed."
    nvidia-smi
fi

echo "=== [3/5] Setting up CUDA Environment Variables ==="
if ! grep -q "/usr/local/cuda/bin" ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi

echo "=== [4/5] Downloading GitHub Actions Runner ==="
RUNNER_DIR="${HOME}/actions-runner"
if [ ! -d "$RUNNER_DIR" ]; then
    mkdir -p "$RUNNER_DIR"
    cd "$RUNNER_DIR"
    
    # Fetch latest runner version programmatically
    LATEST_RUNNER_VERSION=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | jq -r '.tag_name' | sed 's/^v//')
    if [ -z "$LATEST_RUNNER_VERSION" ] || [ "$LATEST_RUNNER_VERSION" == "null" ]; then
        LATEST_RUNNER_VERSION="2.316.1" # Fallback to a recent version
    fi
    
    echo "Downloading runner version $LATEST_RUNNER_VERSION..."
    curl -o actions-runner-linux-x64.tar.gz -L "https://github.com/actions/runner/releases/download/v${LATEST_RUNNER_VERSION}/actions-runner-linux-x64-${LATEST_RUNNER_VERSION}.tar.gz"
    tar xzf ./actions-runner-linux-x64.tar.gz
    rm actions-runner-linux-x64.tar.gz
else
    echo "Runner directory already exists at $RUNNER_DIR"
    cd "$RUNNER_DIR"
fi

echo "=== [5/5] Registration Instructions ==="
echo ""
echo "To complete the setup, please run the following commands manually inside $RUNNER_DIR :"
echo ""
echo "1. Register the runner to your repository:"
echo "   ./config.sh --url https://github.com/YOUR_ORG/YOUR_REPO --token YOUR_GITHUB_TOKEN --labels self-hosted,linux,gpu"
echo ""
echo "2. Install and start the service:"
echo "   sudo ./svc.sh install"
echo "   sudo ./svc.sh start"
echo ""
echo "Please replace YOUR_ORG, YOUR_REPO, and YOUR_GITHUB_TOKEN with the values from your GitHub Repository -> Settings -> Actions -> Runners page."
echo "Also, please REBOOT the machine if the NVIDIA driver was just installed."
