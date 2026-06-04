#!/usr/bin/env bash
# Install NVIDIA driver on a GCP GPU VM (Ubuntu 22.04/24.04).
# Run on the VM with sudo after the instance has a GPU attached.
#
#   chmod +x scripts/install_nvidia_driver_gcp.sh
#   sudo ./scripts/install_nvidia_driver_gcp.sh
#
# Then reboot if prompted, SSH back in, and run: nvidia-smi
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run with sudo:  sudo $0" >&2
  exit 1
fi

echo "=== GCP NVIDIA driver install (Ubuntu) ==="
echo "  kernel: $(uname -r)"
echo ""

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y curl ca-certificates build-essential "linux-headers-$(uname -r)"

# Google-maintained installer for Compute Engine GPU VMs (vanilla Ubuntu images).
echo "Running Google compute-gpu-installation installer…"
curl -fsSL https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py -o /tmp/install_gpu_driver.py
python3 /tmp/install_gpu_driver.py

echo ""
if command -v nvidia-smi &>/dev/null; then
  echo "=== nvidia-smi (before reboot — may need reboot to load kernel module) ==="
  nvidia-smi || true
else
  echo "nvidia-smi not in PATH yet; reboot often required."
fi

echo ""
echo "If nvidia-smi failed, run:  sudo reboot"
echo "After reboot, verify:      nvidia-smi"
echo "Then:                      ./scripts/setup_gcp_gpu_vm.sh"
