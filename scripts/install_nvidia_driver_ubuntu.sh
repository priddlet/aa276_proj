#!/usr/bin/env bash
# Install NVIDIA driver on Ubuntu 22.04/24.04 GCP VM (GPU must already be attached).
# Run BEFORE setup_gcp_gpu_vm.sh if nvidia-smi fails on a plain Ubuntu image.
#
#   chmod +x scripts/install_nvidia_driver_ubuntu.sh
#   sudo ./scripts/install_nvidia_driver_ubuntu.sh
#   sudo reboot
#   nvidia-smi
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run with sudo:  sudo $0" >&2
  exit 1
fi

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  echo "nvidia-smi already works:"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  echo "No install needed. Next: ./scripts/setup_gcp_gpu_vm.sh"
  exit 0
fi

if ! [[ -f /etc/os-release ]]; then
  echo "Unsupported OS (expected Ubuntu)." >&2
  exit 1
fi
# shellcheck source=/dev/null
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "This script targets Ubuntu; found ID=$ID" >&2
  exit 1
fi

DRIVER_PKG="${NVIDIA_DRIVER_PKG:-nvidia-driver-535}"

echo "=== NVIDIA driver install (Ubuntu ${VERSION_ID:-?}) ==="
echo "  package: $DRIVER_PKG"
echo "  (override: NVIDIA_DRIVER_PKG=nvidia-driver-550 sudo $0)"
echo ""

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y \
  build-essential \
  "linux-headers-$(uname -r)" \
  dkms \
  ubuntu-drivers-common \
  "$DRIVER_PKG"

echo ""
echo "=== Blacklist nouveau (if present) ==="
if lsmod | grep -q nouveau; then
  cat >/etc/modprobe.d/blacklist-nouveau.conf <<'EOF'
blacklist nouveau
options nouveau modeset=0
EOF
  update-initramfs -u
fi

echo ""
echo "Driver package installed."
echo "REBOOT required, then verify:"
echo "  sudo reboot"
echo "  nvidia-smi"
echo "  cd ~/aa276_proj && ./scripts/setup_gcp_gpu_vm.sh"
