# New GCP GPU VM (Ubuntu + manual NVIDIA driver)

Use a **plain Ubuntu** VM with a GPU attached, install the driver manually, then train v3.

## 1. Create the VM (Cloud Console)

1. **Compute Engine → Create instance**
2. **Name:** `proj-gpu-v3` (or any name)
3. **Region / zone:** where you have GPU quota (e.g. `asia-east1-b`)
4. **Machine configuration → GPUs**
   - Enable **GPU**
   - Type: **NVIDIA T4** (or L4 on G2 if available)
   - Count: **1**
   - Machine type example: **N1 → n1-standard-8** (8 vCPU, 30 GB RAM)
5. **OS and storage**
   - **Operating system:** Ubuntu
   - **Version:** Ubuntu 22.04 LTS (or 24.04 LTS)
   - **Boot disk:** **100 GB** minimum (Standard persistent disk is fine)
6. **Networking:** allow SSH
7. **Create** → **SSH** (browser or `gcloud compute ssh`)

Do **not** skip attaching a GPU — driver install only works if the hardware is on the VM config.

## 2. Install NVIDIA driver (manual, on the VM)

```bash
# optional: clone repo first so the script is available
cd ~
git clone https://github.com/priddlet/aa276_proj.git
cd aa276_proj
chmod +x scripts/install_nvidia_driver_gcp.sh

sudo ./scripts/install_nvidia_driver_gcp.sh
sudo reboot
```

SSH in again:

```bash
nvidia-smi
```

You must see GPU name, driver version, and memory. If this fails, check **Compute Engine → VM → Edit** → GPU is still attached.

### Alternative driver install (if Google script fails)

```bash
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r) build-essential
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

## 3. CUDA note (PyTorch)

Full **CUDA toolkit** on the system is **optional**. Training uses **pip PyTorch** with bundled CUDA libraries:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

You need a working **kernel driver** (`nvidia-smi`), not a separate `nvcc` install.

Optional toolkit (only if you want `nvcc`):

```bash
# Ubuntu 22.04 example — match driver/CUDA docs if you need nvcc
sudo apt-get install -y nvidia-cuda-toolkit
nvcc --version
```

## 4. Project setup (conda + deps)

```bash
cd ~/aa276_proj   # or git clone again after reboot
git pull
chmod +x scripts/setup_gcp_gpu_vm.sh scripts/train_koz_brt_gpu.sh
./scripts/setup_gcp_gpu_vm.sh
```

Creates conda env **`proj`**, installs CUDA PyTorch + `requirements-deepreach.txt`.

## 5. Train v3

```bash
cd ~/aa276_proj
export CONDA_ENV=proj
./scripts/train_koz_brt_gpu.sh --background
tail -f simulation_output/deepreach_mpc_koz_v3/train.log
```

| Setting | Value |
|---------|--------|
| Checkpoints | `simulation_output/deepreach_mpc_koz_v3/` |
| Epochs | 100000 |
| `counter_end` | 75000 |

## 6. gcloud: create Ubuntu + T4 VM

Replace `PROJECT`, `ZONE`, and ensure GPU quota:

```bash
gcloud compute instances create proj-gpu-v3 \
  --project=YOUR_PROJECT_ID \
  --zone=asia-east1-b \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --provisioning-model=STANDARD \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced

gcloud compute ssh proj-gpu-v3 --zone=asia-east1-b
```

Then run **§2–§5** on the VM.

## 7. Copy v2 checkpoints (optional)

```bash
# from old VM
tar czf /tmp/koz_v2.tgz -C ~/aa276_proj simulation_output/deepreach_mpc_koz_v2
# scp to new VM, extract under ~/aa276_proj/simulation_output/
```

## Checklist

- [ ] VM has **1 GPU** in console
- [ ] `nvidia-smi` works after reboot
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → `True`
- [ ] `./scripts/train_koz_brt_gpu.sh` starts without CUDA errors
