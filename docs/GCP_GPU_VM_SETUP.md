# New GCP GPU VM (Ubuntu + manual NVIDIA driver)

Fresh VM setup when you use **plain Ubuntu** and install the GPU driver yourself (no Deep Learning marketplace image).

## 1. Create the VM (Cloud Console)

| Setting | Value |
|---------|--------|
| **Name** | `proj-gpu-v3` (or any name) |
| **Region / zone** | Where you have GPU quota (e.g. `asia-east1-b`) |
| **Machine type** | GPU: e.g. `n1-standard-8` + **1× NVIDIA T4**, or `g2-standard-8` + **1× L4** |
| **OS image** | **Ubuntu 22.04 LTS** (standard, not “Deep Learning”) |
| **Boot disk** | **≥ 100 GB** |
| **GPU** | Must show under machine config **before** first boot |

Create the instance → **SSH** (browser SSH or `gcloud compute ssh`).

## 2. Install NVIDIA driver (manual)

On the VM, after clone **or** before clone (driver install does not need the repo).

```bash
# Option A — from repo (after clone)
cd ~/aa276_proj
chmod +x scripts/install_nvidia_driver_ubuntu.sh
sudo ./scripts/install_nvidia_driver_ubuntu.sh
sudo reboot
```

```bash
# Option B — one-shot without repo yet
sudo apt-get update -y
sudo apt-get install -y linux-headers-$(uname -r) dkms ubuntu-drivers-common nvidia-driver-535
sudo reboot
```

After reboot:

```bash
nvidia-smi
```

You must see GPU name, driver version, and memory. If this fails:

- Stop VM → Edit → confirm **GPU** is attached → Start
- Ensure image is **Ubuntu 22.04** x86_64 (not ARM)
- Try `sudo ubuntu-drivers install` then reboot

**Note:** You do **not** need the full CUDA toolkit on the system for training. PyTorch wheels include the CUDA runtime. Only the **driver** + `nvidia-smi` are required.

## 3. Clone repo and Python env

```bash
cd ~
git clone https://github.com/priddlet/aa276_proj.git
cd aa276_proj
git pull   # ensure train/setup scripts are present

chmod +x scripts/install_nvidia_driver_ubuntu.sh
chmod +x scripts/setup_gcp_gpu_vm.sh scripts/train_koz_brt_gpu.sh

./scripts/setup_gcp_gpu_vm.sh
```

This installs Miniconda (if needed), env **`proj`**, **PyTorch cu124**, and project dependencies.

Verify:

```bash
conda activate proj
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. Train v3

```bash
cd ~/aa276_proj
export CONDA_ENV=proj
./scripts/train_koz_brt_gpu.sh --background
tail -f simulation_output/deepreach_mpc_koz_v3/train.log
```

| Setting | Default |
|---------|---------|
| Output | `simulation_output/deepreach_mpc_koz_v3/` |
| Epochs | 100000 |
| `counter_end` | 75000 |

## 5. gcloud: create Ubuntu + T4 (optional)

```bash
gcloud compute instances create proj-gpu-v3 \
  --zone=asia-east1-b \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced

gcloud compute ssh proj-gpu-v3 --zone=asia-east1-b
```

Then run **§2 driver install** → reboot → **§3–4**.

## 6. Copy old v2 checkpoints (optional)

```bash
# old VM
tar -czf /tmp/koz_v2.tgz -C ~/aa276_proj simulation_output/deepreach_mpc_koz_v2

# laptop
gcloud compute scp OLD_VM:/tmp/koz_v2.tgz . --zone=OLD_ZONE
gcloud compute scp koz_v2.tgz NEW_VM:~/ --zone=NEW_ZONE

# new VM
mkdir -p ~/aa276_proj/simulation_output
tar -xzf ~/koz_v2.tgz -C ~/aa276_proj/simulation_output
```

## Checklist

- [ ] VM has GPU in machine config  
- [ ] `nvidia-smi` works after driver + reboot  
- [ ] `torch.cuda.is_available()` is True in env `proj`  
- [ ] Training log: `simulation_output/deepreach_mpc_koz_v3/train.log`  

## After training (laptop / CPU)

```bash
export DEEPREACH_AUTO_TRAIN=0
export DEEPREACH_CHECKPOINT_DIR=/path/to/deepreach_mpc_koz_v3
```
