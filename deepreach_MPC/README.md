# DeepReach_MPC
 Bridging Model Predictive Control and Deep  Learning for Scalable Reachability Analysis

Authors: [Zeyuan Feng](https://thezeyuanfeng.github.io/), Le Qiu, [Somil Bansal](http://people.eecs.berkeley.edu/~somil/index.html)

Acknowledgement: This repo is built on [DeepReach](https://github.com/smlbansal/deepreach). Thanks all the maintainers for the supports! <br>
[Albert Lin](https://www.linkedin.com/in/albertkuilin/),
[Zeyuan Feng](https://thezeyuanfeng.github.io/),
[Javier Borquez](https://javierborquez.github.io/),
[Somil Bansal](http://people.eecs.berkeley.edu/~somil/index.html)<br>

## High-Level Structure
The code is organized as follows:
* `dynamics/dynamics.py` defines the example dynamics of the system.
* `experiments/experiments.py` contains generic training routines.
* `utils/MPC.py` contains the MPC class for the different reachability cases.
* `utils/modules.py` contains neural network layers and modules.
* `utils/dataio.py` loads training and testing data.
* `utils/diff_operators.py` contains implementations of differential operators.
* `utils/losses.py` contains loss functions for the different reachability cases.
* `utils/error_evaluators.py` contains the helper functions for formal verification.
* `utils/quaternion.py` contains the helper functions for quaternion computation.

**This project (aa276 KOZ):** train via `python -m simulation.brt.train --force` from the repo root
(see `scripts/train_koz_brt_gpu.sh`). Upstream `run_experiment.py` was removed; training uses
`experiments/experiments.py` through `simulation/brt/deepreach_mpc_brt.py`.


## Environment Setup
Create and activate a virtual python environment (env) in the DeepReach_MPC folder to manage dependencies:
```
python -m venv env
```
Activate virtual environment
```
source env/bin/activate # Linux user
env\Scripts\activate # Windows user
```

Install DeepReach dependencies:
```
pip install -r requirements.txt
```
Install the appropriate PyTorch package for your system. For example, for a Windows system with CUDA 12.1:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## External Tutorial
Follow along these [tutorial slides](https://docs.google.com/presentation/d/1qLU4i1aBQR58G-FiyGb-l9IycMWoJlgq/edit?usp=sharing&ouid=112832011741826436488&rtpof=true&sd=true) to get started, or continue reading below. Currently the tutorial slides include the instruction for writing your own reachability problems, training the network for BRTs, and verifying the BRTs. More tutorials are coming soon.

## Running a KOZ BRT experiment (aa276)

From the repository root (with CUDA PyTorch installed):

```bash
./scripts/train_koz_brt_gpu.sh
# or: python -m simulation.brt.train --force
```

Checkpoints default to `simulation_output/deepreach_mpc_koz_v3/`.

## Upstream DeepReach examples (other dynamics)

The vendored upstream CLI (`run_experiment.py`) was removed from this fork. See the
[DeepReach-MPC tutorial slides](https://docs.google.com/presentation/d/1qLU4i1aBQR58G-FiyGb-l9IycMWoJlgq/edit?usp=sharing&ouid=112832011741826436488&rtpof=true&sd=true)
for VertDrone / Dubins / Quadrotor workflows on the original repo.


For any question, please feel free to raise an issue. 


## Coming soon
Forward Reachable Set computation and Reach-Avoid Problem solutions.