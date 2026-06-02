"""Verify a finished DeepReach-MPC KOZ training run (artifacts + optional load tests).

Usage:
  python -m simulation.brt.verify_training
  python -m simulation.brt.verify_training --checkpoint-dir simulation_output/deepreach_mpc_koz
  python -m simulation.brt.verify_training --epoch 98000 --regenerate-plot
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from simulation.brt.config import BRT_HORIZON_S, U_MAX_M_S2, DeepReachTrainConfig, KozBRTConfig
from simulation.brt.deepreach_mpc_brt import (
    DEEPREACH_MPC_AVAILABLE,
    DEEPREACH_MPC_IMPORT_ERROR,
    default_checkpoint_dir,
    load_brt_config,
    save_brt_config,
)


def _ellipsoid_boundary(xyz: np.ndarray, semi_axes: tuple[float, float, float]) -> float:
    """Signed distance style: <0 inside KOZ, >0 outside (matches Cw6DKoz.boundary_fn)."""
    r = np.asarray(xyz, dtype=np.float64).reshape(3)
    ax, ay, az = semi_axes
    s = (r[0] / ax) ** 2 + (r[1] / ay) ** 2 + (r[2] / az) ** 2
    return float(math.sqrt(s + 1e-18) - 1.0)


def _collect_epoch_losses(ckpt_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-epoch training loss from cumulative ``train_losses_epoch_*.txt`` files."""
    files = sorted(
        ckpt_dir.glob("train_losses_epoch_*.txt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    epochs: list[int] = []
    losses: list[float] = []
    for f in files:
        ep = int(f.stem.rsplit("_", 1)[-1])
        L = np.loadtxt(f)
        if len(L) < ep:
            continue
        epochs.append(ep)
        losses.append(float(L[ep - 1]))
    return np.asarray(epochs, dtype=np.int64), np.asarray(losses, dtype=np.float64)


def _plot_loss_curve(epochs: np.ndarray, losses: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(epochs, np.maximum(losses, 1.0), ".-", markersize=3, linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("epoch loss (log)")
    ax.set_title("DeepReach-MPC training loss (end-of-epoch)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _artifact_report(checkpoint_dir: Path) -> list[str]:
    lines: list[str] = []
    ckpt = checkpoint_dir / "training" / "checkpoints"
    if not ckpt.is_dir():
        lines.append(f"FAIL: missing {ckpt}")
        return lines

    finals = list(ckpt.glob("model_epoch_*.pth"))
    lines.append(f"OK: {len(finals)} epoch checkpoints under training/checkpoints/")
    best_ep = 0
    for p in finals:
        try:
            best_ep = max(best_ep, int(p.stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    lines.append(f"    latest epoch checkpoint: {best_ep}")

    for name in ("model_final.pth",):
        p = ckpt / name
        lines.append(f"{'OK' if p.is_file() else 'WARN'}: {p.name} {'present' if p.is_file() else 'missing'}")

    val_plots = sorted(ckpt.glob("BRS_validation_plot_epoch_*.png"))
    lines.append(f"OK: {len(val_plots)} BRS validation PNGs")
    if val_plots:
        lines.append(f"    newest: {val_plots[-1].name}")

    cfg = checkpoint_dir / "koz_brt_config.json"
    lines.append(f"{'OK' if cfg.is_file() else 'INFO'}: koz_brt_config.json")
    return lines


def _numeric_checks(brt, semi_axes: tuple[float, float, float], horizon_s: float) -> list[str]:
    lines: list[str] = []
    T = horizon_s

    inside = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    outside = np.array([0.0, 3200.0, 0.0, 0.0, 0.0, 0.0])
    near_surface = np.array([semi_axes[0] * 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])

    g_in = _ellipsoid_boundary(inside[:3], semi_axes)
    g_out = _ellipsoid_boundary(outside[:3], semi_axes)
    lines.append(f"exact boundary g(origin)={g_in:.4f} (expect <0), g(y=3200)={g_out:.4f} (expect >0)")

    for label, state in (
        ("origin", inside),
        ("LLM start y=3200", outside),
        ("near KOZ +x", near_surface),
    ):
        v0 = brt.value_at_tau(state, 0.0)
        vT = brt.value_at_tau(state, T)
        lines.append(f"  {label}: V(t=0)={v0:.3f}, V(t=T)={vT:.3f}, unsafe@T={vT <= 0}")

    if brt.value_at_tau(outside, T) <= 0:
        lines.append("WARN: LLM nominal start is BRT-unsafe at horizon (filter may reject burns)")
    else:
        lines.append("OK: LLM nominal start is BRT-safe at horizon (V>0)")

    if brt.value_at_tau(inside, 0.0) > 0.5:
        lines.append("WARN: V(origin,t=0) not near KOZ boundary (exact BC may be off)")
    return lines


def _regenerate_validation(checkpoint_dir: Path, epoch: int, device: str) -> Path:
    if not DEEPREACH_MPC_AVAILABLE:
        raise RuntimeError(DEEPREACH_MPC_IMPORT_ERROR or "torch not installed")

    import os

    import torch

    from simulation.brt.deepreach_mpc_brt import (
        KozDeepReachBRT,
        _deepreach_mpc_imports,
        build_dynamics,
        build_model,
    )

    config = load_brt_config(checkpoint_dir)
    ckpt = checkpoint_dir / "training" / "checkpoints" / f"model_epoch_{epoch:05d}.pth"
    if not ckpt.is_file():
        ckpt = checkpoint_dir / "training" / "checkpoints" / "model_final.pth"

    dev = device
    dynamics = build_dynamics(config)
    model = build_model(dynamics, config.train)
    state = torch.load(ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.to(dev)

    prev = os.getcwd()
    os.chdir(checkpoint_dir)
    try:
        with _deepreach_mpc_imports():
            from utils import dataio  # noqa: WPS433
            from experiments import experiments as dr_experiments  # noqa: WPS433

            dataset = dataio.ReachabilityDataset(
                dynamics=dynamics,
                numpoints=config.train.numpoints,
                pretrain=False,
                pretrain_iters=config.train.pretrain_iters,
                tMin=0.0,
                tMax=float(config.horizon_s),
                counter_start=0,
                counter_end=config.train.counter_end,
                num_src_samples=config.train.num_src_samples,
                num_target_samples=0,
                use_MPC=False,
                time_curr=config.train.time_curr,
                MPC_data_path="none",
                num_MPC_perturbation_samples=0,
                MPC_dt=config.train.MPC_dt,
                MPC_mode="MPC",
                MPC_sample_mode="gaussian",
                MPC_style="direct",
                MPC_lambda_=0.1,
                MPC_batch_size=1,
                MPC_receding_horizon=-1,
                num_MPC_data_samples=0,
                num_iterative_refinement=0,
                time_till_refinement=config.horizon_s / 10.0,
                num_MPC_batches=0,
                aug_with_MPC_data=0,
                policy=None,
                refine_dataset=False,
            )
            exp = dr_experiments.DeepReach(
                model=model,
                dataset=dataset,
                experiment_dir=str(checkpoint_dir),
                use_wandb=False,
            )
            out = (
                checkpoint_dir
                / "training"
                / "checkpoints"
                / f"BRS_validation_verify_epoch_{epoch:05d}.png"
            )
            exp.validate(
                epoch=epoch,
                save_path=str(out),
                x_resolution=120,
                y_resolution=120,
                z_resolution=3,
                time_resolution=3,
            )
            return out
    finally:
        os.chdir(prev)


def main() -> None:
    p = argparse.ArgumentParser(description="Verify DeepReach-MPC KOZ training artifacts.")
    p.add_argument("--checkpoint-dir", type=str, default=str(default_checkpoint_dir()))
    p.add_argument("--altitude-km", type=float, default=400.0)
    p.add_argument("--semi-axes", type=str, default="28,45,18")
    p.add_argument("--epoch", type=int, default=0, help="Load this epoch (0 = latest/final).")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--regenerate-plot", action="store_true")
    p.add_argument("--skip-load", action="store_true", help="Only scan artifacts and loss curve.")
    args = p.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    ckpt_sub = checkpoint_dir / "training" / "checkpoints"
    semi_axes = tuple(float(x) for x in args.semi_axes.split(","))

    print(f"Checkpoint dir: {checkpoint_dir}\n")
    print("=== Artifacts ===")
    for line in _artifact_report(checkpoint_dir):
        print(line)

    print("\n=== Loss curve ===")
    epochs, losses = _collect_epoch_losses(ckpt_sub)
    if len(epochs) == 0:
        print("No train_losses_epoch_*.txt files found.")
    else:
        out_png = checkpoint_dir / "training" / "verify_training_loss.png"
        _plot_loss_curve(epochs, losses, out_png)
        print(f"Wrote {out_png}")
        best_i = int(np.argmin(losses))
        print(f"Lowest logged epoch loss: {losses[best_i]:.4e} at epoch {epochs[best_i]}")
        print(f"Final checkpoint epoch: {epochs[-1]}, loss={losses[-1]:.4e}")
        spikes = epochs[losses > 5e5]
        if len(spikes):
            print(f"Spike epochs (loss > 5e5): {spikes.tolist()}")

    if args.skip_load:
        print("\n(skip-load) Install torch and re-run without --skip-load for V(x,t) checks.")
        return

    if not DEEPREACH_MPC_AVAILABLE:
        print(
            f"\nNumeric checks skipped: {DEEPREACH_MPC_IMPORT_ERROR}\n"
            "  pip install -r requirements-deepreach.txt",
            file=sys.stderr,
        )
        sys.exit(0)

    from simulation.brt.deepreach_mpc_brt import load_or_train_koz_brt
    from simulation.cw_dynamics import leo_circular_orbit

    leo = leo_circular_orbit(args.altitude_km)
    cfg_path = checkpoint_dir / "koz_brt_config.json"
    if not cfg_path.is_file():
        tc = DeepReachTrainConfig(device=args.device)
        cfg = KozBRTConfig(
            n_rad_s=leo.n_rad_s,
            semi_axes_m=semi_axes,
            horizon_s=BRT_HORIZON_S,
            u_max_m_s2=U_MAX_M_S2,
            train=tc,
        )
        save_brt_config(checkpoint_dir, cfg)
        print(f"\nWrote default {cfg_path}")

    if args.epoch > 0:
        import os
        import shutil

        src = ckpt_sub / f"model_epoch_{args.epoch:05d}.pth"
        if not src.is_file():
            print(f"Missing {src}", file=sys.stderr)
            sys.exit(1)
        dst = ckpt_sub / "model_final.pth"
        shutil.copy2(src, dst)
        print(f"\nUsing epoch {args.epoch} via {dst.name}")

    print("\n=== Load + value checks ===")
    brt, was_loaded = load_or_train_koz_brt(
        leo.n_rad_s,
        semi_axes_m=semi_axes,
        checkpoint_dir=checkpoint_dir,
        train_config=DeepReachTrainConfig(device=args.device),
        force_train=False,
    )
    assert was_loaded
    for line in _numeric_checks(brt, semi_axes, brt.horizon_s):
        print(line)

    if args.regenerate_plot:
        ep = args.epoch if args.epoch > 0 else int(
            max(
                (int(p.stem.rsplit("_", 1)[-1]) for p in ckpt_sub.glob("model_epoch_*.pth")),
                default=0,
            )
        )
        out = _regenerate_validation(checkpoint_dir, ep, brt._device)
        print(f"\nRegenerated validation plot: {out}")


if __name__ == "__main__":
    main()
