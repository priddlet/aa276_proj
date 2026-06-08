"""Verify a finished DeepReach-MPC KOZ training run (artifacts + optional load tests).

Inference loads best-by-loss checkpoint by default ('DEEPREACH_CHECKPOINT_SELECT=best'),
not the latest epoch. Use '--sync-final' to copy that weights file to 'model_final.pth'.

Usage:
  python -m simulation.brt.verify_training
  python -m simulation.brt.verify_training --sync-final
  python -m simulation.brt.verify_training --checkpoint-dir simulation_output/deepreach_mpc_koz_v3
  python -m simulation.brt.verify_training --epoch 98000
  DEEPREACH_CHECKPOINT_SELECT=latest python -m simulation.brt.verify_training
"""

from __future__ import annotations

import argparse
import math
import os
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


def _plot_loss_curve(
    epochs: np.ndarray,
    point_losses: np.ndarray,
    window_means: np.ndarray,
    out_path: Path,
    *,
    best_ep: int,
    infer_ep: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(
        epochs,
        np.maximum(point_losses, 1.0),
        ".",
        markersize=3,
        alpha=0.45,
        label="checkpoint step loss L[ep-1] (noisy)",
    )
    ax.semilogy(
        epochs,
        np.maximum(window_means, 1.0),
        "-",
        linewidth=1.2,
        label=f"last-{int(os.environ.get('DEEPREACH_LOSS_WINDOW', '200'))} step mean",
    )
    ax.axvline(best_ep, color="tab:green", linestyle="--", linewidth=1.0, alpha=0.8, label=f"best-by-loss ep {best_ep}")
    if infer_ep != best_ep:
        ax.axvline(infer_ep, color="tab:red", linestyle=":", linewidth=1.0, alpha=0.8, label=f"inference ep {infer_ep}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log)")
    ax.set_title("DeepReach-MPC training loss at checkpoints")
    ax.legend(loc="upper right", fontsize=8)
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

    if hasattr(brt, "koz_boundary_g"):
        g_in = float(np.asarray(brt.koz_boundary_g(inside[:3])).reshape(-1)[0])
    if g_in <= 0:
        for t_check in (0.0, T / 2.0, T):
            v_in = brt.value_at_tau(inside, t_check)
            ok = v_in <= 0.0
            lines.append(f"  KOZ origin @ t={t_check:.0f}: V={v_in:.3f}  {'OK' if ok else 'FAIL'} (expect <=0)")
        if os.environ.get("BRT_KOZ_PROJECT", "1").lower() in ("0", "false", "no"):
            lines.append("  (BRT_KOZ_PROJECT=0: raw network values, no inference projection)")

    if brt.value_at_tau(outside, T) <= 0:
        lines.append("WARN: LLM nominal start is BRT-unsafe at horizon (filter may reject burns)")
    else:
        lines.append("OK: LLM nominal start is BRT-safe at horizon (V>0)")

    if brt.value_at_tau(inside, 0.0) > 0.5:
        lines.append("WARN: V(origin,t=0) not near KOZ boundary (exact BC may be off)")
    return lines


def _passive_rest_unsafe(
    plant,
    inner,
    y_m: float,
    horizon_s: float,
) -> bool:
    from simulation.sampling.passive import natural_coast_hits_inner_koz

    state = np.array([0.0, float(y_m), 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return bool(
        natural_coast_hits_inner_koz(plant, state, inner, float(horizon_s), n_samples=128)
    )


def _grid_vs_network_report(
    brt,
    plant,
    semi_axes: tuple[float, float, float],
    horizon_s: float,
) -> list[str]:
    """Compare passive KOZ reach vs learned V along along-track rest positions."""
    from simulation.cw_dynamics import CWDynamics
    from simulation.keepout import EllipsoidKeepOut

    inner = EllipsoidKeepOut(np.array(semi_axes, dtype=np.float64))
    if not isinstance(plant, CWDynamics):
        plant = CWDynamics(float(getattr(brt, "n_rad_s", plant)))

    lines = ["", "=== Passive grid vs learned V (rest states, x=z=v=0) ==="]
    lines.append(f"{'y_m':>6}  {'passive':>8}  {'V@0':>8}  {'V@T':>8}  {'match':>6}")
    ys = [35.0, 45.0, 55.0, 70.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0]
    mism = 0
    for y in ys:
        state = np.array([0.0, y, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        passive = _passive_rest_unsafe(plant, inner, y, horizon_s)
        v0 = float(brt.value_at_tau(state, 0.0))
        vT = float(brt.value_at_tau(state, horizon_s))
        brt_unsafe = vT <= 0.0
        ok = passive == brt_unsafe
        if not ok:
            mism += 1
        lines.append(
            f"{y:6.0f}  {str(passive):>8}  {v0:8.3f}  {vT:8.3f}  {'OK' if ok else 'DIFF':>6}"
        )
    lines.append(f"Agreement @ t=T: {len(ys) - mism}/{len(ys)} samples")
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
    dynamics = build_dynamics(config, device=dev)
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
    p.add_argument(
        "--sync-final",
        action="store_true",
        help="Copy best-by-loss epoch checkpoint to model_final.pth for legacy loaders.",
    )
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
    from simulation.brt.training_metrics import (
        best_epoch_from_losses,
        collect_epoch_losses,
        resolve_inference_checkpoint,
        sync_model_final_from_best,
    )

    epochs, point, wmean = collect_epoch_losses(ckpt_sub)
    if len(epochs) == 0:
        print("No train_losses_epoch_*.txt files found.")
    else:
        out_png = checkpoint_dir / "training" / "verify_training_loss.png"
        best_ep, best_loss = best_epoch_from_losses(epochs, wmean)
        best_pt_ep, best_pt = best_epoch_from_losses(epochs, point)
        try:
            _, infer_ep, infer_reason = resolve_inference_checkpoint(checkpoint_dir, ckpt_sub=ckpt_sub)
        except FileNotFoundError:
            infer_ep = int(epochs[-1])
            infer_reason = "latest (fallback)"
        _plot_loss_curve(epochs, point, wmean, out_png, best_ep=best_ep, infer_ep=infer_ep)
        print(f"Wrote {out_png}")
        print(f"Best window-mean loss: {best_loss:.4e} at epoch {best_ep}")
        print(f"Best single-step loss: {best_pt:.4e} at epoch {best_pt_ep}")
        print(f"Latest logged checkpoint: epoch {epochs[-1]}, step loss={point[-1]:.4e}")
        print(f"Inference selection ({os.environ.get('DEEPREACH_CHECKPOINT_SELECT', 'best')}): {infer_reason}")
        spikes = epochs[point > 5e5]
        if len(spikes):
            print(f"Spike checkpoints (step loss > 5e5): {spikes.tolist()}")

    if args.sync_final and len(epochs) > 0:
        dst = sync_model_final_from_best(checkpoint_dir)
        print(f"Synced model_final.pth from best-by-loss: {dst}")

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

    from simulation.cw_dynamics import CWDynamics

    plant = CWDynamics(leo.n_rad_s)
    for line in _grid_vs_network_report(brt, plant, semi_axes, brt.horizon_s):
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
