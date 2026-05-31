"""Train DeepReach KOZ BRT from CLI: ``python -m simulation.brt.train``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from simulation.brt.config import BRT_HORIZON_S, DeepReachTrainConfig, KozBRTConfig
from simulation.brt.deepreach_brt import (
    DEEPREACH_AVAILABLE,
    _latest_epoch_checkpoint,
    default_checkpoint_dir,
    train_koz_deepreach,
)
from simulation.cw_dynamics import leo_circular_orbit


def main() -> None:
    if not DEEPREACH_AVAILABLE:
        print("DeepReach requires torch. Install: pip install -r requirements-deepreach.txt", file=sys.stderr)
        sys.exit(1)

    p = argparse.ArgumentParser(description="Train DeepReach 6D CW KOZ BRT")
    p.add_argument("--altitude-km", type=float, default=float(os.environ.get("LEO_ALTITUDE_KM", "400")))
    p.add_argument("--checkpoint-dir", type=str, default=str(default_checkpoint_dir()))
    p.add_argument(
        "--epochs",
        type=int,
        default=int(os.environ.get("DEEPREACH_EPOCHS", "8000")),
        help="Total epoch target (not extra epochs when resuming).",
    )
    p.add_argument("--device", type=str, default=os.environ.get("DEEPREACH_DEVICE", "auto"))
    p.add_argument("--force", action="store_true", help="Delete checkpoint dir and train from scratch.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue from latest model_epoch_*.pth (or model_current.pth).",
    )
    p.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Explicit checkpoint .pth (e.g. training/checkpoints/model_epoch_2000.pth).",
    )
    p.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="Override resume epoch when using model_current.pth (no epoch in filename).",
    )
    p.add_argument("--semi-axes", type=str, default=os.environ.get("KOZ_INNER_SEMIAXES_M", "28,45,18"))
    p.add_argument(
        "--resume-from-v2-epoch2000",
        action="store_true",
        help="Resume PDE training from v2 model_epoch_2000.pth into the default v3 checkpoint dir (no CSL).",
    )
    args = p.parse_args()

    resume_from = args.resume_from.strip() or None
    start_epoch = args.start_epoch
    resume = args.resume
    if args.resume_from_v2_epoch2000:
        root = Path(__file__).resolve().parents[2]
        v2_ckpt = root / "simulation_output" / "deepreach_koz_v2" / "training" / "checkpoints" / "model_epoch_2000.pth"
        if not v2_ckpt.is_file():
            print(f"v2 checkpoint not found: {v2_ckpt}", file=sys.stderr)
            sys.exit(1)
        resume_from = str(v2_ckpt)
        start_epoch = 2000 if start_epoch is None else start_epoch
        resume = True
        if args.checkpoint_dir == str(default_checkpoint_dir()):
            print(f"Retrain (no CSL) → {default_checkpoint_dir()} from {v2_ckpt.name}")

    ck_dir = args.checkpoint_dir
    if resume:
        if resume_from:
            print(f"Will resume from {Path(resume_from).name} (epoch {start_epoch or '?'})")
        else:
            path, ep = _latest_epoch_checkpoint(Path(ck_dir))
            if path is not None:
                print(f"Will resume from {path.name} (epoch {ep or start_epoch or '?'})")

    leo = leo_circular_orbit(args.altitude_km)
    axes = tuple(float(x) for x in args.semi_axes.split(","))

    device = os.environ.get("DEEPREACH_DEVICE", args.device)
    train_cfg = DeepReachTrainConfig(
        device=device,
        num_epochs=args.epochs,
        counter_end=args.epochs,
    )
    config = KozBRTConfig(
        n_rad_s=leo.n_rad_s,
        semi_axes_m=axes,
        horizon_s=float(os.environ.get("BRT_HORIZON_S", str(BRT_HORIZON_S))),
        train=train_cfg,
    )
    train_koz_deepreach(
        config,
        ck_dir,
        force=args.force,
        resume=resume,
        resume_from=resume_from,
        start_epoch=start_epoch,
    )
    print(f"Done. Checkpoint dir: {ck_dir}")


if __name__ == "__main__":
    main()
