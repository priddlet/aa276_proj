"""Train DeepReach-MPC KOZ BRT: ``python -m simulation.brt.train --force``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from simulation.brt.config import BRT_HORIZON_S, U_MAX_M_S2, DeepReachTrainConfig, KozBRTConfig
from simulation.brt.deepreach_mpc_brt import (
    DEEPREACH_MPC_AVAILABLE,
    DEEPREACH_MPC_IMPORT_ERROR,
    default_checkpoint_dir,
    train_koz_deepreach_mpc,
)
from simulation.cw_dynamics import leo_circular_orbit


def main() -> None:
    if not DEEPREACH_MPC_AVAILABLE:
        print(
            "DeepReach-MPC import failed:\n"
            "  pip install torch -r requirements-deepreach.txt\n"
            f"  ({DEEPREACH_MPC_IMPORT_ERROR})",
            file=sys.stderr,
        )
        sys.exit(1)

    p = argparse.ArgumentParser(description="Train DeepReach-MPC 6D CW KOZ avoid-BRT.")
    p.add_argument("--altitude-km", type=float, default=float(os.environ.get("LEO_ALTITUDE_KM", "400")))
    p.add_argument("--checkpoint-dir", type=str, default=str(default_checkpoint_dir()))
    p.add_argument(
        "--epochs",
        type=int,
        default=int(os.environ.get("DEEPREACH_EPOCHS", "100000")),
    )
    p.add_argument(
        "--counter-end",
        type=int,
        default=None,
        help="Curriculum end epoch (default: same as --epochs).",
    )
    p.add_argument("--pretrain-iters", type=int, default=1000)
    p.add_argument("--num-target-samples", type=int, default=8000)
    p.add_argument("--device", type=str, default=os.environ.get("DEEPREACH_DEVICE", "auto"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--semi-axes", type=str, default=os.environ.get("KOZ_INNER_SEMIAXES_M", "28,45,18"))
    p.add_argument("--u-max", type=float, default=None)
    args = p.parse_args()

    leo = leo_circular_orbit(args.altitude_km)
    axes = tuple(float(x) for x in args.semi_axes.split(","))
    counter_end = args.counter_end
    if counter_end is None:
        counter_end = int(os.environ.get("DEEPREACH_COUNTER_END", str(args.epochs)))
    u_max = args.u_max
    if u_max is None:
        u_max = float(os.environ.get("BRT_U_MAX_M_S2", str(U_MAX_M_S2)))

    train_cfg = DeepReachTrainConfig(
        device=args.device,
        num_epochs=args.epochs,
        counter_end=counter_end,
        pretrain_iters=args.pretrain_iters,
        num_target_samples=args.num_target_samples,
        deepreach_model="exact",
        hidden_features=512,
    )
    config = KozBRTConfig(
        n_rad_s=leo.n_rad_s,
        semi_axes_m=axes,
        horizon_s=float(os.environ.get("BRT_HORIZON_S", str(BRT_HORIZON_S))),
        u_max_m_s2=u_max,
        train=train_cfg,
    )
    train_koz_deepreach_mpc(config, Path(args.checkpoint_dir), force=args.force)
    print(f"Done. Checkpoint dir: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
