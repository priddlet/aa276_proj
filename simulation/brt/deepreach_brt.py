"""DeepReach neural BRT for 6D CW KOZ collision (replaces hj_reachability grid)."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from simulation.brt.config import (
    BRT_HORIZON_S,
    DOMAIN_HI,
    DOMAIN_LO,
    KozBRTConfig,
    DeepReachTrainConfig,
)

_ROOT = Path(__file__).resolve().parents[2]
_DEEPREACH = _ROOT / "deepreach"
if _DEEPREACH.is_dir() and str(_DEEPREACH) not in sys.path:
    sys.path.insert(0, str(_DEEPREACH))

DEEPREACH_AVAILABLE = False
DEEPREACH_IMPORT_ERROR: str | None = None
try:
    import torch
    from dynamics import dynamics as dr_dynamics  # noqa: E402
    from utils import modules  # noqa: E402

    DEEPREACH_AVAILABLE = True
except ImportError as exc:
    torch = None  # type: ignore[assignment]
    dr_dynamics = None  # type: ignore[assignment]
    modules = None  # type: ignore[assignment]
    DEEPREACH_IMPORT_ERROR = str(exc)


def default_checkpoint_dir(project_root: str | Path | None = None) -> Path:
    root = Path(project_root).resolve() if project_root else _ROOT
    env = os.environ.get("DEEPREACH_CHECKPOINT_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (root / "simulation_output" / "deepreach_koz_v3").resolve()


def _resolve_device(requested: str) -> str:
    """Pick cuda/mps when available; ``auto`` tries accelerators then cpu."""
    req = (requested or "cpu").strip().lower()
    if req.startswith("cuda"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req in ("auto", "default"):
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def _config_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "koz_brt_config.json"


def _latest_epoch_checkpoint(checkpoint_dir: Path) -> tuple[Path | None, int]:
    """Return ``(path, epoch)`` for the newest ``model_epoch_*.pth``, else ``model_current.pth``."""
    ckpt_dir = checkpoint_dir / "training" / "checkpoints"
    if not ckpt_dir.is_dir():
        return None, 0
    best_epoch = 0
    best_path: Path | None = None
    for p in ckpt_dir.glob("model_epoch_*.pth"):
        try:
            ep = int(p.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if ep >= best_epoch:
            best_epoch = ep
            best_path = p
    if best_path is not None:
        return best_path, best_epoch
    current = ckpt_dir / "model_current.pth"
    if current.is_file():
        return current, 0
    final = ckpt_dir / "model_final.pth"
    if final.is_file():
        return final, 0
    return None, 0


def _model_path(checkpoint_dir: Path) -> Path:
    """Prefer newest ``model_epoch_*.pth`` over stale ``model_final.pth``."""
    latest, _ = _latest_epoch_checkpoint(checkpoint_dir)
    if latest is not None:
        return latest
    ckpt_dir = checkpoint_dir / "training" / "checkpoints"
    final = ckpt_dir / "model_final.pth"
    if final.is_file():
        return final
    return ckpt_dir / "model_current.pth"


def _finalize_training_checkpoint(checkpoint_dir: Path) -> Path:
    """Copy the latest epoch checkpoint to ``model_final.pth`` (full checkpoint dict)."""
    import shutil

    ckpt_sub = checkpoint_dir / "training" / "checkpoints"
    latest, epoch = _latest_epoch_checkpoint(checkpoint_dir)
    final = ckpt_sub / "model_final.pth"
    if latest is not None and epoch > 0:
        shutil.copy2(latest, final)
        return final
    current = ckpt_sub / "model_current.pth"
    if current.is_file():
        shutil.copy2(current, final)
        return final
    raise FileNotFoundError(f"No checkpoint to finalize under {ckpt_sub}")


def save_brt_config(checkpoint_dir: Path, config: KozBRTConfig) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _config_path(checkpoint_dir).write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")


def load_brt_config(checkpoint_dir: Path) -> KozBRTConfig:
    p = _config_path(checkpoint_dir)
    if not p.is_file():
        raise FileNotFoundError(f"Missing DeepReach config: {p}")
    return KozBRTConfig.from_dict(json.loads(p.read_text(encoding="utf-8")))


def build_dynamics(config: KozBRTConfig) -> Any:
    if not DEEPREACH_AVAILABLE or dr_dynamics is None:
        raise RuntimeError("DeepReach requires torch and the vendored deepreach/ package.")
    dyn = dr_dynamics.Cw6DKoz(
        n_rad_s=config.n_rad_s,
        u_max_m_s2=config.u_max_m_s2,
        semi_axes_m=config.semi_axes_m,
        center_m=config.center_m,
        d_max_m_s2=config.d_max_m_s2,
        domain_lo=tuple(float(x) for x in config.domain_lo),
        domain_hi=tuple(float(x) for x in config.domain_hi),
    )
    dyn.deepreach_model = config.train.deepreach_model
    return dyn


def build_model(dynamics: Any, train: DeepReachTrainConfig) -> Any:
    return modules.SingleBVPNet(
        in_features=dynamics.input_dim,
        out_features=1,
        type=train.model_type,
        mode="mlp",
        final_layer_factor=1.0,
        hidden_features=train.hidden_features,
        num_hidden_layers=train.num_hidden_layers,
    )


def train_koz_deepreach(
    config: KozBRTConfig,
    checkpoint_dir: Path,
    *,
    force: bool = False,
    resume: bool = False,
    resume_from: str | Path | None = None,
    start_epoch: int | None = None,
) -> Path:
    """Train DeepReach value function; returns path to model checkpoint."""
    if not DEEPREACH_AVAILABLE:
        msg = "Install torch and deepreach dependencies (see requirements-deepreach.txt)."
        if DEEPREACH_IMPORT_ERROR:
            msg += f" ({DEEPREACH_IMPORT_ERROR})"
        raise RuntimeError(msg)

    from experiments import experiments as dr_experiments  # noqa: E402
    from utils import dataio, losses  # noqa: E402

    checkpoint_dir = Path(checkpoint_dir).resolve()
    ckpt = _model_path(checkpoint_dir)
    if ckpt.is_file() and not force and not resume:
        return ckpt

    resume_ckpt: Path | None = None
    resume_epoch = 0
    if resume:
        if resume_from is not None:
            resume_ckpt = Path(resume_from).resolve()
            if not resume_ckpt.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")
            resume_epoch = int(start_epoch) if start_epoch is not None else 0
        else:
            resume_ckpt, resume_epoch = _latest_epoch_checkpoint(checkpoint_dir)
            if resume_ckpt is None:
                raise FileNotFoundError(
                    f"No checkpoint to resume in {checkpoint_dir / 'training' / 'checkpoints'}"
                )
        if start_epoch is not None:
            resume_epoch = int(start_epoch)
        if _config_path(checkpoint_dir).is_file():
            saved = load_brt_config(checkpoint_dir)
            tc = asdict(saved.train)
            tc["num_epochs"] = int(config.train.num_epochs)
            tc["counter_end"] = max(int(config.train.counter_end), int(config.train.num_epochs))
            if config.train.device:
                tc["device"] = config.train.device
            config = KozBRTConfig(
                n_rad_s=saved.n_rad_s,
                semi_axes_m=saved.semi_axes_m,
                center_m=saved.center_m,
                domain_lo=saved.domain_lo,
                domain_hi=saved.domain_hi,
                horizon_s=saved.horizon_s,
                u_max_m_s2=saved.u_max_m_s2,
                d_max_m_s2=saved.d_max_m_s2,
                train=DeepReachTrainConfig(**tc),
            )

    if force and checkpoint_dir.exists() and not resume:
        import shutil

        shutil.rmtree(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not resume or not _config_path(checkpoint_dir).is_file():
        save_brt_config(checkpoint_dir, config)

    dynamics = build_dynamics(config)
    train_cfg = config.train
    device = _resolve_device(train_cfg.device)

    dataset = dataio.ReachabilityDataset(
        dynamics=dynamics,
        numpoints=train_cfg.numpoints,
        pretrain=True,
        pretrain_iters=train_cfg.pretrain_iters,
        tMin=0.0,
        tMax=float(config.horizon_s),
        counter_start=0,
        counter_end=train_cfg.counter_end,
        num_src_samples=train_cfg.num_src_samples,
        num_target_samples=0,
    )

    model = build_model(dynamics, train_cfg)
    model.to(device)

    experiment_dir = str(checkpoint_dir)
    experiment = dr_experiments.DeepReach(
        model=model,
        dataset=dataset,
        experiment_dir=experiment_dir,
        use_wandb=False,
    )
    experiment.init_special()

    loss_fn = losses.init_brt_hjivi_loss(dynamics, train_cfg.min_with, 1.0)
    if resume and resume_ckpt is not None:
        print(
            f"Resuming DeepReach from epoch {resume_epoch} → {train_cfg.num_epochs} "
            f"({resume_ckpt.name}), device={device}"
        )
    else:
        print(
            f"Training DeepReach KOZ BRT: horizon={config.horizon_s:.0f} s, "
            f"model={train_cfg.deepreach_model}, CSL={train_cfg.use_csl}, "
            f"domain pos x∈[{config.domain_lo[0]:.0f},{config.domain_hi[0]:.0f}] m, "
            f"y∈[{config.domain_lo[1]:.0f},{config.domain_hi[1]:.0f}] m, "
            f"epochs={train_cfg.num_epochs}, device={device}"
        )
    t0 = time.perf_counter()
    experiment.train(
        device=device,
        batch_size=train_cfg.batch_size,
        epochs=train_cfg.num_epochs,
        lr=train_cfg.lr,
        steps_til_summary=train_cfg.steps_til_summary,
        epochs_til_checkpoint=train_cfg.epochs_til_checkpoint,
        loss_fn=loss_fn,
        clip_grad=train_cfg.clip_grad,
        use_lbfgs=False,
        adjust_relative_grads=True,
        val_x_resolution=120,
        val_y_resolution=120,
        val_z_resolution=3,
        val_time_resolution=3,
        use_CSL=train_cfg.use_csl,
        CSL_lr=train_cfg.lr,
        CSL_dt=train_cfg.csl_dt,
        epochs_til_CSL=train_cfg.epochs_til_csl if train_cfg.use_csl else 10**9,
        num_CSL_samples=train_cfg.num_csl_samples if train_cfg.use_csl else 0,
        CSL_loss_frac_cutoff=train_cfg.csl_loss_frac_cutoff,
        max_CSL_epochs=train_cfg.max_csl_epochs if train_cfg.use_csl else 0,
        CSL_loss_weight=train_cfg.csl_loss_weight,
        CSL_batch_size=train_cfg.csl_batch_size,
        start_epoch=resume_epoch if resume else 0,
        resume_checkpoint=str(resume_ckpt) if resume and resume_ckpt is not None else None,
    )
    elapsed = time.perf_counter() - t0
    out = _finalize_training_checkpoint(checkpoint_dir)
    print(f"DeepReach training finished in {elapsed / 60:.1f} min. Checkpoint: {out}")
    return out


class KozDeepReachBRT:
    """Learned 6D BRT value function V(τ, x). Unsafe iff V ≤ 0 at horizon τ = ``horizon_s``."""

    def __init__(
        self,
        model: Any,
        dynamics: Any,
        config: KozBRTConfig,
        *,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._dynamics = dynamics
        self._config = config
        self._device = device
        self._lo = np.asarray(config.domain_lo, dtype=np.float64).reshape(6)
        self._hi = np.asarray(config.domain_hi, dtype=np.float64).reshape(6)
        self._horizon = float(config.horizon_s)
        self._model.eval()

    @property
    def domain_lo(self) -> np.ndarray:
        return self._lo.copy()

    @property
    def domain_hi(self) -> np.ndarray:
        return self._hi.copy()

    @property
    def horizon_s(self) -> float:
        return self._horizon

    @property
    def config(self) -> KozBRTConfig:
        return self._config

    @classmethod
    def load(cls, checkpoint_dir: str | Path, *, device: str | None = None) -> KozDeepReachBRT:
        if not DEEPREACH_AVAILABLE:
            detail = f" ({DEEPREACH_IMPORT_ERROR})" if DEEPREACH_IMPORT_ERROR else ""
            raise RuntimeError(f"DeepReach not available{detail}.")
        checkpoint_dir = Path(checkpoint_dir).resolve()
        config = load_brt_config(checkpoint_dir)
        dev = _resolve_device(device or config.train.device)
        dynamics = build_dynamics(config)
        model = build_model(dynamics, config.train)
        ckpt = _model_path(checkpoint_dir)
        if not ckpt.is_file():
            raise FileNotFoundError(f"No DeepReach checkpoint at {ckpt}")
        state = torch.load(ckpt, map_location=dev, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
        model.to(dev)
        return cls(model, dynamics, config, device=dev)

    def value_at_tau(self, x_lvlh_m: np.ndarray, tau_s: float) -> float:
        return float(self.value_batch_at_tau(np.asarray(x_lvlh_m, dtype=np.float64).reshape(1, 6), tau_s)[0])

    def value_batch_at_tau(self, x6: np.ndarray, tau_s: float) -> np.ndarray:
        pts = np.asarray(x6, dtype=np.float64).reshape(-1, 6)
        tau = np.full((pts.shape[0], 1), float(tau_s), dtype=np.float64)
        coords = np.concatenate([tau, pts], axis=1)
        return self._eval_coords(coords)

    def value(self, x_lvlh_m: np.ndarray) -> float:
        """Value at backward horizon (full BRT)."""
        return self.value_at_tau(x_lvlh_m, self._horizon)

    def value_batch(self, x6: np.ndarray) -> np.ndarray:
        return self.value_batch_at_tau(x6, self._horizon)

    def is_unsafe(self, x_lvlh_m: np.ndarray) -> bool:
        v = self.value(x_lvlh_m)
        if not np.isfinite(v):
            return True
        return v <= 0.0

    def backward_times_s(self, n_nodes: int) -> np.ndarray:
        """τ from 0 (terminal) to -horizon (matches legacy HJ convention for viz)."""
        return np.linspace(0.0, -abs(self._horizon), int(n_nodes), dtype=np.float64)

    def _torch_dtype(self) -> torch.dtype:
        return next(self._model.parameters()).dtype

    def _eval_coords(self, coords: np.ndarray) -> np.ndarray:
        dtype = self._torch_dtype()
        c = torch.tensor(coords, dtype=dtype, device=self._device)
        inp = self._dynamics.coord_to_input(c)
        with torch.no_grad():
            res = self._model({"coords": inp})
            vals = self._dynamics.io_to_value(res["model_in"], res["model_out"].squeeze(dim=-1))
        return vals.detach().cpu().numpy().astype(np.float64).reshape(-1)


def load_or_train_koz_brt(
    n_rad_s: float,
    *,
    semi_axes_m: tuple[float, float, float] | np.ndarray,
    center_m: np.ndarray | None = None,
    checkpoint_dir: str | Path | None = None,
    train_config: DeepReachTrainConfig | None = None,
    force_train: bool = False,
) -> tuple[KozDeepReachBRT, bool]:
    """Load trained DeepReach BRT or train if missing. Returns ``(brt, loaded_from_disk)``."""
    ck_dir = Path(checkpoint_dir).resolve() if checkpoint_dir else default_checkpoint_dir()
    axes = tuple(float(x) for x in np.asarray(semi_axes_m, dtype=np.float64).reshape(3))
    cen = tuple(float(x) for x in (center_m if center_m is not None else np.zeros(3)).reshape(3))
    tc = train_config or DeepReachTrainConfig()
    if os.environ.get("DEEPREACH_DEVICE", "").strip():
        tc = DeepReachTrainConfig(**{**asdict(tc), "device": os.environ["DEEPREACH_DEVICE"].strip()})
    if os.environ.get("DEEPREACH_EPOCHS", "").strip():
        tc = DeepReachTrainConfig(
            **{**asdict(tc), "num_epochs": int(os.environ["DEEPREACH_EPOCHS"]), "counter_end": int(os.environ["DEEPREACH_EPOCHS"])}
        )

    config = KozBRTConfig(
        n_rad_s=float(n_rad_s),
        semi_axes_m=axes,
        center_m=cen,
        domain_lo=DOMAIN_LO.copy(),
        domain_hi=DOMAIN_HI.copy(),
        horizon_s=float(os.environ.get("BRT_HORIZON_S", str(BRT_HORIZON_S))),
        u_max_m_s2=float(os.environ.get("BRT_U_MAX_M_S2", "0.2")),
        d_max_m_s2=float(os.environ.get("BRT_D_MAX_M_S2", "0")),
        train=tc,
    )

    force = force_train or os.environ.get("DEEPREACH_FORCE_TRAIN", "0").lower() in ("1", "true", "yes")
    auto_train = os.environ.get("DEEPREACH_AUTO_TRAIN", "1").lower() not in ("0", "false", "no")
    ckpt = _model_path(ck_dir)
    loaded = ckpt.is_file() and not force

    if not loaded:
        if not auto_train:
            raise FileNotFoundError(
                f"No DeepReach checkpoint at {ckpt}. Train first: python -m simulation.brt.train "
                "or set DEEPREACH_AUTO_TRAIN=1."
            )
        train_koz_deepreach(config, ck_dir, force=force)
    elif not _config_path(ck_dir).is_file():
        save_brt_config(ck_dir, config)

    brt = KozDeepReachBRT.load(ck_dir, device=config.train.device)
    return brt, loaded
