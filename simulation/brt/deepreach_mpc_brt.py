"""DeepReach-MPC neural BRT for 6D CW KOZ collision (avoid game, exact boundary model)."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from simulation.brt.config import (
    BRT_HORIZON_S,
    DEFAULT_DEEPREACH_CHECKPOINT_SUBDIR,
    TRAIN_DOMAIN_HI,
    TRAIN_DOMAIN_LO,
    U_MAX_M_S2,
    DeepReachTrainConfig,
    KozBRTConfig,
)

_ROOT = Path(__file__).resolve().parents[2]
_DEEPREACH_MPC = _ROOT / "deepreach_MPC"

DEEPREACH_MPC_AVAILABLE = False
DEEPREACH_MPC_IMPORT_ERROR: str | None = None

try:
    import torch

    DEEPREACH_MPC_AVAILABLE = True
except ImportError as exc:
    torch = None  # type: ignore[assignment]
    DEEPREACH_MPC_IMPORT_ERROR = str(exc)


@contextmanager
def _deepreach_mpc_imports() -> Iterator[None]:
    """Import ``deepreach_MPC`` with its package-relative imports."""
    if not _DEEPREACH_MPC.is_dir():
        raise RuntimeError(f"deepreach_MPC/ not found at {_DEEPREACH_MPC}")
    prev = os.getcwd()
    added = str(_DEEPREACH_MPC) not in sys.path
    if added:
        sys.path.insert(0, str(_DEEPREACH_MPC))
    try:
        yield
    finally:
        if added:
            sys.path.remove(str(_DEEPREACH_MPC))
        os.chdir(prev)


def default_checkpoint_dir(project_root: str | Path | None = None) -> Path:
    root = Path(project_root).resolve() if project_root else _ROOT
    env = os.environ.get("DEEPREACH_CHECKPOINT_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (root / "simulation_output" / DEFAULT_DEEPREACH_CHECKPOINT_SUBDIR).resolve()


def _resolve_device(requested: str) -> str:
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


def _model_path(checkpoint_dir: Path) -> Path:
    from simulation.brt.training_metrics import resolve_inference_checkpoint

    path, _, _ = resolve_inference_checkpoint(checkpoint_dir)
    return path


def _finalize_training_checkpoint(checkpoint_dir: Path) -> Path:
    from simulation.brt.training_metrics import sync_model_final_from_best

    return sync_model_final_from_best(checkpoint_dir)


def save_brt_config(checkpoint_dir: Path, config: KozBRTConfig) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _config_path(checkpoint_dir).write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")


def load_brt_config(checkpoint_dir: Path) -> KozBRTConfig:
    p = _config_path(checkpoint_dir)
    if not p.is_file():
        raise FileNotFoundError(f"Missing DeepReach config: {p}")
    return KozBRTConfig.from_dict(json.loads(p.read_text(encoding="utf-8")))


def _apply_mpc_env_overrides(train_cfg: DeepReachTrainConfig) -> DeepReachTrainConfig:
    """Optional VM tuning without editing config.py."""
    kw: dict[str, float | int] = {}
    if v := os.environ.get("DEEPREACH_MPC_DT", "").strip():
        kw["MPC_dt"] = float(v)
    if v := os.environ.get("DEEPREACH_MPC_BATCH_SIZE", "").strip():
        kw["MPC_batch_size"] = int(v)
    if v := os.environ.get("DEEPREACH_MPC_PERTURB_SAMPLES", "").strip():
        kw["num_MPC_perturbation_samples"] = int(v)
    if v := os.environ.get("DEEPREACH_MPC_NUM_BATCHES", "").strip():
        kw["num_MPC_batches"] = int(v)
    return replace(train_cfg, **kw) if kw else train_cfg


def log_control_authority(dynamics: Any) -> None:
    if not hasattr(dynamics, "u_max"):
        return
    a_max = max(dynamics._semi_axes)
    u = float(dynamics.u_max)
    T = float(dynamics.horizon_s)
    reach = 0.5 * u * T * T
    print(f"  KOZ max semi-axis (m): {a_max:.1f}")
    print(f"  u_max={u:.3f} m/s², T={T:.0f} s → 0.5*u_max*T²~{reach:.0f} m")
    if u > 1e-8:
        print(f"  Displacement/a_max~{reach / max(a_max, 1e-9):.1f}")


def build_dynamics(config: KozBRTConfig, *, device: str | None = None) -> Any:
    with _deepreach_mpc_imports():
        from dynamics import dynamics as dr_dynamics  # noqa: WPS433

    dev = _resolve_device(device or config.train.device) if DEEPREACH_MPC_AVAILABLE else "cpu"
    ax = config.semi_axes_m
    cen = config.center_m
    dyn = dr_dynamics.Cw6DKoz(
        n_rad_s=config.n_rad_s,
        u_max_m_s2=config.u_max_m_s2,
        semi_axis_x_m=ax[0],
        semi_axis_y_m=ax[1],
        semi_axis_z_m=ax[2],
        center_x_m=cen[0],
        center_y_m=cen[1],
        center_z_m=cen[2],
        d_max_m_s2=config.d_max_m_s2,
        horizon_s=config.horizon_s,
        set_mode="avoid",
        device=dev,
    )
    dyn.set_model(config.train.deepreach_model)
    train = config.train
    dyn.enforce_koz_invariant = bool(getattr(train, "enforce_koz_invariant", False))
    dyn.num_koz_invariant_samples = int(getattr(train, "num_koz_invariant_samples", 0) or 0)
    return dyn


def build_model(dynamics: Any, train: DeepReachTrainConfig) -> Any:
    with _deepreach_mpc_imports():
        from utils import modules  # noqa: WPS433

    return modules.SingleBVPNet(
        in_features=dynamics.input_dim,
        out_features=1,
        type=train.model_type,
        mode="mlp",
        final_layer_factor=1.0,
        hidden_features=train.hidden_features,
        num_hidden_layers=train.num_hidden_layers,
        periodic_transform_fn=dynamics.periodic_transform_fn,
    )


def train_koz_deepreach_mpc(
    config: KozBRTConfig,
    checkpoint_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Train DeepReach-MPC value function; returns path to ``model_final.pth``."""
    if not DEEPREACH_MPC_AVAILABLE:
        msg = "Install torch (see requirements-deepreach.txt)."
        if DEEPREACH_MPC_IMPORT_ERROR:
            msg += f" ({DEEPREACH_MPC_IMPORT_ERROR})"
        raise RuntimeError(msg)
    if not torch.cuda.is_available():
        diag = [
            f"torch {torch.__version__}",
            "cuda.is_available()=False",
        ]
        if "+cpu" in torch.__version__.lower() or "cpu" in str(getattr(torch.version, "cuda", "") or "").lower():
            diag.append("This looks like a CPU-only PyTorch build.")
        diag.extend(
            [
                "DeepReach-MPC training hard-codes .cuda() in experiments.py (NVIDIA GPU required).",
                "Fix: install CUDA PyTorch in this env, e.g.",
                "  pip install torch --index-url https://download.pytorch.org/whl/cu124",
                "Then verify: python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\"",
                "Also run: nvidia-smi  (driver + GPU must be visible to the session).",
            ]
        )
        raise RuntimeError("DeepReach-MPC training requires CUDA.\n  " + "\n  ".join(diag))

    checkpoint_dir = Path(checkpoint_dir).resolve()
    ckpt = _model_path(checkpoint_dir)
    if ckpt.is_file() and not force:
        return ckpt

    if force and checkpoint_dir.exists():
        import shutil

        shutil.rmtree(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_brt_config(checkpoint_dir, config)

    train_cfg = _apply_mpc_env_overrides(config.train)
    config = KozBRTConfig(
        n_rad_s=config.n_rad_s,
        semi_axes_m=config.semi_axes_m,
        center_m=config.center_m,
        domain_lo=config.domain_lo,
        domain_hi=config.domain_hi,
        horizon_s=config.horizon_s,
        u_max_m_s2=config.u_max_m_s2,
        d_max_m_s2=config.d_max_m_s2,
        train=train_cfg,
    )
    t_max = float(config.horizon_s)
    time_hr = train_cfg.time_till_refinement
    if time_hr is None:
        time_hr = t_max / 10.0
    mpc_steps = int(math.ceil(time_hr / train_cfg.MPC_dt))

    prev_cwd = os.getcwd()
    os.chdir(checkpoint_dir)
    try:
        with _deepreach_mpc_imports():
            from experiments import experiments as dr_experiments  # noqa: WPS433
            from utils import dataio, losses  # noqa: WPS433

            dynamics = build_dynamics(config, device="cuda")
            num_target = train_cfg.num_target_samples if train_cfg.num_target_samples > 0 else 0

            dataset = dataio.ReachabilityDataset(
                dynamics=dynamics,
                numpoints=train_cfg.numpoints,
                pretrain=True,
                pretrain_iters=train_cfg.pretrain_iters,
                tMin=0.0,
                tMax=t_max,
                counter_start=0,
                counter_end=train_cfg.counter_end,
                num_src_samples=train_cfg.num_src_samples,
                num_target_samples=num_target,
                use_MPC=train_cfg.use_mpc,
                time_curr=train_cfg.time_curr,
                MPC_data_path="none",
                num_MPC_perturbation_samples=train_cfg.num_MPC_perturbation_samples,
                MPC_dt=train_cfg.MPC_dt,
                MPC_mode="MPC",
                MPC_sample_mode="gaussian",
                MPC_style="direct",
                MPC_lambda_=0.1,
                MPC_batch_size=train_cfg.MPC_batch_size,
                MPC_receding_horizon=-1,
                num_MPC_data_samples=train_cfg.num_MPC_data_samples,
                num_iterative_refinement=train_cfg.num_iterative_refinement,
                time_till_refinement=time_hr,
                num_MPC_batches=train_cfg.num_MPC_batches,
                aug_with_MPC_data=train_cfg.aug_with_MPC_data,
                policy=None,
                refine_dataset=train_cfg.refine_dataset,
            )

            model = build_model(dynamics, train_cfg)
            model.cuda()

            experiment = dr_experiments.DeepReach(
                model=model,
                dataset=dataset,
                experiment_dir=str(checkpoint_dir),
                use_wandb=False,
            )
            experiment.init_special()

            loss_fn = losses.init_brt_hjivi_loss(
                dynamics,
                train_cfg.min_with,
                train_cfg.dirichlet_loss_divisor,
                "l1",
                train_cfg.use_mpc,
                MPC_finetune_lambda=train_cfg.MPC_finetune_lambda,
                koz_invariant_loss_divisor=train_cfg.koz_invariant_loss_divisor,
            )

            if getattr(dynamics, "enforce_koz_invariant", False):
                experiment.loss_weights["koz_invariant"] = train_cfg.koz_invariant_loss_weight
            else:
                experiment.loss_weights["koz_invariant"] = 0.0

            print(
                f"Training DeepReach-MPC KOZ BRT: T={config.horizon_s:.0f} s, "
                f"tMax={t_max:.0f} s, model={train_cfg.deepreach_model}, set_mode=avoid, "
                f"MPC H_R={time_hr:.0f}s, dt={train_cfg.MPC_dt:g}s ({mpc_steps} steps), "
                f"batch={train_cfg.MPC_batch_size}, perturb={train_cfg.num_MPC_perturbation_samples}, "
                f"target_samples={num_target}, "
                f"koz_invariant={getattr(dynamics, 'enforce_koz_invariant', False)} "
                f"(n={getattr(dynamics, 'num_koz_invariant_samples', 0)}, "
                f"w={train_cfg.koz_invariant_loss_weight:g}), "
                f"curriculum counter_end={train_cfg.counter_end}, epochs={train_cfg.num_epochs}"
            )
            log_control_authority(dynamics)

            t0 = time.perf_counter()
            experiment.train(
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
                MPC_importance_init=train_cfg.MPC_importance_init,
                MPC_importance_final=train_cfg.MPC_importance_final,
                MPC_decay_scheme="exponential",
            )
            elapsed = time.perf_counter() - t0
            out = _finalize_training_checkpoint(checkpoint_dir)
            print(f"DeepReach-MPC training finished in {elapsed / 60:.1f} min. Checkpoint: {out}")
            return out
    finally:
        os.chdir(prev_cwd)


class KozDeepReachBRT:
    """Learned 6D BRT V(t, x). Unsafe iff V ≤ 0.

    KOZ invariant (default on): if g(x) ≤ 0 (inside/on KOZ), V ← min(V, g(x)) so the
    keep-out zone stays unsafe at every query time, not only at t = 0.
    Disable: ``BRT_KOZ_PROJECT=0``.
    """

    def __init__(self, model: Any, dynamics: Any, config: KozBRTConfig, *, device: str = "cpu") -> None:
        self._model = model
        self._dynamics = dynamics
        self._config = config
        self._device = device
        self._lo = np.asarray(config.domain_lo, dtype=np.float64).reshape(6)
        self._hi = np.asarray(config.domain_hi, dtype=np.float64).reshape(6)
        self._horizon = float(config.horizon_s)
        self._koz_center = np.asarray(config.center_m, dtype=np.float64).reshape(3)
        self._koz_axes = np.asarray(config.semi_axes_m, dtype=np.float64).reshape(3)
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
        if not DEEPREACH_MPC_AVAILABLE:
            detail = f" ({DEEPREACH_MPC_IMPORT_ERROR})" if DEEPREACH_MPC_IMPORT_ERROR else ""
            raise RuntimeError(f"DeepReach-MPC not available{detail}.")
        checkpoint_dir = Path(checkpoint_dir).resolve()
        config = load_brt_config(checkpoint_dir)
        dev = _resolve_device(device or config.train.device)
        dynamics = build_dynamics(config, device=dev)
        model = build_model(dynamics, config.train)
        from simulation.brt.training_metrics import resolve_inference_checkpoint

        ckpt, ep, reason = resolve_inference_checkpoint(checkpoint_dir)
        if not ckpt.is_file():
            raise FileNotFoundError(f"No checkpoint at {ckpt}")
        print(f"Loading DeepReach checkpoint: {ckpt.name} ({reason})")
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
        return self.value_at_tau(x_lvlh_m, self._horizon)

    def value_batch(self, x6: np.ndarray) -> np.ndarray:
        return self.value_batch_at_tau(x6, self._horizon)

    def is_unsafe(self, x_lvlh_m: np.ndarray) -> bool:
        v = self.value(x_lvlh_m)
        if not np.isfinite(v):
            return True
        return v <= 0.0

    def backward_times_s(self, n_nodes: int) -> np.ndarray:
        return np.linspace(0.0, -abs(self._horizon), int(n_nodes), dtype=np.float64)

    def koz_boundary_g(self, pos_lvlh_m: np.ndarray) -> np.ndarray:
        """Ellipsoid boundary g(x)=√(Σ (r_i/a_i)²)−1; negative inside KOZ (matches Cw6DKoz)."""
        pos = np.asarray(pos_lvlh_m, dtype=np.float64).reshape(-1, 3)
        r = pos - self._koz_center.reshape(1, 3)
        ax = self._koz_axes.reshape(1, 3)
        s = np.sum((r / ax) ** 2, axis=-1)
        return np.sqrt(s + 1e-18) - 1.0

    def _project_koz_unsafe(self, states_si: np.ndarray, vals: np.ndarray) -> np.ndarray:
        """Inside/on KOZ (g≤0), enforce V≤0 via V ← min(V, g)."""
        if os.environ.get("BRT_KOZ_PROJECT", "1").lower() in ("0", "false", "no"):
            return vals
        states = np.asarray(states_si, dtype=np.float64).reshape(-1, 6)
        v = np.asarray(vals, dtype=np.float64).reshape(-1).copy()
        g = self.koz_boundary_g(states[:, :3])
        inside = g <= 0.0
        if np.any(inside):
            v[inside] = np.minimum(v[inside], g[inside])
        return v

    def _eval_coords(self, coords: np.ndarray) -> np.ndarray:
        """``coords`` are ``[t_s, x, y, z, vx, vy, vz]`` in SI (like Quadrotor: raw time in col 0)."""
        dtype = next(self._model.parameters()).dtype
        c = torch.tensor(coords, dtype=dtype, device=self._device)
        states_si = c[..., 1:]
        states_norm = (states_si - self._dynamics.state_mean.to(device=self._device, dtype=dtype)) / (
            self._dynamics.state_var.to(device=self._device, dtype=dtype)
        )
        inp = torch.cat((c[..., :1], states_norm), dim=-1)
        with torch.no_grad():
            res = self._model({"coords": inp})
            vals = self._dynamics.io_to_value(res["model_in"], res["model_out"].squeeze(dim=-1))
        states_np = states_si.detach().cpu().numpy().astype(np.float64).reshape(-1, 6)
        v_np = vals.detach().cpu().numpy().astype(np.float64).reshape(-1)
        return self._project_koz_unsafe(states_np, v_np)


def load_or_train_koz_brt(
    n_rad_s: float,
    *,
    semi_axes_m: tuple[float, float, float] | np.ndarray,
    center_m: np.ndarray | None = None,
    checkpoint_dir: str | Path | None = None,
    train_config: DeepReachTrainConfig | None = None,
    force_train: bool = False,
) -> tuple[KozDeepReachBRT, bool]:
    ck_dir = Path(checkpoint_dir).resolve() if checkpoint_dir else default_checkpoint_dir()
    axes = tuple(float(x) for x in np.asarray(semi_axes_m, dtype=np.float64).reshape(3))
    cen = tuple(float(x) for x in (center_m if center_m is not None else np.zeros(3)).reshape(3))
    tc = _apply_mpc_env_overrides(train_config or DeepReachTrainConfig())
    if os.environ.get("DEEPREACH_DEVICE", "").strip():
        tc = DeepReachTrainConfig(**{**asdict(tc), "device": os.environ["DEEPREACH_DEVICE"].strip()})
    if os.environ.get("DEEPREACH_EPOCHS", "").strip():
        n = int(os.environ["DEEPREACH_EPOCHS"])
        tc = DeepReachTrainConfig(**{**asdict(tc), "num_epochs": n, "counter_end": n})

    config = KozBRTConfig(
        n_rad_s=float(n_rad_s),
        semi_axes_m=axes,
        center_m=cen,
        domain_lo=TRAIN_DOMAIN_LO.copy(),
        domain_hi=TRAIN_DOMAIN_HI.copy(),
        horizon_s=float(os.environ.get("BRT_HORIZON_S", str(BRT_HORIZON_S))),
        u_max_m_s2=float(os.environ.get("BRT_U_MAX_M_S2", str(U_MAX_M_S2))),
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
                f"No checkpoint at {ckpt}. Train: python -m simulation.brt.train --force"
            )
        train_koz_deepreach_mpc(config, ck_dir, force=force)
    elif not _config_path(ck_dir).is_file():
        save_brt_config(ck_dir, config)

    return KozDeepReachBRT.load(ck_dir, device=config.train.device), loaded
