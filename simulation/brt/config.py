"""Locked BRT / DeepReach domain and training defaults (reproducible experiments)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Full 6D state box for legacy HJ / wide visualization (LVLH SI).
DOMAIN_LO = np.array([-1200.0, -600.0, -600.0, -3.0, -3.0, -3.0], dtype=np.float64)
DOMAIN_HI = np.array([1200.0, 6600.0, 600.0, 3.0, 3.0, 3.0], dtype=np.float64)

# Smaller training domain: KOZ at origin + deputy approach corridor (~3200 m along-track).
TRAIN_DOMAIN_LO = np.array([-800.0, -200.0, -400.0, -2.0, -2.0, -2.0], dtype=np.float64)
TRAIN_DOMAIN_HI = np.array([800.0, 4000.0, 400.0, 2.0, 2.0, 2.0], dtype=np.float64)

# Backward reachability horizon in physical seconds. Nondim time τ_max = n * horizon_s (~2 for LEO).
BRT_HORIZON_S = 1800.0

# Length scale L (m) for nondim positions x̃ = x/L; velocities ṽ = v/(nL).
BRT_LENGTH_SCALE_M = 1000.0

U_MAX_M_S2 = 0.2
D_MAX_M_S2 = 0.0

# Visualization sampling density (not a PDE grid — queries the learned V).
SLICE_GRID_NX = 80
SLICE_GRID_NY = 100
SLICE_TIME_NODES = 12

ISO_SNAPSHOT = (32, 32, 26)
FORMATION_ISO = (18, 18, 14)


@dataclass(frozen=True)
class DeepReachTrainConfig:
    """Training hyperparameters for KOZ DeepReach."""

    numpoints: int = 65000
    pretrain_iters: int = 3000
    num_epochs: int = 8000
    counter_end: int = 8000
    num_src_samples: int = 1000
    num_target_samples: int = 8000
    lr: float = 2e-5
    num_hidden_layers: int = 3
    hidden_features: int = 512
    model_type: str = "sine"
    deepreach_model: str = "exact"
    min_with: str = "target"
    batch_size: int = 1
    steps_til_summary: int = 200
    epochs_til_checkpoint: int = 2000
    clip_grad: float = 1.0
    device: str = "cpu"
    use_csl: bool = False
    epochs_til_csl: int = 500
    num_csl_samples: int = 50000
    max_csl_epochs: int = 50
    csl_dt: float = 0.05
    csl_loss_weight: float = 1.0
    csl_batch_size: int = 1000
    csl_loss_frac_cutoff: float = 0.1


@dataclass
class KozBRTConfig:
    """Full BRT experiment configuration persisted with checkpoints."""

    n_rad_s: float
    semi_axes_m: tuple[float, float, float] = (28.0, 45.0, 18.0)
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    domain_lo: np.ndarray = field(default_factory=lambda: TRAIN_DOMAIN_LO.copy())
    domain_hi: np.ndarray = field(default_factory=lambda: TRAIN_DOMAIN_HI.copy())
    horizon_s: float = BRT_HORIZON_S
    length_scale_m: float = BRT_LENGTH_SCALE_M
    u_max_m_s2: float = U_MAX_M_S2
    d_max_m_s2: float = D_MAX_M_S2
    train: DeepReachTrainConfig = field(default_factory=DeepReachTrainConfig)

    @property
    def tau_max(self) -> float:
        return float(self.n_rad_s * self.horizon_s)

    @property
    def state_mean(self) -> np.ndarray:
        return 0.5 * (self.domain_lo + self.domain_hi)

    @property
    def state_var(self) -> np.ndarray:
        return 0.5 * (self.domain_hi - self.domain_lo)

    def as_dict(self) -> dict:
        return {
            "n_rad_s": self.n_rad_s,
            "semi_axes_m": self.semi_axes_m,
            "center_m": self.center_m,
            "domain_lo": self.domain_lo.tolist(),
            "domain_hi": self.domain_hi.tolist(),
            "horizon_s": self.horizon_s,
            "length_scale_m": self.length_scale_m,
            "u_max_m_s2": self.u_max_m_s2,
            "d_max_m_s2": self.d_max_m_s2,
            "train": {
                k: getattr(self.train, k)
                for k in DeepReachTrainConfig.__dataclass_fields__
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> KozBRTConfig:
        train_d = d.get("train", {})
        fields = DeepReachTrainConfig.__dataclass_fields__
        train = DeepReachTrainConfig(**{k: train_d[k] for k in fields if k in train_d})
        return cls(
            n_rad_s=float(d["n_rad_s"]),
            semi_axes_m=tuple(float(x) for x in d["semi_axes_m"]),
            center_m=tuple(float(x) for x in d["center_m"]),
            domain_lo=np.asarray(d.get("domain_lo", TRAIN_DOMAIN_LO), dtype=np.float64),
            domain_hi=np.asarray(d.get("domain_hi", TRAIN_DOMAIN_HI), dtype=np.float64),
            horizon_s=float(d.get("horizon_s", BRT_HORIZON_S)),
            length_scale_m=float(d.get("length_scale_m", BRT_LENGTH_SCALE_M)),
            u_max_m_s2=float(d.get("u_max_m_s2", U_MAX_M_S2)),
            d_max_m_s2=float(d.get("d_max_m_s2", D_MAX_M_S2)),
            train=train,
        )
