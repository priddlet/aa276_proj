"""Locked BRT / DeepReach-MPC domain and training defaults."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


DOMAIN_LO = np.array([-1200.0, -600.0, -600.0, -3.0, -3.0, -3.0], dtype=np.float64)
DOMAIN_HI = np.array([1200.0, 6600.0, 600.0, 3.0, 3.0, 3.0], dtype=np.float64)

TRAIN_DOMAIN_LO = np.array([-800.0, -200.0, -400.0, -2.0, -2.0, -2.0], dtype=np.float64)
TRAIN_DOMAIN_HI = np.array([800.0, 4000.0, 400.0, 2.0, 2.0, 2.0], dtype=np.float64)

BRT_HORIZON_S = 1800.0
U_MAX_M_S2 = 0.15
D_MAX_M_S2 = 0.0


@dataclass(frozen=True)
class DeepReachTrainConfig:
    """DeepReach-MPC training hyperparameters (KOZ 6D CW)."""

    numpoints: int = 65000
    pretrain_iters: int = 1000
    num_epochs: int = 100_000
    counter_end: int = 100_000
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
    device: str = "auto"

    use_mpc: bool = True
    time_curr: bool = True
    refine_dataset: bool = True
    time_till_refinement: float | None = None
    # SI seconds: use ~1 s MPC steps (Quadrotor uses 0.02 s because tMax≈1).
    MPC_batch_size: int = 512
    num_MPC_batches: int = 10
    num_MPC_data_samples: int = 5000
    num_iterative_refinement: int = 10
    MPC_dt: float = 1.0
    MPC_importance_init: float = 1.0
    MPC_importance_final: float = 1.0
    MPC_finetune_lambda: float = 100.0
    num_MPC_perturbation_samples: int = 32
    aug_with_MPC_data: int = 0
    dirichlet_loss_divisor: float = 1.0


@dataclass
class KozBRTConfig:
    n_rad_s: float
    semi_axes_m: tuple[float, float, float] = (28.0, 45.0, 18.0)
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    domain_lo: np.ndarray = field(default_factory=lambda: TRAIN_DOMAIN_LO.copy())
    domain_hi: np.ndarray = field(default_factory=lambda: TRAIN_DOMAIN_HI.copy())
    horizon_s: float = BRT_HORIZON_S
    u_max_m_s2: float = U_MAX_M_S2
    d_max_m_s2: float = D_MAX_M_S2
    train: DeepReachTrainConfig = field(default_factory=DeepReachTrainConfig)

    def as_dict(self) -> dict:
        return {
            "n_rad_s": self.n_rad_s,
            "semi_axes_m": self.semi_axes_m,
            "center_m": self.center_m,
            "domain_lo": self.domain_lo.tolist(),
            "domain_hi": self.domain_hi.tolist(),
            "horizon_s": self.horizon_s,
            "u_max_m_s2": self.u_max_m_s2,
            "d_max_m_s2": self.d_max_m_s2,
            "train": {k: getattr(self.train, k) for k in DeepReachTrainConfig.__dataclass_fields__},
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
            u_max_m_s2=float(d.get("u_max_m_s2", U_MAX_M_S2)),
            d_max_m_s2=float(d.get("d_max_m_s2", D_MAX_M_S2)),
            train=train,
        )
