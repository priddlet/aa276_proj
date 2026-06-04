"""Training loss logs and checkpoint selection for DeepReach-MPC KOZ runs."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def collect_epoch_losses(
    ckpt_dir: Path,
    *,
    window: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-checkpoint losses from cumulative ``train_losses_epoch_*.txt`` files.

    Returns ``(epochs, point_loss, window_mean)`` where:

    - ``point_loss[i]`` = last logged step in that epoch file (``L[ep-1]``; noisy).
    - ``window_mean[i]`` = mean of the last ``window`` steps in the same file (smoother).
    """
    files = sorted(
        ckpt_dir.glob("train_losses_epoch_*.txt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    epochs: list[int] = []
    point: list[float] = []
    wmean: list[float] = []
    w = max(1, int(window))
    for f in files:
        ep = int(f.stem.rsplit("_", 1)[-1])
        L = np.loadtxt(f)
        if L.ndim == 0:
            L = np.asarray([float(L)])
        L = np.asarray(L, dtype=np.float64).reshape(-1)
        if len(L) < ep:
            continue
        epochs.append(ep)
        point.append(float(L[ep - 1]))
        sl = L[max(0, ep - w) : ep]
        wmean.append(float(np.mean(sl)) if sl.size else float(L[ep - 1]))
    return (
        np.asarray(epochs, dtype=np.int64),
        np.asarray(point, dtype=np.float64),
        np.asarray(wmean, dtype=np.float64),
    )


def best_epoch_from_losses(
    epochs: np.ndarray,
    losses: np.ndarray,
    *,
    metric: str = "window_mean",
) -> tuple[int, float]:
    """Return ``(epoch, loss)`` for the best checkpoint by the chosen metric."""
    if epochs.size == 0:
        return 0, float("inf")
    i = int(np.argmin(losses))
    return int(epochs[i]), float(losses[i])


def _checkpoint_select_mode() -> str:
    return os.environ.get("DEEPREACH_CHECKPOINT_SELECT", "best").strip().lower()


def resolve_inference_checkpoint(
    checkpoint_dir: Path,
    *,
    ckpt_sub: Path | None = None,
) -> tuple[Path, int, str]:
    """Pick the ``.pth`` used for inference (best-by-loss, latest, or explicit epoch).

    ``DEEPREACH_CHECKPOINT_SELECT``:

    - ``best`` (default): lowest window-mean loss among logged checkpoints.
    - ``latest``: highest epoch ``model_epoch_*.pth``.
    - ``<integer>``: that epoch (e.g. ``98000``).
    - ``point``: best single-step loss at checkpoint epoch (legacy metric).
    """
    sub = ckpt_sub or (checkpoint_dir / "training" / "checkpoints")
    mode = _checkpoint_select_mode()
    window = int(os.environ.get("DEEPREACH_LOSS_WINDOW", "200"))

    if mode.isdigit():
        ep = int(mode)
        p = sub / f"model_epoch_{ep:05d}.pth"
        if p.is_file():
            return p, ep, f"epoch {ep} (DEEPREACH_CHECKPOINT_SELECT)"
        raise FileNotFoundError(f"No checkpoint for epoch {ep}: {p}")

    if mode == "latest":
        best_ep = 0
        best_path: Path | None = None
        for p in sub.glob("model_epoch_*.pth"):
            try:
                ep = int(p.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if ep >= best_ep:
                best_ep = ep
                best_path = p
        if best_path is not None:
            return best_path, best_ep, f"latest epoch {best_ep}"
        for name in ("model_current.pth", "model_final.pth"):
            p = sub / name
            if p.is_file():
                return p, 0, name
        raise FileNotFoundError(f"No checkpoints under {sub}")

    epochs, point, wmean = collect_epoch_losses(sub, window=window)
    if epochs.size == 0:
        for name in ("model_final.pth", "model_current.pth"):
            p = sub / name
            if p.is_file():
                return p, 0, name
        raise FileNotFoundError(f"No train_losses_epoch_*.txt under {sub}")

    metric = "point" if mode == "point" else "window_mean"
    losses = point if metric == "point" else wmean
    ep, loss = best_epoch_from_losses(epochs, losses, metric=metric)
    p = sub / f"model_epoch_{ep:05d}.pth"
    if not p.is_file():
        raise FileNotFoundError(f"Best epoch {ep} missing checkpoint: {p}")
    tag = "point" if metric == "point" else f"window-{window} mean"
    return p, ep, f"best-by-loss ({tag}) epoch {ep}, loss={loss:.4e}"


def sync_model_final_from_best(checkpoint_dir: Path) -> Path:
    """Copy the best-by-loss epoch checkpoint to ``model_final.pth``."""
    import shutil

    sub = checkpoint_dir / "training" / "checkpoints"
    src, ep, reason = resolve_inference_checkpoint(checkpoint_dir, ckpt_sub=sub)
    dst = sub / "model_final.pth"
    shutil.copy2(src, dst)
    meta = sub / "model_final_source.txt"
    meta.write_text(f"epoch={ep}\npath={src.name}\nreason={reason}\n", encoding="utf-8")
    return dst
