"""Load frozen LLM maneuver benchmarks from ``llm/`` (see ``llm/llm_plans.json``)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def _vec3(obj: Any) -> np.ndarray:
    a = np.asarray(obj, dtype=np.float64).reshape(-1)
    if a.size != 3:
        raise ValueError("dv must be length-3")
    return a.copy()


def default_llm_dir(project_root: str | Path | None = None) -> Path:
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    env = os.environ.get("LLM_PLANS_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (root / "llm").resolve()


def _resolve_data_file(llm_dir: Path, env_key: str, candidates: tuple[str, ...]) -> Path:
    env = os.environ.get(env_key, "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_absolute() else llm_dir / p
    for name in candidates:
        p = llm_dir / name
        if p.is_file():
            return p
    return llm_dir / candidates[0]


def _bundle_path(llm_dir: Path) -> Path:
    return _resolve_data_file(
        llm_dir,
        "LLM_PLANS_BUNDLE",
        ("llm_plans.json", "llm_plans_leo.json"),
    )


def _segments_path(llm_dir: Path) -> Path:
    return _resolve_data_file(
        llm_dir,
        "LLM_PLANS_SEGMENTS",
        ("llm_plans_segments.jsonl", "llm_plans_leo_segments.jsonl"),
    )


_RENDEZVOUS_MIN = re.compile(
    r"rendezvous\s+exactly.*?in\s+(\d+(?:\.\d+)?)\s*min",
    re.IGNORECASE | re.DOTALL,
)


def rollout_horizon_s(scenario: LLMScenario, prompt: str) -> float:
    """Passive-rollout horizon: rendezvous target time or scenario BRT horizon."""
    m = _RENDEZVOUS_MIN.search(prompt or "")
    if m:
        return float(m.group(1)) * 60.0
    return float(scenario.brt_horizon_s)


def finalize_segments_for_rollout(
    segments: list[tuple[float, np.ndarray | None]],
    rollout_horizon_s: float,
) -> list[tuple[float, np.ndarray | None]]:
    """Append trailing coast so burn-only segment lists actually propagate in CW."""
    segs = list(segments)
    total = sum(float(dt) for dt, _ in segs)
    pad = max(0.0, float(rollout_horizon_s) - total)
    if pad > 1e-9:
        segs.append((pad, None))
    return segs


def _summary_path(llm_dir: Path) -> Path:
    return llm_dir / "plans_summary.csv"


@dataclass(frozen=True)
class LLMScenario:
    """Scenario block shared by all plans in a bundle."""

    id: str
    start_state_lvlh_m: np.ndarray
    semi_axes_m: tuple[float, float, float]
    brt_horizon_s: float
    mean_motion_rad_s: float
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LLMScenario:
        axes = tuple(float(x) for x in d["inner_koz"]["semi_axes_m"])
        return cls(
            id=str(d["id"]),
            start_state_lvlh_m=np.asarray(d["start_state_m_m_s"], dtype=np.float64).reshape(6),
            semi_axes_m=axes,
            brt_horizon_s=float(d["brt_horizon_s"]),
            mean_motion_rad_s=float(d["mean_motion_rad_s"]),
            raw=d,
        )

    @property
    def dv_cap_m_s(self) -> float | None:
        raw = self.raw.get("dv_cap_m_s")
        return float(raw) if raw is not None else None


@dataclass
class LLMPlan:
    """One benchmark plan with simulator-ready segments."""

    plan_id: str
    prompt: str
    segments: list[tuple[float, np.ndarray | None]]
    tags: dict[str, Any]
    expected_intervention: int | None
    label: dict[str, Any] | None
    maneuvers_absolute: list[dict[str, Any]] | None

    @property
    def n_burns(self) -> int:
        return sum(1 for _, dv in self.segments if dv is not None)


def segments_record_to_sim_segments(
    segments: list[dict[str, Any]],
) -> list[tuple[float, np.ndarray | None]]:
    """Convert 'llm_plans_segments.jsonl' entries to 'simulate_impulsive_segments' format."""
    out: list[tuple[float, np.ndarray | None]] = []
    for s in segments:
        coast_s = float(s["coast_s"])
        dv_raw = s.get("dv_m_s")
        if dv_raw is None:
            out.append((coast_s, None))
            continue
        dv = _vec3(dv_raw)
        out.append((coast_s, None if not np.any(dv != 0) else dv))
    return out


def absolute_burns_to_segments(
    maneuvers: list[dict[str, Any]],
) -> list[tuple[float, np.ndarray | None]]:
    """Absolute burn times -> interleaved coast / burn segments for burn-at-start-then-coast."""
    burns = sorted(maneuvers, key=lambda m: float(m["t_s"]))
    segs: list[tuple[float, np.ndarray | None]] = []
    t_prev = 0.0
    for m in burns:
        t_burn = float(m["t_s"])
        gap = t_burn - t_prev
        if gap > 0.0:
            segs.append((gap, None))
        dv_raw = m.get("dv_m_s", m.get("dv", m.get("delta_v")))
        if dv_raw is None:
            dv_raw = [0.0, 0.0, 0.0]
        dv = _vec3(dv_raw)
        segs.append((0.0, None if not np.any(dv != 0) else dv))
        t_prev = t_burn
    return segs


def load_llm_bundle(llm_dir: str | Path | None = None) -> tuple[LLMScenario, list[dict[str, Any]]]:
    """Load 'llm_plans.json'; returns scenario dict and raw plan records."""
    root = Path(llm_dir).resolve() if llm_dir else default_llm_dir()
    path = _bundle_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark bundle: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    scenario = LLMScenario.from_dict(data["scenario"])
    plans = list(data["plans"])
    return scenario, plans


def load_llm_segments_index(llm_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load 'llm_plans_segments.jsonl' keyed by 'plan_id'."""
    root = Path(llm_dir).resolve() if llm_dir else default_llm_dir()
    path = _segments_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing segments file: {path}")
    index: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        index[str(rec["plan_id"])] = rec
    return index


def load_llm_plans(llm_dir: str | Path | None = None) -> tuple[LLMScenario, list[LLMPlan]]:
    """Merge bundle labels with segment lists ready for simulation."""
    scenario, raw_plans = load_llm_bundle(llm_dir)
    seg_index = load_llm_segments_index(llm_dir)
    plans: list[LLMPlan] = []
    for rec in raw_plans:
        pid = str(rec["plan_id"])
        if pid not in seg_index:
            raise KeyError(f"plan_id {pid!r} missing from llm_plans_segments.jsonl")
        seg_rec = seg_index[pid]
        label = rec.get("label")
        expected = None
        if label is not None:
            expected = int(label.get("expected_intervention", 0))
        prompt = str(rec.get("prompt", ""))
        raw_segs = segments_record_to_sim_segments(seg_rec["segments"])
        sim_segs = finalize_segments_for_rollout(
            raw_segs, rollout_horizon_s(scenario, prompt)
        )
        plans.append(
            LLMPlan(
                plan_id=pid,
                prompt=prompt,
                segments=sim_segs,
                tags=dict(rec.get("tags", {})),
                expected_intervention=expected,
                label=label,
                maneuvers_absolute=list(rec.get("maneuvers", [])),
            )
        )
    return scenario, plans


def iter_llm_plans(llm_dir: str | Path | None = None) -> Iterator[LLMPlan]:
    _, plans = load_llm_plans(llm_dir)
    yield from plans
