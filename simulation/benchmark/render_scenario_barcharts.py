#!/usr/bin/env python3
"""Render benchmark bar charts for boundary and BRT-V corpora."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.benchmark.generate_report import render_benchmark_barchart_png

DEFAULT_SCENARIOS: tuple[tuple[str, str, str, str], ...] = (
    (
        "boundary",
        "simulation_output/llm_benchmark_summary_margin0.json",
        "Boundary corpus - y=70-110 m, brt_margin=0, capture=100 m (72 plans)",
        "simulation_output/report/figures/benchmark_results_barchart_boundary.png",
    ),
    (
        "brt_v",
        "simulation_output/ablation_brt_v_passive_off.json",
        "BRT-V corpus - y=48-65 m, brt_margin=0, capture=50 m (66 plans)",
        "simulation_output/report/figures/benchmark_results_barchart_brt_v.png",
    ),
)


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_all(
    out_dir: Path,
    *,
    only: frozenset[str] | None = None,
) -> list[Path]:
    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, summary_rel, title, _out_rel in DEFAULT_SCENARIOS:
        if only is not None and key not in only:
            continue
        summary_path = ROOT / summary_rel
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing summary JSON for {key}: {summary_path}")
        out_path = out_dir / f"benchmark_results_barchart_{key}.png"
        summary = _load_summary(summary_path)
        result = render_benchmark_barchart_png(summary, out_path, title=title)
        if result is None:
            raise RuntimeError(f"Could not render bar chart for {key}")
        written.append(result)
        print(f"Wrote {result}")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Render per-scenario benchmark bar charts.")
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "simulation_output" / "report" / "figures"),
        help="Directory for PNG outputs.",
    )
    p.add_argument(
        "--scenario",
        type=str,
        default="",
        help="Render one scenario only: boundary | brt_v",
    )
    args = p.parse_args()

    only = frozenset({args.scenario.strip()}) if args.scenario.strip() else None
    if only and only - {s[0] for s in DEFAULT_SCENARIOS}:
        raise SystemExit(f"Unknown scenario {args.scenario!r}; choose boundary or brt_v")

    render_all(Path(args.output_dir).resolve(), only=only)


if __name__ == "__main__":
    main()
