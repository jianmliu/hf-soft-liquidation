"""Synthetic stress batch for the target-HF soft-liquidation mechanism.

Under initial-CR normalization (each loan starts at CR_0 at its own start price),
a uniform price shock cancels out of the CR path, so stress must come from the
SHAPE of the price path. This batch therefore generates diverse seeded
crash-and-rebound paths (varying drawdown depth, trough timing, and rebound
strength) and runs the three matched policies on healthy CR_0=1.30 loans.

Output: runs/<batch_id>/batch_summary.csv with one row per (run, scenario),
columns aligned with the historical backtest so the table/stats generator can
consume both.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aave_counterfactual_pipeline import run_counterfactual

LT = 0.83
DEBT = 100_000.0
INITIAL_CR = 1.30
LENGTH = 180
N_RUNS = 120


def synthetic_path(length: int, seed: int) -> np.ndarray:
    """Crash-and-rebound path: gentle trend + a drawdown trough + partial
    recovery + noise. Drawdown depth, trough location, and rebound strength
    vary with the seed to span diverse local trend/reversal structures."""
    rng = np.random.default_rng(seed)
    x = np.arange(length)
    p0 = 3000.0
    drift = 1.0 + 0.04 * np.sin(x / 40.0)
    # Drawdown: V-shaped dip to a trough at a seed-dependent location.
    depth = rng.uniform(0.18, 0.45)
    trough = int(rng.uniform(0.30, 0.70) * length)
    width = max(1.0, length * rng.uniform(0.18, 0.30))
    dip = -depth * np.maximum(0.0, 1.0 - np.abs(x - trough) / width)
    # Partial rebound after the trough (seed-dependent recovery fraction).
    rebound_frac = rng.uniform(0.3, 1.0)
    rebound = np.where(x > trough, rebound_frac * depth * (x - trough) / max(1.0, length - trough), 0.0)
    noise = rng.normal(0.0, 0.012, size=length)
    prices = p0 * drift * (1.0 + dip + rebound + noise)
    return np.clip(prices, 100.0, None)


def main() -> None:
    scenario_path = ROOT / "runs" / "sweeps" / "scenario_candidate_best.json"
    if not scenario_path.exists():
        raise SystemExit(f"missing best-candidate scenario: {scenario_path}; run the sweep first")

    batch_id = datetime.now(tz=timezone.utc).strftime("synbatch_%Y%m%d_%H%M%S")
    batch_dir = ROOT / "runs" / batch_id
    (batch_dir / "datasets").mkdir(parents=True, exist_ok=True)
    runs_dir = batch_dir / "runs"

    summary_rows: list[dict] = []
    for run_idx in range(1, N_RUNS + 1):
        seed = 40_000 + run_idx
        prices = synthetic_path(LENGTH, seed)
        ds_dir = batch_dir / "datasets" / f"run_{run_idx:03d}"
        (ds_dir / "normalized").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "block_number": np.arange(LENGTH),
                "timestamp": pd.date_range("2024-01-01", periods=LENGTH, freq="h").astype(str),
                "asset_symbol": "WETH",
                "price_usd": prices,
            }
        ).to_csv(ds_dir / "normalized" / "prices.csv", index=False)
        pd.DataFrame(
            [
                {
                    "account": "0xstress",
                    "asset_symbol": "WETH",
                    "collateral_amount": 1.0,  # overridden by initial_cr
                    "debt_amount": DEBT,
                    "liquidation_threshold": LT,
                    "initial_cr": INITIAL_CR,
                }
            ]
        ).to_csv(ds_dir / "normalized" / "positions_initial.csv", index=False)

        run_dir = run_counterfactual(
            dataset_dir=ds_dir,
            scenario_path=scenario_path,
            output_dir=runs_dir,
            run_id=f"run_{run_idx:03d}",
        )
        metrics = pd.read_csv(run_dir / "scenario_metrics.csv")
        metrics.insert(0, "batch_run_id", f"run_{run_idx:03d}")
        metrics.insert(1, "seed", seed)
        summary_rows.extend(metrics.to_dict(orient="records"))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = batch_dir / "batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("batch_summary", summary_path)
    piv = summary_df.pivot_table(
        index="scenario",
        values=["avg_borrower_final_loss_usd", "max_bad_debt_usd", "total_sell_events", "total_buy_events"],
        aggfunc="mean",
    )
    print(piv.to_string())


if __name__ == "__main__":
    main()
