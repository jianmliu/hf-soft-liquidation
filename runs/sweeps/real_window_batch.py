"""Real-window stress batch for the target-HF soft-liquidation mechanism.

Stress batch = 120 windows sampled from the real ETH price series (different
start points), each loan initialized at CR_0=1.30 at its window start price.
This reuses realistic drawdown/rebound dynamics (rather than invented crash
shapes) and yields 120 paired runs across the three matched policies.

Output: runs/<batch_id>/batch_summary.csv with one row per (run, scenario).
"""
from __future__ import annotations

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
# CR levels all start above the trigger CR (1/LLTV = 1/0.85 = 1.176) so target-HF
# has room to act; lower levels trigger sooner, raising the count of informative
# (non-tie) paired runs and the statistical power of the buyback comparison.
INITIAL_CRS = [1.25, 1.35, 1.50]
WINDOW = 180
N_WINDOWS = 200
SEED = 20260529


def main() -> None:
    scenario_path = ROOT / "runs" / "sweeps" / "scenario_candidate_best.json"
    if not scenario_path.exists():
        raise SystemExit(f"missing best-candidate scenario: {scenario_path}; run the sweep first")

    base = pd.read_csv(ROOT / "data" / "aave" / "normalized" / "prices.csv")
    base = base.sort_values("block_number").reset_index(drop=True)
    series = base["price_usd"].astype(float).to_numpy()
    n = len(series)
    if n <= WINDOW + 1:
        raise SystemExit("price series too short for the chosen window")

    rng = np.random.default_rng(SEED)
    # Distinct sampled start points (real-window draws) spanning the series.
    max_start = n - WINDOW
    starts = sorted(int(s) for s in rng.choice(np.arange(max_start), size=min(N_WINDOWS, max_start), replace=False))

    batch_id = datetime.now(tz=timezone.utc).strftime("realbatch_%Y%m%d_%H%M%S")
    batch_dir = ROOT / "runs" / batch_id
    (batch_dir / "datasets").mkdir(parents=True, exist_ok=True)
    runs_dir = batch_dir / "runs"

    summary_rows: list[dict] = []
    for w_idx, start in enumerate(starts, start=1):
        window_prices = series[start : start + WINDOW]
        for cr in INITIAL_CRS:
            run_tag = f"w{w_idx:03d}_cr{int(cr*100)}"
            ds_dir = batch_dir / "datasets" / run_tag
            (ds_dir / "normalized").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "block_number": np.arange(WINDOW),
                    "timestamp": pd.date_range("2024-01-01", periods=WINDOW, freq="h").astype(str),
                    "asset_symbol": "WETH",
                    "price_usd": window_prices,
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
                        "initial_cr": cr,
                    }
                ]
            ).to_csv(ds_dir / "normalized" / "positions_initial.csv", index=False)

            run_dir = run_counterfactual(
                dataset_dir=ds_dir,
                scenario_path=scenario_path,
                output_dir=runs_dir,
                run_id=run_tag,
            )
            metrics = pd.read_csv(run_dir / "scenario_metrics.csv")
            metrics.insert(0, "batch_run_id", run_tag)
            metrics.insert(1, "start_index", start)
            metrics.insert(2, "initial_cr", cr)
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
