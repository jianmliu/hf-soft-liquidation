"""Delta (buyback risk-guard bandwidth) sensitivity replay.

Replays the best-candidate policy family over the same historical-window
protocol for delta in {0.00, 0.04, 0.08}, holding all other parameters fixed.
Reports the three-policy borrower-loss ranking per delta.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aave_counterfactual_pipeline import historical_backtest

DELTAS = [0.00, 0.04, 0.08]


def main() -> None:
    dataset_dir = ROOT / "data" / "aave"
    sweep_dir = ROOT / "runs" / "sweeps"
    base = json.loads((sweep_dir / "scenario_candidate_best.json").read_text(encoding="utf-8"))
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    rows: list[dict] = []
    for delta in DELTAS:
        scenario = json.loads(json.dumps(base))  # deep copy
        for s in scenario:
            if "dynamic" in s:
                s["dynamic"]["recovery_ltv_gap"] = delta
        scenario_path = sweep_dir / "hist_candidates" / f"delta_{stamp}_{int(delta*100):02d}.json"
        scenario_path.write_text(json.dumps(scenario, indent=2), encoding="utf-8")

        _, aggregate_path, _ = historical_backtest(
            dataset_dir=dataset_dir,
            scenario_path=scenario_path,
            output_dir=sweep_dir / "hist_runs",
            window_size=120,
            window_step=24,
            max_windows=24,
            loan_mid_starts=True,
            loan_min_duration_blocks=36,
        )
        agg = pd.read_csv(aggregate_path)
        by_s = {r["scenario"]: r for _, r in agg.iterrows()}
        rows.append(
            {
                "delta": delta,
                "mean_loss_fixed": float(by_s["traditional_fixed_cf"]["mean_avg_borrower_final_loss_usd"]),
                "mean_loss_no_buyback": float(by_s["target_hf_no_buyback"]["mean_avg_borrower_final_loss_usd"]),
                "mean_loss_buyback": float(by_s["baseline_dynamic"]["mean_avg_borrower_final_loss_usd"]),
                "buy_events": int(by_s["baseline_dynamic"]["total_buy_events"]),
                "worst_bad_debt_buyback": float(by_s["baseline_dynamic"]["worst_max_bad_debt_usd"]),
            }
        )

    out = pd.DataFrame(rows)
    out_path = sweep_dir / f"delta_sensitivity_summary_{stamp}.csv"
    out.to_csv(out_path, index=False)
    print("delta_sensitivity_csv", out_path)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
