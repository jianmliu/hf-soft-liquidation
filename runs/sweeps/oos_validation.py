"""Out-of-sample validation on multi-asset, multi-year data.

Takes the FIXED best candidate (scenario_candidate_best.json, selected on the
~1-year ETH series) and applies it WITHOUT re-tuning to 15 assets over 2019-2024
(data/oos/), on non-overlapping windows clustered to the window level. This is a
genuine out-of-sample test of (a) the target-HF > fixed-CF baseline, (b) buyback
safety (no ping-pong, no bad debt), and (c) restoration / USD effect.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aave_counterfactual_pipeline import run_counterfactual

LT, DEBT = 0.83, 100_000.0
CRS = [1.25, 1.35, 1.50]
W = 270  # ~9 months: long enough to contain a drawdown-rebound cycle
SCN = {"buy": "baseline_dynamic", "no": "target_hf_no_buyback", "fix": "traditional_fixed_cf"}


def main() -> None:
    oos = ROOT / "data" / "oos"
    scn_path = ROOT / "runs" / "sweeps" / "scenario_candidate_best.json"  # FIXED, not re-tuned
    batch = ROOT / "runs" / "oos_validation"
    import shutil
    shutil.rmtree(batch, ignore_errors=True)
    (batch / "ds").mkdir(parents=True, exist_ok=True)

    rows = []
    for csv in sorted(oos.glob("*.csv")):
        sym = csv.stem
        prices = pd.read_csv(csv)["price_usd"].astype(float).to_numpy()
        n_win = len(prices) // W
        for wi in range(n_win):
            seg = prices[wi * W:(wi + 1) * W]
            for cr in CRS:
                ds = batch / "ds" / f"{sym}_{wi}_{int(cr*100)}"
                (ds / "normalized").mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"block_number": np.arange(W),
                              "timestamp": pd.date_range("2024-01-01", periods=W, freq="h").astype(str),
                              "asset_symbol": "WETH", "price_usd": seg}).to_csv(ds / "normalized" / "prices.csv", index=False)
                pd.DataFrame([{"account": "x", "asset_symbol": "WETH", "collateral_amount": 1.0,
                               "debt_amount": DEBT, "liquidation_threshold": LT, "initial_cr": cr}]).to_csv(
                    ds / "normalized" / "positions_initial.csv", index=False)
                rd = run_counterfactual(dataset_dir=ds, scenario_path=scn_path, output_dir=batch / "runs",
                                        run_id=f"{sym}_{wi}_{int(cr*100)}")
                m = pd.read_csv(rd / "scenario_metrics.csv")
                m.insert(0, "asset", sym); m.insert(1, "window", wi); m.insert(2, "cr", cr)
                rows.append(m)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(batch / "oos_summary.csv", index=False)

    # cluster to (asset, window): one independent observation per window-path
    idx = ["asset", "window"]
    loss = df.pivot_table(index=idx, columns="scenario", values="avg_borrower_final_loss_usd")
    rho = df.pivot_table(index=idx, columns="scenario", values="avg_restoration_ratio")
    sell = df.pivot_table(index=idx, columns="scenario", values="total_sell_events")
    buy = df.pivot_table(index=idx, columns="scenario", values="total_buy_events")
    bd = df.pivot_table(index=idx, columns="scenario", values="max_bad_debt_usd")

    N = len(loss)
    # (a) baseline OOS: target-HF (no buyback) vs fixed-CF
    base = loss[SCN["fix"]] - loss[SCN["no"]]   # >0 => target-HF better
    base_nz = base[base.abs() > 1e-9]
    p_base = stats.wilcoxon(base_nz, alternative="greater").pvalue if len(base_nz) > 5 else float("nan")
    # (b) buyback vs no-buyback
    dloss = loss[SCN["no"]] - loss[SCN["buy"]]
    drho = rho[SCN["buy"]] - rho[SCN["no"]]
    active = buy[SCN["buy"]] > 0.001
    dloss_a = dloss[active]
    nz = dloss_a[dloss_a.abs() > 1e-9]
    p_buy = stats.wilcoxon(nz, alternative="greater").pvalue if len(nz) > 5 else float("nan")
    drho_nz = drho[drho.abs() > 1e-12]
    p_rho = stats.wilcoxon(drho_nz, alternative="greater").pvalue if len(drho_nz) > 5 else float("nan")

    out = {
        "window_days": W, "n_assets": int(df["asset"].nunique()), "n_independent_windows": int(N),
        "BASELINE_target_vs_fixed": {
            "mean_loss_fixed": float(loss[SCN["fix"]].mean()),
            "mean_loss_targetHF": float(loss[SCN["no"]].mean()),
            "target_better_win": int((base > 1e-9).sum()), "target_worse": int((base < -1e-9).sum()),
            "wilcoxon_p": float(p_base),
            "bad_debt_fixed_max": float(bd[SCN["fix"]].max()),
            "bad_debt_targetHF_max": float(bd[SCN["no"]].max()),
        },
        "BUYBACK_safety": {
            "windows_active": int(active.sum()), "activation_rate": float(active.mean()),
            "extra_sells_mean": float((sell[SCN["buy"]] - sell[SCN["no"]]).mean()),
            "extra_sells_max": float((sell[SCN["buy"]] - sell[SCN["no"]]).max()),
            "bad_debt_buyback_max": float(bd[SCN["buy"]].max()),
        },
        "BUYBACK_restoration_collateral": {
            "mean_rho_gain": float(drho.mean()), "win": int((drho > 1e-9).sum()),
            "lose": int((drho < -1e-9).sum()), "informative": int(len(drho_nz)), "wilcoxon_p": float(p_rho),
        },
        "BUYBACK_usd_on_active": {
            "active_windows": int(active.sum()), "informative": int(len(nz)),
            "mean_loss_reduction": float(dloss_a.mean()) if active.sum() else 0.0,
            "win": int((dloss_a > 1e-9).sum()), "lose": int((dloss_a < -1e-9).sum()),
            "wilcoxon_p": float(p_buy),
        },
    }
    (batch / "oos_clustered_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
