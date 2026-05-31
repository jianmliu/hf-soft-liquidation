"""Sell-side sweep to reduce out-of-sample bad debt.

OOS bad debt comes from the sell side: with LLTV=0.85 > LT=0.83 the first sell
fires only after HF<1 (late), so multi-leg crashes leave a residual. Earlier
triggers (lower LLTV) require a higher repair target HF* for feasibility
(HF* > LT/LLTV). This sweep co-varies (LLTV, HF*) and reports, on the 15-asset
OOS set (non-overlapping 270-day windows), worst/again bad debt, borrower loss,
and sell intensity for the no-buyback policy, against the fixed-CF baseline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aave_counterfactual_pipeline import run_counterfactual

LT, DEBT, W = 0.83, 100_000.0, 270
CRS = [1.25, 1.35, 1.50]
# (LLTV, HF*) with HF* > LT/LLTV; earlier trigger (lower LLTV) needs higher HF*.
PAIRS = [(0.85, 1.05), (0.83, 1.05), (0.80, 1.10), (0.78, 1.12),
         (0.75, 1.15), (0.72, 1.20), (0.70, 1.25)]


def make_windows(batch: Path):
    out = []
    for csv in sorted((ROOT / "data" / "oos").glob("*.csv")):
        sym = csv.stem
        prices = pd.read_csv(csv)["price_usd"].astype(float).to_numpy()
        for wi in range(len(prices) // W):
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
                out.append((sym, wi, ds))
    return out


def run(windows, scn: list[dict], tag: str, batch: Path, name: str) -> pd.DataFrame:
    sp = batch / f"scn_{tag}.json"
    sp.write_text(json.dumps(scn), encoding="utf-8")
    rows = []
    for sym, wi, ds in windows:
        rd = run_counterfactual(dataset_dir=ds, scenario_path=sp, output_dir=batch / "runs" / tag, run_id=f"{sym}_{wi}_{ds.name.split('_')[-1]}")
        m = pd.read_csv(rd / "scenario_metrics.csv")
        m = m[m["scenario"] == name].iloc[0]
        rows.append({"asset": sym, "window": wi, "loss": m.avg_borrower_final_loss_usd,
                     "sells": m.total_sell_events, "bad_debt": m.max_bad_debt_usd, "min_hf": m.avg_min_hf})
    return pd.DataFrame(rows).groupby(["asset", "window"]).mean(numeric_only=True)


def dynamic_no_buyback(lltv: float, hf: float) -> list[dict]:
    return [{"name": "target_hf_no_buyback", "dynamic": {
        "lltv": lltv, "target_hf": hf, "min_close_factor": 0.15, "max_close_factor": 0.60,
        "cf_slope": 1.6, "liquidation_bonus": 0.01, "buyback_ratio": 0.2, "buyback_funding": "reborrow",
        "enable_buyback": False, "recovery_ltv_gap": 0.08, "sell_cooldown_steps": 1, "buy_cooldown_steps": 1_000_000}}]


def fixed_cf() -> list[dict]:
    return [{"name": "traditional_fixed_cf", "buyback_bandwidth": 10.0, "tiers": [
        {"name": "Fixed CF 50%", "hf_down": 1.01, "close_factor": 0.50, "liquidation_bonus": 0.06, "buyback_ratio": 0.70}]}]


def main() -> None:
    batch = ROOT / "runs" / "sell_side_sweep"
    import shutil
    shutil.rmtree(batch, ignore_errors=True)
    (batch / "ds").mkdir(parents=True, exist_ok=True)
    windows = make_windows(batch)

    fx = run(windows, fixed_cf(), "fixed", batch, "traditional_fixed_cf")
    print(f"fixed-CF baseline: mean_loss={fx['loss'].mean():.0f} bad_debt max={fx['bad_debt'].max():.0f} "
          f"windows_with_bad_debt={(fx['bad_debt']>1).sum()}/{len(fx)}")
    print("-" * 92)
    rows = []
    for lltv, hf in PAIRS:
        r = run(windows, dynamic_no_buyback(lltv, hf), f"l{int(lltv*100)}_h{int(hf*100)}", batch, "target_hf_no_buyback")
        nbd = int((r["bad_debt"] > 1).sum())
        rows.append({"lltv": lltv, "target_hf": hf, "trigger_hf": LT / lltv,
                     "mean_loss": float(r["loss"].mean()), "bad_debt_max": float(r["bad_debt"].max()),
                     "bad_debt_total": float(r["bad_debt"].sum()), "windows_with_bad_debt": nbd,
                     "mean_sells": float(r["sells"].mean())})
        x = rows[-1]
        print(f"LLTV={lltv:.2f} HF*={hf:.2f} (trigger HF={x['trigger_hf']:.3f}): "
              f"mean_loss={x['mean_loss']:.0f} bad_debt max={x['bad_debt_max']:.0f} total={x['bad_debt_total']:.0f} "
              f"n_bad={nbd}/{len(r)} sells={x['mean_sells']:.1f}")
    pd.DataFrame(rows).to_csv(ROOT / "runs" / "sweeps" / "sell_side_oos_results.csv", index=False)
    print("\nsaved sell_side_oos_results.csv")


if __name__ == "__main__":
    main()
