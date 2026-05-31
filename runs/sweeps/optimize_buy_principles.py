"""Optimize the buy principles (WHEN and HOW MUCH) on out-of-sample data.

Holds the fixed best candidate (target-HF sizing, HF floor 1.20) and sweeps two
added principles on the 15-asset OOS set (data/oos/, non-overlapping 270-day
windows, clustered to the window level):

  WHEN  -- buyback_uptrend_lookback L: buy only if price rose vs L steps ago.
  HOW MUCH -- buyback_stress_drawdown d: size so post-buy HF>=1 after a further d drop.

Goal: drive buyback-induced extra liquidations to zero and bad debt down while
keeping the restoration / USD benefit. Reports, per (L,d), paired vs no-buyback.
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aave_counterfactual_pipeline import run_counterfactual

LT, DEBT, W = 0.83, 100_000.0, 270
CRS = [1.25, 1.35, 1.50]
SCN = {"buy": "baseline_dynamic", "no": "target_hf_no_buyback"}
# (lookback, min_bounce, stress_drawdown) — refine the WHEN gate (confirmed
# bounce off a local low) on top of the stress-tested HOW-MUCH cap.
COMBOS = [
    (0, 0.00, 0.00), (5, 0.00, 0.15),   # baselines: unguarded sizing
    (5, 0.00, 0.25), (10, 0.10, 0.25),  # the effective lever
    (20, 0.10, 0.30), (10, 0.10, 0.35),
]


def base_dynamic() -> dict:
    scn = json.loads((ROOT / "runs" / "sweeps" / "scenario_candidate_best.json").read_text(encoding="utf-8"))
    for s in scn:
        if s.get("name") == "baseline_dynamic":
            return dict(s["dynamic"])
    raise SystemExit("no baseline_dynamic in candidate")


def make_windows(batch: Path) -> list[tuple[str, int, Path]]:
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


def run(windows, dyn: dict, tag: str, batch: Path) -> pd.DataFrame:
    scn = [{"name": "baseline_dynamic", "dynamic": dyn}]
    sp = batch / f"scn_{tag}.json"
    sp.write_text(json.dumps(scn), encoding="utf-8")
    rows = []
    for sym, wi, ds in windows:
        rd = run_counterfactual(dataset_dir=ds, scenario_path=sp, output_dir=batch / "runs" / tag, run_id=f"{sym}_{wi}_{ds.name.split('_')[-1]}")
        m = pd.read_csv(rd / "scenario_metrics.csv").iloc[0]
        rows.append({"asset": sym, "window": wi, "loss": m.avg_borrower_final_loss_usd,
                     "rho": m.avg_restoration_ratio, "sells": m.total_sell_events,
                     "buys": m.total_buy_events, "bad_debt": m.max_bad_debt_usd})
    return pd.DataFrame(rows)


def cluster(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["asset", "window"]).mean(numeric_only=True)


def main() -> None:
    batch = ROOT / "runs" / "opt_principles"
    import shutil
    shutil.rmtree(batch, ignore_errors=True)
    (batch / "ds").mkdir(parents=True, exist_ok=True)
    windows = make_windows(batch)

    base = base_dynamic()
    no_dyn = {**base, "enable_buyback": False, "buy_cooldown_steps": 1_000_000}
    no = cluster(run(windows, no_dyn, "nobuy", batch))

    results = []
    for L, m, d in COMBOS:
        dyn = {**base, "enable_buyback": True, "buyback_ratio": 1.0,
               "buyback_uptrend_lookback": L, "buyback_min_bounce": m, "buyback_stress_drawdown": d}
        buy = cluster(run(windows, dyn, f"L{L}_m{int(m*100)}_d{int(d*100)}", batch))
        j = no.join(buy, lsuffix="_no", rsuffix="_buy")
        active = j["buys_buy"] > 0.001
        dloss = j["loss_no"] - j["loss_buy"]
        drho = j["rho_buy"] - j["rho_no"]
        extra = j["sells_buy"] - j["sells_no"]
        dloss_a = dloss[active]
        nz = dloss_a[dloss_a.abs() > 1e-9]
        p = stats.wilcoxon(nz, alternative="greater").pvalue if len(nz) > 5 else float("nan")
        results.append({
            "lookback": L, "min_bounce": m, "drawdown": d,
            "active": int(active.sum()),
            "extra_sells_mean": float(extra.mean()), "extra_sells_max": float(extra.max()),
            "bad_debt_max": float(j["bad_debt_buy"].max()),
            "rho_win": int((drho > 1e-9).sum()), "rho_lose": int((drho < -1e-9).sum()),
            "usd_active_mean": float(dloss_a.mean()) if active.sum() else 0.0,
            "usd_win": int((dloss_a > 1e-9).sum()), "usd_lose": int((dloss_a < -1e-9).sum()),
            "usd_p": float(p),
        })
        r = results[-1]
        print(f"L={L:2d} m={m:.2f} d={d:.2f}: active={r['active']:2d} extra_sells mean={r['extra_sells_mean']:+.2f} max={r['extra_sells_max']:+.2f} "
              f"bad_debt={r['bad_debt_max']:.0f} | rho W/L={r['rho_win']}/{r['rho_lose']} | "
              f"usd {r['usd_active_mean']:+.0f} W/L={r['usd_win']}/{r['usd_lose']} p={r['usd_p']:.3f}")

    out = pd.DataFrame(results)
    out.to_csv(ROOT / "runs" / "sweeps" / "buy_principles_oos_results.csv", index=False)
    print("\nsaved buy_principles_oos_results.csv")


if __name__ == "__main__":
    main()
