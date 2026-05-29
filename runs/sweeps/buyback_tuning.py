"""Can the buyback leg be tuned to robustly help borrowers?

On the real-window stress batch, sweep the buyback re-leverage guards
(buyback_hf_floor, min_buyback_spread, buyback_ratio) and, for each combo,
compare buyback-on vs buyback-off (paired, same windows). Reports mean loss
difference, win/lose/tie rates, induced extra sells, and a one-sided Wilcoxon
p-value for "buyback reduces borrower loss".
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

LT = 0.83
DEBT = 100_000.0
INITIAL_CR = 1.30
WINDOW = 180
N_RUNS = 120
SEED = 20260529


def make_windows(tmp: Path) -> list[Path]:
    base = pd.read_csv(ROOT / "data" / "aave" / "normalized" / "prices.csv").sort_values("block_number")
    series = base["price_usd"].astype(float).to_numpy()
    n = len(series)
    rng = np.random.default_rng(SEED)
    max_start = n - WINDOW
    starts = rng.choice(np.arange(max_start), size=min(N_RUNS, max_start), replace=False)
    starts = sorted(int(s) for s in starts)
    dirs = []
    for i, start in enumerate(starts, start=1):
        ds = tmp / f"w{i:03d}"
        (ds / "normalized").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "block_number": np.arange(WINDOW),
                "timestamp": pd.date_range("2024-01-01", periods=WINDOW, freq="h").astype(str),
                "asset_symbol": "WETH",
                "price_usd": series[start : start + WINDOW],
            }
        ).to_csv(ds / "normalized" / "prices.csv", index=False)
        pd.DataFrame(
            [{"account": "0x", "asset_symbol": "WETH", "collateral_amount": 1.0,
              "debt_amount": DEBT, "liquidation_threshold": LT, "initial_cr": INITIAL_CR}]
        ).to_csv(ds / "normalized" / "positions_initial.csv", index=False)
        dirs.append(ds)
    return dirs


def dyn(enable_buyback: bool, target_hf: float, lltv: float, ratio: float,
        floor: float | None, spread: float) -> dict:
    return {
        "name": "baseline_dynamic" if enable_buyback else "target_hf_no_buyback",
        "dynamic": {
            "lltv": lltv, "target_hf": target_hf,
            "min_close_factor": 0.15, "max_close_factor": 0.60, "cf_slope": 1.6,
            "liquidation_bonus": 0.01, "buyback_ratio": ratio, "buyback_funding": "reborrow",
            "enable_buyback": enable_buyback, "recovery_ltv_gap": 0.08,
            "sell_cooldown_steps": 1, "buy_cooldown_steps": 1 if enable_buyback else 1000000,
            "buyback_hf_floor": floor, "min_buyback_spread": spread,
        },
    }


def run_batch(dirs: list[Path], scenario: list[dict], out: Path, tag: str) -> pd.DataFrame:
    sp = out / f"scn_{tag}.json"
    sp.write_text(json.dumps(scenario), encoding="utf-8")
    rows = []
    for i, ds in enumerate(dirs, start=1):
        rd = run_counterfactual(dataset_dir=ds, scenario_path=sp, output_dir=out / f"runs_{tag}", run_id=f"w{i:03d}")
        m = pd.read_csv(rd / "scenario_metrics.csv")
        m.insert(0, "w", i)
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    tmp = ROOT / "runs" / "buyback_tuning"
    tmp.mkdir(parents=True, exist_ok=True)
    dirs = make_windows(tmp / "windows")

    target_hf, lltv = 1.05, 0.85
    floors = [1.10, 1.15, 1.20, 1.25, 1.30]
    spreads = [0.0, 0.05]
    ratios = [0.20, 0.50]

    results = []
    for floor, spread, ratio in product(floors, spreads, ratios):
        tag = f"f{floor}_s{int(spread*100)}_r{int(ratio*100)}"
        on = run_batch(dirs, [dyn(True, target_hf, lltv, ratio, floor, spread)], tmp, tag + "_on")
        off = run_batch(dirs, [dyn(False, target_hf, lltv, ratio, floor, spread)], tmp, tag + "_off")
        buy = on.set_index("w")["avg_borrower_final_loss_usd"]
        no = off.set_index("w")["avg_borrower_final_loss_usd"]
        d = no - buy  # >0 => buyback better
        nz = d[d != 0]
        p = stats.wilcoxon(nz, alternative="greater").pvalue if len(nz) > 5 else np.nan
        results.append({
            "floor": floor, "spread": spread, "ratio": ratio,
            "mean_diff_no_minus_buy": float(d.mean()),
            "buyback_win_pct": float(100 * (d > 0).mean()),
            "buyback_lose_pct": float(100 * (d < 0).mean()),
            "tie_pct": float(100 * (d == 0).mean()),
            "buy_sells": float(on["total_sell_events"].mean()),
            "no_sells": float(off["total_sell_events"].mean()),
            "buy_events": float(on["total_buy_events"].mean()),
            "worst_bad_debt": float(on["max_bad_debt_usd"].max()),
            "wilcoxon_p_buyback_better": float(p),
        })
        r = results[-1]
        print(f"{tag}: mean_diff={r['mean_diff_no_minus_buy']:+.1f} win={r['buyback_win_pct']:.0f}% "
              f"lose={r['buyback_lose_pct']:.0f}% tie={r['tie_pct']:.0f}% "
              f"sells {r['no_sells']:.2f}->{r['buy_sells']:.2f} p={r['wilcoxon_p_buyback_better']:.2e}")

    out = pd.DataFrame(results).sort_values("mean_diff_no_minus_buy", ascending=False)
    out_path = tmp / "buyback_tuning_results.csv"
    out.to_csv(out_path, index=False)
    print("\nresults_csv", out_path)
    print("\nTOP (buyback most beneficial):")
    print(out.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
