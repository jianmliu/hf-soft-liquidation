"""Independent-window stress batch (addresses sample-dependence).

The earlier real-window batch drew 200 overlapping windows from a single 1101-point
ETH series, so the paired differences were autocorrelated and the Wilcoxon
p-values overstated. This batch instead uses NON-OVERLAPPING windows across TWO
assets (ETH and BTC), and clusters the per-CR runs to the window level, so the
unit of analysis is an independent window-path.

For each (asset, non-overlapping window) it runs the three policies at three CR
levels, averages the buyback-vs-no-buyback difference over CR within the window
(one clustered observation per window), and reports a Wilcoxon test over the
independent window-level differences plus the number of independent windows.
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
SCN = {"buy": "baseline_dynamic", "no": "target_hf_no_buyback", "fix": "traditional_fixed_cf"}
# (window length, buyback HF floor) combos to characterize honestly:
# safe vs aggressive floor, across window lengths long enough to contain a cycle.
COMBOS = [(180, 1.20), (180, 1.10), (120, 1.20), (120, 1.10)]


def load_series() -> dict[str, np.ndarray]:
    eth = pd.read_csv(ROOT / "data" / "aave" / "normalized" / "prices.csv").sort_values("block_number")["price_usd"].astype(float).to_numpy()
    out = {"ETH": eth}
    btc_path = ROOT / "BTC_1Y_graph_coinmarketcap.csv"
    if btc_path.exists():
        df = pd.read_csv(btc_path, sep=None, engine="python")
        col = "price" if "price" in df.columns else df.select_dtypes("number").columns[0]
        out["BTC"] = df[col].astype(float).to_numpy()
    return out


def scenario_with_floor(floor: float) -> list[dict]:
    scn = json.loads((ROOT / "runs" / "sweeps" / "scenario_candidate_best.json").read_text(encoding="utf-8"))
    for s in scn:
        if "dynamic" in s and s["dynamic"].get("enable_buyback"):
            s["dynamic"]["buyback_hf_floor"] = floor
            s["dynamic"]["buyback_ratio"] = 1.0  # let the floor be the sole governor
    return scn


def run_combo(series: dict[str, np.ndarray], W: int, floor: float, batch_dir: Path) -> dict:
    sp = batch_dir / f"scn_{W}_{int(floor*100)}.json"
    sp.write_text(json.dumps(scenario_with_floor(floor)), encoding="utf-8")
    rows = []
    for asset, s in series.items():
        for wi in range(len(s) // W):
            seg = s[wi * W:(wi + 1) * W]
            for cr in CRS:
                ds = batch_dir / "ds" / f"{asset}_{W}_w{wi}_{int(cr*100)}"
                (ds / "normalized").mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"block_number": np.arange(W),
                              "timestamp": pd.date_range("2024-01-01", periods=W, freq="h").astype(str),
                              "asset_symbol": "WETH", "price_usd": seg}).to_csv(ds / "normalized" / "prices.csv", index=False)
                pd.DataFrame([{"account": "x", "asset_symbol": "WETH", "collateral_amount": 1.0,
                               "debt_amount": DEBT, "liquidation_threshold": LT, "initial_cr": cr}]).to_csv(
                    ds / "normalized" / "positions_initial.csv", index=False)
                rd = run_counterfactual(dataset_dir=ds, scenario_path=sp, output_dir=batch_dir / "runs" / f"{W}_{int(floor*100)}",
                                        run_id=f"{asset}_w{wi}_{int(cr*100)}")
                m = pd.read_csv(rd / "scenario_metrics.csv")
                m.insert(0, "asset", asset); m.insert(1, "window", wi); m.insert(2, "cr", cr)
                rows.append(m)
    df = pd.concat(rows, ignore_index=True)
    # Window-level clustering: average the buyback-vs-no difference over CR within
    # each (asset, window), giving one independent observation per window-path.
    piv_loss = df.pivot_table(index=["asset", "window"], columns="scenario", values="avg_borrower_final_loss_usd")
    piv_rho = df.pivot_table(index=["asset", "window"], columns="scenario", values="avg_restoration_ratio")
    piv_sell = df.pivot_table(index=["asset", "window"], columns="scenario", values="total_sell_events")
    piv_bd = df.pivot_table(index=["asset", "window"], columns="scenario", values="max_bad_debt_usd")

    d_loss = (piv_loss[SCN["no"]] - piv_loss[SCN["buy"]])   # >0 => buyback reduces loss
    d_rho = (piv_rho[SCN["buy"]] - piv_rho[SCN["no"]])      # >0 => buyback restores more
    nz_l = d_loss[d_loss.abs() > 1e-9]
    nz_r = d_rho[d_rho.abs() > 1e-12]
    p_loss = stats.wilcoxon(nz_l, alternative="greater").pvalue if len(nz_l) > 5 else float("nan")
    p_rho = stats.wilcoxon(nz_r, alternative="greater").pvalue if len(nz_r) > 5 else float("nan")

    return {
        "window_len": W, "hf_floor": floor,
        "n_independent_windows": int(len(d_loss)), "n_assets": int(df["asset"].nunique()),
        "loss_mean_reduction": float(d_loss.mean()),
        "loss_win": int((d_loss > 1e-9).sum()), "loss_lose": int((d_loss < -1e-9).sum()),
        "loss_informative": int(len(nz_l)), "wilcoxon_p_loss": float(p_loss),
        "rho_mean_gain": float(d_rho.mean()),
        "rho_win": int((d_rho > 1e-9).sum()), "rho_lose": int((d_rho < -1e-9).sum()),
        "rho_informative": int(len(nz_r)), "wilcoxon_p_rho": float(p_rho),
        "extra_sells_mean": float((piv_sell[SCN["buy"]] - piv_sell[SCN["no"]]).mean()),
        "worst_bad_debt_buy": float(piv_bd[SCN["buy"]].max()),
        "mean_loss_fix": float(piv_loss[SCN["fix"]].mean()),
        "mean_loss_no": float(piv_loss[SCN["no"]].mean()),
        "mean_loss_buy": float(piv_loss[SCN["buy"]].mean()),
    }


def main() -> None:
    series = load_series()
    batch_dir = ROOT / "runs" / "indep_window_batch"
    import shutil
    shutil.rmtree(batch_dir, ignore_errors=True)
    (batch_dir / "ds").mkdir(parents=True, exist_ok=True)
    results = [run_combo(series, W, floor, batch_dir) for (W, floor) in COMBOS]
    out = pd.DataFrame(results)
    out.to_csv(ROOT / "runs" / "sweeps" / "independent_window_results.csv", index=False)
    print(out.to_string(index=False))
    print("\nNote: unit of analysis is a non-overlapping window-path across ETH+BTC")
    print("(n_independent_windows). Wilcoxon over these independent window-level diffs.")


if __name__ == "__main__":
    main()
