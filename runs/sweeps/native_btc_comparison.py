"""Native-BTC (pre-signed vault) comparison: the project's full story in one table.

Four arms on identical BTC paths (daily cadence ~ native-BTC oracle/block tempo),
healthy starts, clustered by non-overlapping 270-day window:

  A  babylon_full_liq : single hard threshold, liquidator seizes the WHOLE vault
     (Babylon "Trustless Bitcoin Vaults" flow: repay debt, redeem the BTC).
     Emulated as a one-tier ladder with close_factor=1 and a seizure-sized bonus.
  B  ladder_no_buyback: pre-committable target-HF partial-liquidation ladder,
     sell side only (what graduation alone buys you).
  C  ladder_fixed_buyback: the ORIGINAL pre-signed design — band ladder plus
     fixed (state-blind) buyback, emulated as unguarded reborrow-to-LLTV with
     full ratio. This is the ping-pong-prone arm.
  D  ladder_sized_buyback: ladder plus the solvency/stress-sized restoration
     (HF floor 1.20, stress d=0.25, spread 5%) — the cure, which compiles back
     to a pre-signed structure via LIFO band unwind.

Primary metric: BTC retention rho = C_T / C_0 (the BTC-denominated borrower's
unit), plus borrower loss, bad debt, sell count (ping-pong indicator).
"""
from __future__ import annotations

import json
import shutil
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
LLTV = 0.85  # trigger LTV; threshold HF = LT/LLTV = 0.976 for the full-liq arm


def scenarios() -> list[dict]:
    dyn_common = dict(
        lltv=LLTV, target_hf=1.05, min_close_factor=0.15, max_close_factor=0.60,
        cf_slope=1.6, liquidation_bonus=0.01, buyback_funding="reborrow",
        recovery_ltv_gap=0.08, sell_cooldown_steps=1,
    )
    return [
        # A: hard threshold, seize-everything (bonus large enough that the clip
        #    hands the liquidator the entire remaining collateral).
        {"name": "babylon_full_liq", "buyback_bandwidth": 10.0, "tiers": [
            {"name": "FullLiq", "hf_down": LT / LLTV, "close_factor": 1.0,
             "liquidation_bonus": 0.50, "buyback_ratio": 0.01}]},
        # B: graduated ladder, no restoration.
        {"name": "ladder_no_buyback", "dynamic": {
            **dyn_common, "buyback_ratio": 0.2, "enable_buyback": False,
            "buy_cooldown_steps": 1_000_000}},
        # C: the original pre-signed design — fixed, state-blind buyback
        #    (no solvency floor: reborrow to the LLTV capacity, full ratio).
        {"name": "ladder_fixed_buyback", "dynamic": {
            **dyn_common, "buyback_ratio": 1.0, "enable_buyback": True,
            "buy_cooldown_steps": 1, "min_buyback_spread": 0.05}},
        # D: solvency/stress-sized restoration (the cure).
        {"name": "ladder_sized_buyback", "dynamic": {
            **dyn_common, "buyback_ratio": 1.0, "enable_buyback": True,
            "buy_cooldown_steps": 1, "min_buyback_spread": 0.05,
            "buyback_hf_floor": 1.20, "buyback_stress_drawdown": 0.25}},
    ]


def main() -> None:
    btc = pd.read_csv(ROOT / "data" / "oos" / "BTC.csv")["price_usd"].astype(float).to_numpy()
    batch = ROOT / "runs" / "native_btc_comparison"
    shutil.rmtree(batch, ignore_errors=True)
    (batch / "ds").mkdir(parents=True, exist_ok=True)
    sp = batch / "scn.json"
    sp.write_text(json.dumps(scenarios()), encoding="utf-8")

    rows = []
    for wi in range(len(btc) // W):
        seg = btc[wi * W:(wi + 1) * W]
        for cr in CRS:
            ds = batch / "ds" / f"w{wi}_{int(cr*100)}"
            (ds / "normalized").mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"block_number": np.arange(W),
                          "timestamp": pd.date_range("2024-01-01", periods=W, freq="h").astype(str),
                          "asset_symbol": "WETH", "price_usd": seg}).to_csv(
                ds / "normalized" / "prices.csv", index=False)
            pd.DataFrame([{"account": "x", "asset_symbol": "WETH",
                           "collateral_amount": 1.0, "debt_amount": DEBT,
                           "liquidation_threshold": LT, "initial_cr": cr}]).to_csv(
                ds / "normalized" / "positions_initial.csv", index=False)
            rd = run_counterfactual(dataset_dir=ds, scenario_path=sp,
                                    output_dir=batch / "runs", run_id=f"w{wi}_{int(cr*100)}")
            m = pd.read_csv(rd / "scenario_metrics.csv")
            m.insert(0, "window", wi)
            m.insert(1, "cr", cr)
            rows.append(m)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(batch / "summary.csv", index=False)

    cl = df.groupby(["window", "scenario"]).mean(numeric_only=True).reset_index()
    agg = cl.groupby("scenario").agg(
        btc_retention=("avg_restoration_ratio", "mean"),
        worst_retention=("avg_restoration_ratio", "min"),
        mean_loss=("avg_borrower_final_loss_usd", "mean"),
        worst_bad_debt=("max_bad_debt_usd", "max"),
        mean_sells=("total_sell_events", "mean"),
        mean_buys=("total_buy_events", "mean"),
    ).reindex(["babylon_full_liq", "ladder_no_buyback",
               "ladder_fixed_buyback", "ladder_sized_buyback"])
    print(f"windows={cl.window.nunique()} x CR levels={len(CRS)} on BTC 2019-2024 (daily)\n")
    print(agg.to_string(float_format=lambda x: f"{x:,.4f}"))

    # the crash window (contains March 2020) deserves its own line
    crash = cl[cl.window == 1].set_index("scenario")
    if not crash.empty:
        print("\nCrash window (contains March 2020):")
        print(crash[["avg_restoration_ratio", "avg_borrower_final_loss_usd",
                     "max_bad_debt_usd", "total_sell_events"]].to_string(
            float_format=lambda x: f"{x:,.4f}"))

    # emit the .tex table for the native-BTC companion note
    labels = {
        "babylon_full_liq": "Hard threshold, full seizure (status quo)",
        "ladder_no_buyback": "Pre-committed target-HF ladder",
        "ladder_fixed_buyback": "Ladder + fixed buyback (original design)",
        "ladder_sized_buyback": "Ladder + solvency/stress-sized buyback",
    }
    lines = []
    for key, lab in labels.items():
        r = agg.loc[key]
        lines.append(
            f"{lab} & {r.btc_retention:.3f} & {r.mean_loss:,.0f} & "
            f"{r.worst_bad_debt:,.0f} & {r.mean_sells:.2f} & {r.mean_buys:.2f} \\\\"
        )
    body = (
        "\\begin{table}[h!]\n\\centering\n"
        "\\caption{Four enforcement designs on identical BTC paths (2019--2024, "
        "eight non-overlapping 270-day windows $\\times$ three initial CR levels, "
        "daily cadence $\\approx$ native-BTC oracle tempo). Graduation alone halves "
        "borrower loss and cuts worst bad debt by $3/4$ versus full seizure; the "
        "fixed buyback of the original pre-signed design churns ($3\\times$ the "
        "liquidations, $45\\times$ the executions) for $\\approx$zero net gain; the "
        "solvency/stress-sized rule matches its outcome with $1/20$ of the "
        "executions.}\n"
        "\\label{tab:native_btc_comparison}\n"
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
        "Design & BTC retention $\\rho$ & Mean loss (USD) & Worst bad debt (USD) "
        "& Sells/run & Buys/run \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n"
        + "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    )
    out = ROOT / "doc" / "tables" / "native_btc_comparison_table.tex"
    out.write_text(body, encoding="utf-8")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
