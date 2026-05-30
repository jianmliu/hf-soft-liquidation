"""Regenerate the mechanism-illustration figure (doc/figures/price_grid_loans_tiers.png).

Runs the canonical engine on a representative healthy-start, decline-then-rebound
path and plots the price with the endogenous band, the target-HF SELL events at
the band top (deleveraging), and the guarded BUY events at the band bottom
(restoration), with the health factor on a twin axis. Illustrative operating
point (HF_floor=1.10, eta=1) so the recovery buys are visible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aave_counterfactual_pipeline import run_counterfactual

LT, DEBT, CR0 = 0.83, 100_000.0, 1.30


def main() -> None:
    # Representative path: healthy start, staged decline (forms the band), then rebound.
    down = np.linspace(3000, 2040, 46)
    up = np.linspace(2040, 2760, 44)
    prices = np.concatenate([down, up])
    n = len(prices)

    tmp = ROOT / "runs" / "mechanism_fig"
    (tmp / "normalized").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "block_number": np.arange(n),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h").astype(str),
        "asset_symbol": "WETH", "price_usd": prices,
    }).to_csv(tmp / "normalized" / "prices.csv", index=False)
    pd.DataFrame([{"account": "0xdemo", "asset_symbol": "WETH", "collateral_amount": 1.0,
                   "debt_amount": DEBT, "liquidation_threshold": LT, "initial_cr": CR0}]).to_csv(
        tmp / "normalized" / "positions_initial.csv", index=False)

    scn = [{"name": "baseline_dynamic", "dynamic": {
        "lltv": 0.85, "target_hf": 1.05, "min_close_factor": 0.15, "max_close_factor": 0.60,
        "cf_slope": 1.6, "liquidation_bonus": 0.01, "buyback_ratio": 1.0, "buyback_funding": "reborrow",
        "enable_buyback": True, "recovery_ltv_gap": 0.08, "sell_cooldown_steps": 1, "buy_cooldown_steps": 1,
        "buyback_hf_floor": 1.10, "min_buyback_spread": 0.05}}]
    sp = tmp / "scn.json"
    sp.write_text(json.dumps(scn), encoding="utf-8")
    rd = run_counterfactual(dataset_dir=tmp, scenario_path=sp, output_dir=tmp / "runs", run_id="demo")

    ev = pd.read_csv(rd / "event_log.csv")
    sells = ev[ev.event == "SELL"]
    buys = ev[ev.event == "BUY"]

    # Reconstruct the HF path for the plotted scenario from the event ledger.
    coll, debt = DEBT * CR0 / prices[0], DEBT
    hf_path = []
    ev_by_t = {int(t): g for t, g in ev.groupby("block_number")}
    for t in range(n):
        p = prices[t]
        if t in ev_by_t:
            for _, e in ev_by_t[t].iterrows():
                if e.event == "SELL":
                    cs = (1.0 + 0.01) * e.debt_repaid / p
                    coll -= cs; debt -= e.debt_repaid
                else:
                    coll += e.collateral_amount; debt += e.collateral_amount * p
        hf_path.append(coll * p * LT / debt if debt > 0 else np.nan)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    t = np.arange(n)
    ax.plot(t, prices, color="black", lw=1.6, label="Collateral price", zorder=3)

    # Endogenous band = span of executed sell prices.
    if not sells.empty:
        ax.axhspan(sells.price_usd.min(), sells.price_usd.max(), color="#f4a261", alpha=0.18,
                   zorder=0, label="Endogenous sell band")
        ax.axhline(sells.price_usd.max(), color="#e76f51", ls="--", lw=0.8, alpha=0.7)
        ax.axhline(sells.price_usd.min(), color="#e76f51", ls="--", lw=0.8, alpha=0.7)

    ax.scatter(sells.block_number, sells.price_usd, marker="v", s=90, color="#c1121f",
               edgecolor="black", lw=0.5, zorder=5, label=f"Target-HF SELL (band top), n={len(sells)}")
    ax.scatter(buys.block_number, buys.price_usd, marker="^", s=70, color="#2a9d8f",
               edgecolor="black", lw=0.5, zorder=5, label=f"Guarded BUY (band bottom), n={len(buys)}")

    ax.annotate("deleverage as price falls", xy=(sells.block_number.iloc[0], sells.price_usd.iloc[0]),
                xytext=(2, 2820), fontsize=9, color="#c1121f")
    if not buys.empty:
        ax.annotate("restore on rebound", xy=(buys.block_number.iloc[-1], buys.price_usd.iloc[-1]),
                    xytext=(58, 2150), fontsize=9, color="#2a9d8f")

    ax.set_xlabel("Time step"); ax.set_ylabel("Price (USD)")
    ax.set_title("Band-based soft liquidation: sell at the band top, buy back at the band bottom")

    ax2 = ax.twinx()
    ax2.plot(t, hf_path, color="#264653", lw=1.1, alpha=0.7, label="Health factor (right)")
    ax2.axhline(1.0, color="grey", ls=":", lw=0.8)
    ax2.set_ylabel("Health factor")
    ax2.set_ylim(0.9, 1.6)

    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout()

    out = ROOT / "doc" / "figures" / "price_grid_loans_tiers.png"
    fig.savefig(out, dpi=200)
    print("wrote", out, "| sells", len(sells), "buys", len(buys))


if __name__ == "__main__":
    main()
