"""Generate the out-of-sample validation tables for the manuscript.

Consumes the OOS sweep outputs and emits .tex tables:
  - oos_sell_frontier_table.tex   (sell-side dial: borrower loss vs bad debt)
  - oos_buy_principles_table.tex   (buy-side: stress-sizing vs safety/restoration)
  - oos_summary stats are printed for inline citation.
Run after oos_validation.py, sell_side_oos_sweep.py, optimize_buy_principles.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "doc" / "tables"
SWEEP = ROOT / "runs" / "sweeps"


def money(x: float) -> str:
    return f"{x:,.0f}"


def write(name: str, body: str) -> None:
    (TABLES / name).write_text(body, encoding="utf-8")
    print("wrote", TABLES / name)


def sell_frontier() -> None:
    f = SWEEP / "sell_side_oos_results.csv"
    if not f.exists():
        print("skip sell frontier (run sell_side_oos_sweep.py)")
        return
    df = pd.read_csv(f)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{r.lltv:.2f} & {r.target_hf:.2f} & {r.trigger_hf:.3f} & {money(r.mean_loss)} & "
            f"{money(r.bad_debt_max)} & {money(r.bad_debt_total)} & {int(r.windows_with_bad_debt)}/95 & {r.mean_sells:.1f} \\\\"
        )
    body = (
        "\\begin{table}[h!]\n\\centering\n"
        "\\caption{Out-of-sample sell-side governance dial (15 assets, 2019--2024, non-overlapping 270-day windows; no-buyback policy). Earlier triggers (lower $\\mathrm{LLTV}$, with $\\mathrm{HF}^{\\star}$ raised for feasibility) trade borrower opportunity cost for protocol solvency: bad debt falls from $9{,}419$ to $265$ USD as $\\mathrm{LLTV}$ goes $0.85\\to0.70$, at a ${+}27\\%$ borrower-loss cost. Every row Pareto-dominates the fixed-CF baseline (mean loss $48{,}389$, worst bad debt $26{,}555$, $31/95$ windows).}\n"
        "\\label{tab:oos_sell_frontier}\n"
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}rrrrrrrr@{}}\n\\toprule\n"
        "$\\mathrm{LLTV}$ & $\\mathrm{HF}^{\\star}$ & trigger HF & Mean loss & Worst bad debt & Total bad debt & Windows w/ bad debt & Mean sells \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n"
        + "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    )
    write("oos_sell_frontier_table.tex", body)


def buy_principles() -> None:
    f = SWEEP / "buy_principles_oos_results.csv"
    if not f.exists():
        print("skip buy principles (run optimize_buy_principles.py)")
        return
    df = pd.read_csv(f)
    # one representative row per stress-drawdown level (timing adds ~nothing)
    df = df.sort_values(["drawdown", "lookback"]).drop_duplicates("drawdown", keep="first")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{r.drawdown:.2f} & {int(r.active)}/95 & {r.extra_sells_mean:+.2f} & {r.extra_sells_max:+.1f} & "
            f"{int(r.rho_win)}/{int(r.rho_lose)} & {int(r.usd_win)}/{int(r.usd_lose)} & {r.usd_p:.3f} \\\\"
        )
    body = (
        "\\begin{table}[h!]\n\\centering\n"
        "\\caption{Out-of-sample buy-side principle dial (15 assets; buyback vs no-buyback, clustered by window). The stress-drawdown $d$ (size so post-buy HF$\\ge 1$ after a further $d$ drop) is the effective lever: raising it cuts buyback-induced re-liquidation and cleans up restoration/USD outcomes, at the cost of activation. A confirmed-upturn timing gate (not shown) adds little---one cannot reliably distinguish a bottom from a dead-cat bounce ex ante, so conservative \\emph{sizing} beats \\emph{timing}.}\n"
        "\\label{tab:oos_buy_principles}\n"
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}rrrrrrr@{}}\n\\toprule\n"
        "Stress $d$ & Active & Extra sells (mean) & Extra sells (max) & Restoration W/L & USD W/L (active) & USD $p$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n"
        + "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    )
    write("oos_buy_principles_table.tex", body)


def summary() -> None:
    f = ROOT / "runs" / "oos_validation" / "oos_clustered_stats.json"
    if f.exists():
        print("\nOOS headline (optimized candidate):")
        print(json.dumps(json.loads(f.read_text()), indent=2))


def main() -> None:
    sell_frontier()
    buy_principles()
    summary()


if __name__ == "__main__":
    main()
