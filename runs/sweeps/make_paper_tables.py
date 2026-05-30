"""Single reproducible generator for the manuscript's tables and statistics.

Consumes the re-run experiment outputs and emits the .tex tables that main.tex
\\input{}s, plus a stats summary (paired Wilcoxon p-value and bootstrap CI).
This closes the reproducibility gap: previously the .tex tables had no
generating script. Run AFTER the sweep, real-window batch, and delta replay:

    PYTHONHASHSEED=0 python3 runs/sweeps/historical_window_sweep.py
    PYTHONHASHSEED=0 python3 runs/sweeps/real_window_batch.py
    PYTHONHASHSEED=0 python3 runs/sweeps/make_paper_tables.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aave_counterfactual_pipeline import historical_backtest

TABLES = ROOT / "doc" / "tables"
SWEEP = ROOT / "runs" / "sweeps"
DATASET = ROOT / "data" / "aave"
SCN = {"buy": "baseline_dynamic", "no": "target_hf_no_buyback", "fix": "traditional_fixed_cf"}


def latest(pattern: str) -> Path:
    matches = sorted(glob.glob(str(ROOT / pattern)))
    if not matches:
        raise SystemExit(f"no files match {pattern}")
    return Path(matches[-1])


def money(x: float) -> str:
    return f"{x:,.2f}"


def write(name: str, body: str) -> None:
    (TABLES / name).write_text(body, encoding="utf-8")
    print("wrote", TABLES / name)


# --------------------------------------------------------------------------
# Batch tables (real-window stress batch): borrower loss, soft vs traditional,
# three-strategy, plus paired Wilcoxon + bootstrap CI.
# --------------------------------------------------------------------------
def batch_tables() -> dict:
    f = latest("runs/realbatch_*/batch_summary.csv")
    df = pd.read_csv(f)
    loss = df.pivot_table(index="batch_run_id", columns="scenario", values="avg_borrower_final_loss_usd")
    hf = df.pivot_table(index="batch_run_id", columns="scenario", values="avg_min_hf")
    bd = df.pivot_table(index="batch_run_id", columns="scenario", values="max_bad_debt_usd")
    rho = df.pivot_table(index="batch_run_id", columns="scenario", values="avg_restoration_ratio")
    n = len(loss)
    buy, no, fix = loss[SCN["buy"]], loss[SCN["no"]], loss[SCN["fix"]]

    # --- borrower_loss_batch_120 (proposed policy = guarded target-HF + buyback)
    q = buy.quantile
    zero_share = float((buy <= 1e-9).mean())
    rows = [
        ("Batch runs (count)", f"{n}"),
        ("Mean borrower final loss (USD)", money(buy.mean())),
        ("Median borrower final loss (USD)", money(buy.median())),
        ("25th percentile (USD)", money(q(0.25))),
        ("75th percentile (USD)", money(q(0.75))),
        ("90th percentile (USD)", money(q(0.90))),
        ("Minimum (USD)", money(buy.min())),
        ("Maximum (USD)", money(buy.max())),
        ("Share of zero-loss runs", f"{zero_share*100:.2f}\\%"),
        ("Worst-case run bad debt (USD)", money(bd[SCN['buy']].max())),
    ]
    body = (
        "\\begin{table}[h!]\n\\centering\n"
        "\\caption{Borrower final-loss statistics across "
        f"{n} real-window stress runs (proposed guarded target-HF with buyback).}}\n"
        "\\label{tab:borrower_loss_batch_120}\n"
        "\\begin{tabular}{@{}lr@{}}\n\\toprule\nMetric & Value \\\\\n\\midrule\n"
        + "".join(f"{k} & {v} \\\\\n" for k, v in rows)
        + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    write("borrower_loss_batch_120_table.tex", body)

    # --- soft_vs_traditional (proposed buyback vs fixed-CF)
    red = fix - buy  # >0 => soft better
    win = float((red > 1e-9).mean()); lose = float((red < -1e-9).mean()); tie = float((red.abs() <= 1e-9).mean())
    rows = [
        ("Paired runs (count)", f"{n}"),
        ("Mean borrower loss, soft (USD)", money(buy.mean())),
        ("Mean borrower loss, traditional (USD)", money(fix.mean())),
        ("Mean loss reduction (traditional $-$ soft, USD)", money(red.mean())),
        ("Median loss reduction (USD)", money(red.median())),
        ("25th percentile loss reduction (USD)", money(red.quantile(0.25))),
        ("75th percentile loss reduction (USD)", money(red.quantile(0.75))),
        ("Minimum loss reduction (USD)", money(red.min())),
        ("Maximum loss reduction (USD)", money(red.max())),
        ("Soft win-rate (loss lower than traditional)", f"{win*100:.2f}\\%"),
        ("Soft lose-rate (loss higher than traditional)", f"{lose*100:.2f}\\%"),
        ("Tie-rate", f"{tie*100:.2f}\\%"),
    ]
    body = (
        "\\begin{table}[h!]\n\\centering\n"
        f"\\caption{{Soft vs. traditional borrower final-loss comparison across {n} paired runs.}}\n"
        "\\label{tab:soft_vs_traditional_borrower_loss}\n"
        "\\begin{tabular}{@{}lr@{}}\n\\toprule\nMetric & Value \\\\\n\\midrule\n"
        + "".join(f"{k} & {v} \\\\\n" for k, v in rows)
        + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    write("soft_vs_traditional_borrower_loss_table.tex", body)

    # --- three_strategy + pairwise win rates
    def winrate(a, b):  # fraction a < b (a better)
        d = b - a
        return float((d > 1e-9).mean()), float((d.abs() <= 1e-9).mean())
    w_bf, t_bf = winrate(buy, fix)
    w_bn, t_bn = winrate(buy, no)
    w_nf, t_nf = winrate(no, fix)
    srows = [
        ("Traditional fixed CF", fix.mean(), fix.median(), hf[SCN['fix']].mean(), rho[SCN['fix']].mean(), bd[SCN['fix']].max()),
        ("Target-HF (no buyback)", no.mean(), no.median(), hf[SCN['no']].mean(), rho[SCN['no']].mean(), bd[SCN['no']].max()),
        ("Target-HF + buyback (guarded)", buy.mean(), buy.median(), hf[SCN['buy']].mean(), rho[SCN['buy']].mean(), bd[SCN['buy']].max()),
    ]
    body = (
        "\\begin{table}[h!]\n\\centering\n"
        f"\\caption{{Three-strategy comparison on borrower final loss ({n} paired runs).}}\n"
        "\\label{tab:three_strategy_borrower_loss}\n"
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
        "Strategy & Mean loss (USD) & Median loss (USD) & Mean min HF & Mean restoration $\\rho$ & Worst bad debt (USD) \\\\\n\\midrule\n"
        + "".join(f"{a} & {money(b)} & {money(c)} & {d:.4f} & {r:.4f} & {money(e)} \\\\\n" for a, b, c, d, r, e in srows)
        + "\\bottomrule\n\\end{tabular}\n}\n\n\\vspace{0.35em}\n\n"
        "\\begin{tabular}{@{}p{0.64\\textwidth}r@{}}\n\\toprule\n"
        "Pairwise borrower-loss win-rate (lower is better) & Value \\\\\n\\midrule\n"
        f"Target-HF + buyback vs Traditional fixed CF & {w_bf*100:.2f}\\% (tie {t_bf*100:.2f}\\%) \\\\\n"
        f"Target-HF + buyback vs Target-HF (no buyback) & {w_bn*100:.2f}\\% (tie {t_bn*100:.2f}\\%) \\\\\n"
        f"Target-HF (no buyback) vs Traditional fixed CF & {w_nf*100:.2f}\\% (tie {t_nf*100:.2f}\\%) \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    write("three_strategy_borrower_loss_table.tex", body)

    # --- paired stats: buyback vs no-buyback (one-sided Wilcoxon + bootstrap CI)
    d = no - buy  # >0 => buyback reduces loss
    nz = d[d.abs() > 1e-9]
    if len(nz) > 5:
        p_better = float(stats.wilcoxon(nz, alternative="greater").pvalue)
        p_worse = float(stats.wilcoxon(nz, alternative="less").pvalue)
    else:
        p_better = p_worse = float("nan")
    rng = np.random.default_rng(2026)
    boot = np.array([rng.choice(d.to_numpy(), size=len(d), replace=True).mean() for _ in range(10000)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    # restoration ratio: does buyback restore more collateral than no-buyback?
    dr = rho[SCN["buy"]] - rho[SCN["no"]]
    nzr = dr[dr.abs() > 1e-12]
    p_rho = float(stats.wilcoxon(nzr, alternative="greater").pvalue) if len(nzr) > 5 else float("nan")
    stats_out = {
        "batch_file": str(f), "n": n,
        "mean_loss": {k: float(loss[SCN[k]].mean()) for k in SCN},
        "mean_restoration_ratio": {k: float(rho[SCN[k]].mean()) for k in SCN},
        "buyback_minus_nobuyback_mean_reduction_usd": float(d.mean()),
        "wilcoxon_p_buyback_reduces_loss": p_better,
        "wilcoxon_p_buyback_increases_loss": p_worse,
        "bootstrap95_ci_mean_reduction_usd": ci,
        "nonzero_paired": int(len(nz)),
        "buyback_win": int((d > 1e-9).sum()), "buyback_lose": int((d < -1e-9).sum()),
        "restoration_mean_gain_buyback_vs_no": float(dr.mean()),
        "wilcoxon_p_buyback_restores_more": p_rho,
        "restoration_informative": int(len(nzr)),
    }
    return stats_out


def paired_summary(batch_glob: str) -> dict:
    """Buyback-vs-no-buyback paired summary for a given batch (loss + restoration)."""
    f = latest(batch_glob)
    df = pd.read_csv(f)
    loss = df.pivot_table(index="batch_run_id", columns="scenario", values="avg_borrower_final_loss_usd")
    rho = df.pivot_table(index="batch_run_id", columns="scenario", values="avg_restoration_ratio")
    d = loss[SCN["no"]] - loss[SCN["buy"]]
    nz = d[d.abs() > 1e-9]
    p = float(stats.wilcoxon(nz, alternative="greater").pvalue) if len(nz) > 5 else float("nan")
    return {
        "n": len(d), "informative": int(len(nz)),
        "win": int((d > 1e-9).sum()), "lose": int((d < -1e-9).sum()),
        "mean_reduction": float(d.mean()), "wilcoxon_p": p,
        "rho_buy": float(rho[SCN["buy"]].mean()), "rho_no": float(rho[SCN["no"]].mean()),
        "rho_fix": float(rho[SCN["fix"]].mean()),
        "loss_buy": float(loss[SCN["buy"]].mean()), "loss_no": float(loss[SCN["no"]].mean()),
        "loss_fix": float(loss[SCN["fix"]].mean()),
    }


def regime_comparison_table() -> dict:
    real = paired_summary("runs/realbatch_*/batch_summary.csv")
    crash = paired_summary("runs/synbatch_*/batch_summary.csv")

    def pfmt(p):
        if p != p:
            return "---"
        mant, exp = f"{p:.1e}".split("e")
        return f"${mant}\\times10^{{{int(exp)}}}$"

    rows = [
        ("Runs (paired)", f"{real['n']}", f"{crash['n']}"),
        ("Informative (non-tie) pairs", f"{real['informative']}", f"{crash['informative']}"),
        ("Buyback win / lose", f"{real['win']} / {real['lose']}", f"{crash['win']} / {crash['lose']}"),
        ("Mean loss reduction, no-buy $-$ buy (USD)", money(real['mean_reduction']), money(crash['mean_reduction'])),
        ("Wilcoxon $p$ (buyback reduces loss)", pfmt(real['wilcoxon_p']), pfmt(crash['wilcoxon_p'])),
        ("Mean restoration $\\rho$: buyback", f"{real['rho_buy']:.4f}", f"{crash['rho_buy']:.4f}"),
        ("Mean restoration $\\rho$: no-buyback", f"{real['rho_no']:.4f}", f"{crash['rho_no']:.4f}"),
        ("Mean restoration $\\rho$: fixed CF", f"{real['rho_fix']:.4f}", f"{crash['rho_fix']:.4f}"),
    ]
    body = (
        "\\begin{table}[h!]\n\\centering\n"
        "\\caption{Robustness of the guarded buyback across two stress regimes: real sampled ETH windows vs.\\ synthetic deep-crash paths. Buyback reduces borrower loss and restores more collateral in both, with no induced re-liquidation and zero bad debt.}\n"
        "\\label{tab:stress_regime_comparison}\n"
        "\\begin{tabular}{@{}lrr@{}}\n\\toprule\n"
        "Metric & Real windows & Synthetic crash \\\\\n\\midrule\n"
        + "".join(f"{a} & {b} & {c} \\\\\n" for a, b, c in rows)
        + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    write("stress_regime_comparison_table.tex", body)
    return {"real": real, "crash": crash}


# --------------------------------------------------------------------------
# Historical-window tables: summary + delta sensitivity (per-window win rates).
# --------------------------------------------------------------------------
def per_window(summary_path: Path) -> pd.DataFrame:
    s = pd.read_csv(summary_path)
    return s.pivot_table(index="window_id", columns="scenario",
                         values=["avg_borrower_final_loss_usd", "total_sell_events", "total_buy_events", "max_bad_debt_usd"])


def run_backtest(scenario_path: Path, tag: str) -> Path:
    sp, agg, _ = historical_backtest(
        dataset_dir=DATASET, scenario_path=scenario_path, output_dir=SWEEP / "paper_runs",
        window_size=120, window_step=24, max_windows=24, loan_mid_starts=True, loan_min_duration_blocks=36,
    )
    return sp


def hist_summary_table() -> None:
    sweep_csv = latest("runs/sweeps/historical_window_sweep_results_*.csv")
    csv_name_tex = sweep_csv.name.replace("_", r"\_")
    sw = pd.read_csv(sweep_csv)
    n_cand = len(sw)
    n_target = int((sw["delta_no_minus_fixed"] < 0).sum())
    n_buy = int((sw["delta_buy_minus_no"] < 0).sum())
    n_feas = int(sw["bad_debt_feasible"].sum())

    best = sw.sort_values(["objective", "mean_loss_buyback"]).iloc[0]
    sp = run_backtest(SWEEP / "scenario_candidate_best.json", "best")
    pw = per_window(sp)
    loss = pw["avg_borrower_final_loss_usd"]
    sells = pw["total_sell_events"]
    active = sells[SCN["buy"]] > 0
    n_active = int(active.sum())
    bw = ((loss[SCN["no"]] - loss[SCN["buy"]]) > 1e-9) & active
    nf = ((loss[SCN["fix"]] - loss[SCN["no"]]) > 1e-9) & active
    bw_rate = (bw.sum() / n_active * 100) if n_active else 0.0
    nf_rate = (nf.sum() / n_active * 100) if n_active else 0.0
    worst_bd = float(pw["max_bad_debt_usd"][SCN["buy"]].max())

    rows = [
        ("Total rolling windows per candidate", f"{int(best['windows'])}"),
        ("Active windows (non-zero liquidation outcomes)", f"{n_active}"),
        ("Buyback vs no-buyback win-rate (active windows)", f"{bw_rate:.2f}\\% ({int(bw.sum())}/{n_active})"),
        ("Target-HF-no-buyback vs fixed-CF win-rate (active windows)", f"{nf_rate:.2f}\\% ({int(nf.sum())}/{n_active})"),
        ("Candidates in constrained sweep", f"{n_cand}"),
        ("Candidates with buyback mean loss $<$ no-buyback mean loss", f"{n_buy/n_cand*100:.2f}\\% ({n_buy}/{n_cand})"),
        ("Candidates with target-HF-no-buyback mean loss $<$ fixed-CF mean loss", f"{n_target/n_cand*100:.2f}\\% ({n_target}/{n_cand})"),
        ("Bad-debt cap (USD)", "1{,}000"),
        ("Bad-debt-feasible candidates", f"{n_feas/n_cand*100:.2f}\\% ({n_feas}/{n_cand})"),
        ("Best-candidate worst-window bad debt (buyback, USD)", money(worst_bd)),
    ]
    body = (
        "\\begin{table}[h!]\n\\centering\n"
        f"\\caption{{Historical-window backtest summary (constrained sweep, \\texttt{{{csv_name_tex}}}).}}\n"
        "\\label{tab:historical_window_summary}\n"
        "\\begin{tabular}{@{}lr@{}}\n\\toprule\nMetric & Value \\\\\n\\midrule\n"
        + "".join(f"{k} & {v} \\\\\n" for k, v in rows)
        + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    write("historical_window_summary_table.tex", body)


def delta_table() -> None:
    base = json.loads((SWEEP / "scenario_candidate_best.json").read_text(encoding="utf-8"))
    drows = []
    for delta in [0.00, 0.04, 0.08]:
        scn = json.loads(json.dumps(base))
        for s in scn:
            if "dynamic" in s:
                s["dynamic"]["recovery_ltv_gap"] = delta
        sp_path = SWEEP / "paper_runs" / f"delta_{int(delta*100):02d}.json"
        sp_path.parent.mkdir(parents=True, exist_ok=True)
        sp_path.write_text(json.dumps(scn), encoding="utf-8")
        summ = run_backtest(sp_path, f"d{delta}")
        pw = per_window(summ)
        loss, sells, buys, bdd = pw["avg_borrower_final_loss_usd"], pw["total_sell_events"], pw["total_buy_events"], pw["max_bad_debt_usd"]
        active = sells[SCN["buy"]] > 0
        na = int(active.sum())
        bw = ((loss[SCN["no"]] - loss[SCN["buy"]]) > 1e-9) & active
        rate = (bw.sum() / na * 100) if na else 0.0
        drows.append((delta, na, rate, int(bw.sum()), int(buys[SCN["buy"]].sum()), float(bdd[SCN["buy"]].max())))
    body = (
        "\\begin{table}[h!]\n\\centering\n"
        "\\caption{Buyback trigger-band sensitivity in historical-window replay (same policy family, varying $\\delta$).}\n"
        "\\label{tab:delta_sensitivity}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{@{}l p{0.28\\textwidth} p{0.25\\textwidth} r r@{}}\n\\toprule\n"
        "$\\delta$ & Active windows & Buyback vs no-buyback win-rate & Buy events (total) & Worst bad debt (USD) \\\\\n\\midrule\n"
        + "".join(f"{d:.2f} & {na} & {rate:.2f}\\% ({nb}/{na}) & {be} & {money(bd)} \\\\\n"
                 for d, na, rate, nb, be, bd in drows)
        + "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    )
    write("delta_sensitivity_table.tex", body)


def hf_floor_frontier_table() -> None:
    """Restoration-vs-re-leverage frontier as HF_floor varies (governance dial)."""
    f = ROOT / "runs" / "sweeps" / "hf_floor_sweep_results.csv"
    if not f.exists():
        print("skip frontier table (run hf_floor_sweep.py first)")
        return
    df = pd.read_csv(f)
    real = df[df.regime == "real"].set_index("floor")
    crash = df[df.regime == "crash"].set_index("floor")
    order = ["LLTV", "1.05", "1.10", "1.15", "1.20", "1.25", "1.30"]
    body_rows = []
    for fl in order:
        if fl not in real.index:
            continue
        r, c = real.loc[fl], crash.loc[fl]
        label = "LLTV (none)" if fl == "LLTV" else fl
        body_rows.append(
            f"{label} & {money(r.mean_loss_reduction)} & {int(r.buyback_win)}/{int(r.buyback_lose)} & "
            f"{r.restoration_gain:+.4f} & {r.extra_sells:+.2f} & {money(c.mean_loss_reduction)} & "
            f"{c.restoration_gain:+.4f} & {money(max(r.worst_bad_debt, c.worst_bad_debt))} \\\\"
        )
    body = (
        "\\begin{table}[h!]\n\\centering\n"
        "\\caption{Restoration--re-leverage frontier as the buyback health-factor floor varies (governance dial; buyback ratio $\\eta=1$). Lower floors restore more and reduce loss more on average, but hurt more borrowers and induce more re-liquidation; bad debt stays zero throughout. Recommended operating points: $\\mathrm{HF}^{\\text{floor}}=1.20$ (Pareto-safe: essentially never harms a borrower) and $1.10$ (higher net benefit, tolerating a small minority of harmed borrowers).}\n"
        "\\label{tab:hf_floor_frontier}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
        " & \\multicolumn{4}{c}{Real windows ($n=600$)} & \\multicolumn{2}{c}{Crash ($n=120$)} & \\\\\n"
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\n"
        "$\\mathrm{HF}^{\\text{floor}}$ & $\\Delta$loss (USD) & win/lose & $\\Delta\\rho$ & extra sells & $\\Delta$loss (USD) & $\\Delta\\rho$ & Worst bad debt \\\\\n\\midrule\n"
        + "\n".join(body_rows) + "\n"
        + "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    )
    write("hf_floor_frontier_table.tex", body)


def main() -> None:
    stats_out = batch_tables()
    comparison = regime_comparison_table()
    hf_floor_frontier_table()
    hist_summary_table()
    delta_table()
    stats_out["regime_comparison"] = comparison
    (SWEEP / "paper_stats.json").write_text(json.dumps(stats_out, indent=2), encoding="utf-8")
    print("\nPAIRED STATS (buyback vs no-buyback, real-window batch):")
    print(json.dumps(stats_out, indent=2))


if __name__ == "__main__":
    main()
