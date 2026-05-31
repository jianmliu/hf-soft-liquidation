from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aave_counterfactual_pipeline import historical_backtest

# Liquidation threshold used by the dataset positions. Target-HF sizing only
# produces a positive trim when HF* > LT / LLTV, so candidate (lltv, target_hf)
# pairs are filtered against this feasibility condition.
LT = 0.83


def build_scenario(
    core: dict[str, float],
    target_hf: float,
    buy_ratio: float,
    recovery_ltv_gap: float,
    fixed_close_factor: float,
    fixed_liquidation_bonus: float,
) -> list[dict]:
    """Three matched policies.

    - baseline_dynamic: proposed mechanism = target-HF sizing + endogenous
      lot-grid band + lot-matched, reborrow-funded buyback.
    - target_hf_no_buyback: same target-HF sizing, buyback disabled (Aave-v4
      style single-directional target-HF baseline).
    - traditional_fixed_cf: fixed close-factor liquidation baseline.
    """
    dyn = {
        **core,
        "target_hf": target_hf,
        "recovery_ltv_gap": recovery_ltv_gap,
        "buyback_ratio": buy_ratio,
        "buyback_funding": "reborrow",
        "sell_cooldown_steps": 1,
        # Re-leverage guard: cap reborrow so post-buy HF >= 1.20 and require a
        # 5% positive spread vs the matched sell lot. This removes the ping-pong
        # re-liquidation that an unguarded (borrow-to-LLTV) buyback induces.
        "buyback_hf_floor": 1.20,
        "min_buyback_spread": 0.05,
        # OOS-optimized buy principles: stress-tested sizing (survive a further
        # 25% drop) is the effective lever on multi-leg crashes; a light
        # confirmed-upturn gate adds little. See runs/sweeps/optimize_buy_principles.py.
        "buyback_stress_drawdown": 0.25,
        "buyback_uptrend_lookback": 5,
    }
    return [
        {
            "name": "baseline_dynamic",
            "dynamic": {**dyn, "enable_buyback": True, "buy_cooldown_steps": 1},
        },
        {
            "name": "target_hf_no_buyback",
            "dynamic": {**dyn, "enable_buyback": False, "buy_cooldown_steps": 1000000},
        },
        {
            "name": "traditional_fixed_cf",
            "buyback_bandwidth": 10.0,
            "tiers": [
                {
                    "name": "Fixed CF 50%",
                    "hf_down": 1.01,
                    "close_factor": fixed_close_factor,
                    "liquidation_bonus": fixed_liquidation_bonus,
                    "buyback_ratio": 0.70,
                }
            ],
        },
    ]


def main() -> None:
    dataset_dir = ROOT / "data" / "aave"
    sweep_dir = ROOT / "runs" / "sweeps"
    run_output_root = sweep_dir / "hist_runs"
    candidates_dir = sweep_dir / "hist_candidates"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    run_output_root.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Multi-objective policy:
    # 1) hard bad-debt feasibility (cap + not worse than fixed-CF),
    # 2) minimize borrower loss among feasible candidates.
    bad_debt_cap_usd = 1000.0

    # Target-HF mechanism sweep. The close-factor band (min/max/slope) is inert
    # under target-HF sizing, so we sweep the parameters that actually drive the
    # proposed mechanism: the repair target HF*, the LLTV trigger, the buyback
    # ratio, and the buyback risk-guard bandwidth (delta). Fixed-CF baseline
    # parameters are swept for a fair cross-policy comparison.
    lltv_list = [0.80, 0.82, 0.85]
    target_hf_list = [1.05, 1.10, 1.20]
    buy_ratio_list = [0.20, 0.50, 1.00]
    recovery_gap_list = [0.00, 0.04, 0.08]
    fixed_cf_list = [0.30, 0.50]
    fixed_bonus_list = [0.05, 0.06]

    combos = []
    for lltv, target_hf, buy_ratio, rec_gap, fixed_cf, fixed_bonus in product(
        lltv_list,
        target_hf_list,
        buy_ratio_list,
        recovery_gap_list,
        fixed_cf_list,
        fixed_bonus_list,
    ):
        # Feasibility: target-HF must exceed the health factor at the trigger
        # boundary (HF* > LT / LLTV), otherwise no liquidation ever fires.
        if target_hf <= LT / lltv:
            continue
        combos.append(
            {
                "lltv": lltv,
                "target_hf": target_hf,
                "liquidation_bonus": 0.01,
                "buyback_ratio": buy_ratio,
                "recovery_ltv_gap": rec_gap,
                "fixed_close_factor": fixed_cf,
                "fixed_liquidation_bonus": fixed_bonus,
            }
        )

    # Keep runtime manageable while still covering multiple dimensions.
    max_candidates = 24
    if len(combos) > max_candidates:
        step = (len(combos) - 1) / float(max_candidates - 1)
        selected_indices = sorted({int(round(i * step)) for i in range(max_candidates)})
        combos = [combos[i] for i in selected_indices]

    rows: list[dict] = []
    for idx, cfg in enumerate(combos, start=1):
        # min/max close factor + slope are retained only for engine
        # back-compatibility; target-HF sizing ignores them.
        core = {
            "lltv": cfg["lltv"],
            "min_close_factor": 0.15,
            "max_close_factor": 0.60,
            "cf_slope": 1.60,
            "liquidation_bonus": cfg["liquidation_bonus"],
        }
        scenario = build_scenario(
            core=core,
            target_hf=cfg["target_hf"],
            buy_ratio=cfg["buyback_ratio"],
            recovery_ltv_gap=cfg["recovery_ltv_gap"],
            fixed_close_factor=cfg["fixed_close_factor"],
            fixed_liquidation_bonus=cfg["fixed_liquidation_bonus"],
        )

        scenario_path = candidates_dir / f"hist_sweep_{stamp}_{idx:03d}.json"
        scenario_path.write_text(json.dumps(scenario, indent=2), encoding="utf-8")

        summary_path, aggregate_path, report_path = historical_backtest(
            dataset_dir=dataset_dir,
            scenario_path=scenario_path,
            output_dir=run_output_root,
            window_size=120,
            window_step=24,
            max_windows=24,
            loan_mid_starts=True,
            loan_min_duration_blocks=36,
        )

        agg = pd.read_csv(aggregate_path)
        by_s = {row["scenario"]: row for _, row in agg.iterrows()}

        fixed = by_s["traditional_fixed_cf"]
        no_buy = by_s["target_hf_no_buyback"]
        buy = by_s["baseline_dynamic"]

        delta_no_minus_fixed = float(no_buy["mean_avg_borrower_final_loss_usd"] - fixed["mean_avg_borrower_final_loss_usd"])
        delta_buy_minus_no = float(buy["mean_avg_borrower_final_loss_usd"] - no_buy["mean_avg_borrower_final_loss_usd"])

        fixed_bad_debt = float(fixed["worst_max_bad_debt_usd"])
        no_bad_debt = float(no_buy["worst_max_bad_debt_usd"])
        buy_bad_debt = float(buy["worst_max_bad_debt_usd"])

        bad_debt_feasible = (buy_bad_debt <= bad_debt_cap_usd) and (buy_bad_debt <= fixed_bad_debt)

        penalty = 0.0
        # Prioritize bad-debt constraints, then relative performance constraints.
        if not bad_debt_feasible:
            penalty += 1_000_000.0
        if delta_no_minus_fixed > 0:
            penalty += 100_000.0
        if delta_buy_minus_no > 0:
            penalty += 1_000_000.0

        objective = float(buy["mean_avg_borrower_final_loss_usd"]) + penalty

        row = {
            "idx": idx,
            **cfg,
            "summary_path": str(summary_path),
            "aggregate_path": str(aggregate_path),
            "report_path": str(report_path),
            "windows": int(buy["windows"]),
            "mean_loss_fixed": float(fixed["mean_avg_borrower_final_loss_usd"]),
            "mean_loss_no_buyback": float(no_buy["mean_avg_borrower_final_loss_usd"]),
            "mean_loss_buyback": float(buy["mean_avg_borrower_final_loss_usd"]),
            "p90_loss_buyback": float(buy["p90_avg_borrower_final_loss_usd"]),
            "worst_bad_debt_fixed": fixed_bad_debt,
            "worst_bad_debt_no_buyback": no_bad_debt,
            "worst_bad_debt_buyback": buy_bad_debt,
            "bad_debt_cap_usd": bad_debt_cap_usd,
            "bad_debt_feasible": bool(bad_debt_feasible),
            "buy_events_total": int(buy["total_buy_events"]),
            "delta_no_minus_fixed": delta_no_minus_fixed,
            "delta_buy_minus_no": delta_buy_minus_no,
            "objective": objective,
        }
        rows.append(row)
        print(
            f"[{idx}/{len(combos)}] lltv={cfg['lltv']:.2f}, target_hf={cfg['target_hf']:.2f}, "
            f"buy_ratio={cfg['buyback_ratio']:.2f}, delta(rec_gap)={cfg['recovery_ltv_gap']:.2f}, "
            f"fixed_cf={cfg['fixed_close_factor']:.2f}, fixed_bonus={cfg['fixed_liquidation_bonus']:.2f}, "
            f"delta_no-fixed={delta_no_minus_fixed:.2f}, delta_buy-no={delta_buy_minus_no:.2f}, "
            f"buy_bd={buy_bad_debt:.2f}, feasible={bad_debt_feasible}"
        )

    out = pd.DataFrame(rows).sort_values(["objective", "mean_loss_buyback", "p90_loss_buyback"])
    out_path = sweep_dir / f"historical_window_sweep_results_{stamp}.csv"
    out.to_csv(out_path, index=False)

    # Persist the best feasible candidate scenario for downstream batch / delta runs.
    if not out.empty:
        best_idx = int(out.iloc[0]["idx"])
        best_scenario_src = candidates_dir / f"hist_sweep_{stamp}_{best_idx:03d}.json"
        best_dst = sweep_dir / "scenario_candidate_best.json"
        best_dst.write_text(best_scenario_src.read_text(encoding="utf-8"), encoding="utf-8")
        print("best_candidate_scenario", best_dst)

    print("results_csv", out_path)
    print("num_target_beats_fixed", int((out["delta_no_minus_fixed"] < 0).sum()))
    print("num_buyback_beats_no", int((out["delta_buy_minus_no"] < 0).sum()))
    print("num_bad_debt_feasible", int(out["bad_debt_feasible"].sum()))
    both = (out["delta_no_minus_fixed"] < 0) & (out["delta_buy_minus_no"] < 0)
    print("num_both", int(both.sum()))
    print("top5")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
