from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_HF_DOWNS = [1.05, 1.00, 0.95]
DEFAULT_CLOSE_FACTORS = [0.20, 0.30, 0.50]
DEFAULT_LIQUIDATION_BONUSES = [0.05, 0.08, 0.12]
DEFAULT_BUYBACK_RATIOS = [0.50, 0.70, 1.00]


def infer_csv_path(project_root: Path) -> Path | None:
    candidates = [
        p
        for p in project_root.rglob("*.csv")
        if ".venv" not in p.parts and "doc" not in p.parts
    ]
    return candidates[0] if candidates else None


def generate_synthetic_prices(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.arange(length)
    baseline = 95_000 + 12_000 * np.sin(x / 45.0) + 7_000 * np.sin(x / 11.0)
    center = length // 2
    shock = -18_000 * np.maximum(0, 1 - np.abs(x - center) / max(1, length * 0.22))
    noise = rng.normal(0, 900, size=length)
    prices = baseline + shock + noise
    return np.clip(prices, 10_000, None)


def load_prices(csv_path: Path | None, synthetic_length: int, seed: int) -> tuple[np.ndarray, str]:
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path, sep=None, engine="python")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")
        price_col = "close" if "close" in df.columns else None
        if price_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError(f"No numeric price column found in {csv_path}")
            price_col = numeric_cols[0]
        prices = df[price_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        prices = prices[prices > 0]
        if len(prices) < 220:
            raise ValueError(f"Price series from {csv_path} is too short: {len(prices)}")
        return prices, f"historical:{csv_path.name}"
    return generate_synthetic_prices(synthetic_length, seed), "synthetic"


def parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def build_hf_trigger_tiers(
    hf_downs: list[float],
    close_factors: list[float],
    bonuses: list[float],
    buyback_ratios: list[float],
    buyback_trigger_bandwidth: float,
) -> list[dict]:
    size = len(hf_downs)
    if not (len(close_factors) == len(bonuses) == len(buyback_ratios) == size):
        raise ValueError("hf_downs / close_factors / bonuses / buyback_ratios must have the same length")

    trigger_tiers = []
    for idx in range(size):
        down = hf_downs[idx]
        if down <= 0:
            raise ValueError("HF thresholds must be positive")
        close_factor = close_factors[idx]
        bonus = bonuses[idx]
        buyback_ratio = buyback_ratios[idx]
        if close_factor <= 0 or close_factor > 1:
            raise ValueError("close factor must be in (0, 1]")
        if bonus < 0:
            raise ValueError("liquidation bonus must be >= 0")
        if buyback_ratio <= 0 or buyback_ratio > 1:
            raise ValueError("buyback ratio must be in (0, 1]")

        trigger_tiers.append(
            {
                "trigger_tier": f"Tier {idx + 1}",
                "hf_down": down,
                "hf_up": down + buyback_trigger_bandwidth,
                "close_factor": close_factor,
                "liquidation_bonus": bonus,
                "buyback_ratio": buyback_ratio,
            }
        )

    return sorted(trigger_tiers, key=lambda item: item["hf_down"], reverse=True)


def risk_metrics(collateral: float, debt: float, price: float, liquidation_threshold: float) -> tuple[float, float, float]:
    collateral_value = collateral * price
    if debt <= 0:
        return float("inf"), 0.0, float("inf")
    if collateral_value <= 0:
        return 0.0, float("inf"), 0.0
    hf = collateral_value * liquidation_threshold / debt
    ltv = debt / collateral_value
    cr = collateral_value / debt
    return hf, ltv, cr


def simulate(
    prices: np.ndarray,
    num_loans: int,
    loan_value: float,
    initial_cr: float,
    liquidation_threshold: float,
    trigger_tiers: list[dict],
    seed: int,
    min_tail: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if len(prices) <= min_tail + num_loans + 1:
        raise ValueError("Price series too short for chosen num_loans/min_tail")

    rng = np.random.default_rng(seed)
    start_pool = np.arange(0, len(prices) - min_tail)
    loan_indices = sorted(rng.choice(start_pool, size=num_loans, replace=False).tolist())

    path_rows: list[dict] = []
    event_rows: list[dict] = []
    summary_rows: list[dict] = []
    first_loan_levels: list[dict] = []

    for loan_id, start_idx in enumerate(loan_indices):
        sub_prices = prices[start_idx:]
        initial_price = float(sub_prices[0])

        initial_collateral = loan_value * initial_cr / initial_price
        collateral = initial_collateral
        debt = loan_value
        reserve_usd = 0.0

        min_cr_seen = float("inf")
        min_hf_seen = float("inf")
        max_ltv_seen = 0.0
        max_bad_debt = 0.0
        total_sell_usd = 0.0
        total_buy_usd = 0.0
        total_debt_repaid = 0.0
        sell_count = 0
        buy_count = 0

        tier_state: dict[str, dict] = {
            tier["trigger_tier"]: {
                "sold_qty": 0.0,
                "bought_qty": 0.0,
                "outstanding_qty": 0.0,
            }
            for tier in trigger_tiers
        }

        if loan_id == 0:
            first_loan_levels = []
            for tier in trigger_tiers:
                approx_price_down = tier["hf_down"] * loan_value / (initial_collateral * liquidation_threshold)
                approx_price_up = tier["hf_up"] * loan_value / (initial_collateral * liquidation_threshold)
                first_loan_levels.append(
                    {
                        "trigger_tier": tier["trigger_tier"],
                        "hf_down": tier["hf_down"],
                        "hf_up": tier["hf_up"],
                        "price_down": approx_price_down,
                        "price_up": approx_price_up,
                    }
                )

        prev_hf, prev_ltv, prev_cr = risk_metrics(collateral, debt, float(sub_prices[0]), liquidation_threshold)

        for step, raw_price in enumerate(sub_prices):
            price = float(raw_price)
            time_idx = start_idx + step

            hf_now, ltv_now, cr_now = risk_metrics(collateral, debt, price, liquidation_threshold)

            for tier in trigger_tiers:
                hf_down = tier["hf_down"]
                if prev_hf > hf_down and hf_now <= hf_down and debt > 0 and collateral > 0:
                    close_factor = tier["close_factor"]
                    bonus = tier["liquidation_bonus"]
                    debt_repaid = min(close_factor * debt, debt)
                    collateral_sold = (1.0 + bonus) * debt_repaid / price

                    if collateral_sold > collateral:
                        collateral_sold = collateral
                        debt_repaid = min(debt, collateral_sold * price / (1.0 + bonus))

                    sell_value = collateral_sold * price
                    reserve_gain = max(0.0, sell_value - debt_repaid)

                    collateral -= collateral_sold
                    debt -= debt_repaid
                    reserve_usd += reserve_gain

                    state = tier_state[tier["trigger_tier"]]
                    state["sold_qty"] += collateral_sold
                    state["outstanding_qty"] += collateral_sold

                    total_sell_usd += sell_value
                    total_debt_repaid += debt_repaid
                    sell_count += 1

                    event_rows.append(
                        {
                            "loan_id": loan_id,
                            "time": time_idx,
                            "event": "SELL",
                            "trigger_tier": tier["trigger_tier"],
                            "collateral_amount": collateral_sold,
                            "price": price,
                            "usd_value": sell_value,
                            "debt_repaid": debt_repaid,
                            "reserve_change_usd": reserve_gain,
                            "hf": hf_now,
                            "ltv": ltv_now,
                            "close_factor": close_factor,
                            "liquidation_bonus": bonus,
                        }
                    )

                    hf_now, ltv_now, cr_now = risk_metrics(collateral, debt, price, liquidation_threshold)

            for tier in sorted(trigger_tiers, key=lambda item: item["hf_up"]):
                state = tier_state[tier["trigger_tier"]]
                hf_up = tier["hf_up"]
                if prev_hf < hf_up and hf_now >= hf_up and state["outstanding_qty"] > 0 and reserve_usd > 0:
                    target_total_buy = tier["buyback_ratio"] * state["sold_qty"]
                    target_incremental = max(0.0, target_total_buy - state["bought_qty"])
                    if target_incremental <= 0:
                        continue

                    affordable = min(target_incremental, state["outstanding_qty"], reserve_usd / price)
                    if affordable <= 0:
                        continue

                    buy_cost = affordable * price
                    collateral += affordable
                    reserve_usd -= buy_cost
                    state["bought_qty"] += affordable
                    state["outstanding_qty"] -= affordable

                    total_buy_usd += buy_cost
                    buy_count += 1

                    event_rows.append(
                        {
                            "loan_id": loan_id,
                            "time": time_idx,
                            "event": "BUY",
                            "trigger_tier": tier["trigger_tier"],
                            "collateral_amount": affordable,
                            "price": price,
                            "usd_value": buy_cost,
                            "debt_repaid": 0.0,
                            "reserve_change_usd": -buy_cost,
                            "hf": hf_now,
                            "ltv": ltv_now,
                            "close_factor": tier["close_factor"],
                            "liquidation_bonus": tier["liquidation_bonus"],
                        }
                    )

                    hf_now, ltv_now, cr_now = risk_metrics(collateral, debt, price, liquidation_threshold)

            collateral_value = collateral * price
            bad_debt = max(0.0, debt - collateral_value)
            min_cr_seen = min(min_cr_seen, cr_now)
            min_hf_seen = min(min_hf_seen, hf_now)
            max_ltv_seen = max(max_ltv_seen, ltv_now if np.isfinite(ltv_now) else 0.0)
            max_bad_debt = max(max_bad_debt, bad_debt)

            path_rows.append(
                {
                    "loan_id": loan_id,
                    "time": time_idx,
                    "price": price,
                    "cr": cr_now,
                    "hf": hf_now,
                    "ltv": ltv_now,
                    "collateral_amount": collateral,
                    "debt": debt,
                    "bad_debt": bad_debt,
                    "reserve_usd": reserve_usd,
                    "profit": reserve_usd,
                    "start_index": start_idx,
                }
            )

            prev_hf, prev_ltv, prev_cr = hf_now, ltv_now, cr_now

        final_price = float(sub_prices[-1])
        final_collateral_value = collateral * final_price
        restoration_ratio = collateral / initial_collateral if initial_collateral > 0 else np.nan
        impermanent_loss_pct = max(0.0, (1.0 - restoration_ratio) * 100.0)

        summary_rows.append(
            {
                "loan_id": loan_id,
                "start_index": start_idx,
                "initial_price": initial_price,
                "final_price": final_price,
                "initial_collateral": initial_collateral,
                "final_collateral": collateral,
                "restoration_ratio": restoration_ratio,
                "impermanent_loss_pct": impermanent_loss_pct,
                "total_sell_usd": total_sell_usd,
                "total_buy_usd": total_buy_usd,
                "total_debt_repaid_usd": total_debt_repaid,
                "protocol_profit_usd": reserve_usd,
                "final_debt_usd": debt,
                "min_hf": min_hf_seen,
                "max_ltv": max_ltv_seen,
                "min_cr": min_cr_seen,
                "max_bad_debt_usd": max_bad_debt,
                "final_collateral_value_usd": final_collateral_value,
                "sell_events": sell_count,
                "buy_events": buy_count,
            }
        )

    path_df = pd.DataFrame(path_rows)
    events_df = pd.DataFrame(event_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("loan_id").reset_index(drop=True)
    metadata = {
        "loan_indices": loan_indices,
        "first_loan_levels": first_loan_levels,
        "trigger_tier_hf_downs": [tier["hf_down"] for tier in trigger_tiers],
        "trigger_tier_hf_ups": [tier["hf_up"] for tier in trigger_tiers],
    }
    return path_df, events_df, summary_df, metadata


def save_plots(
    prices: np.ndarray,
    path_df: pd.DataFrame,
    events_df: pd.DataFrame,
    metadata: dict,
    fig_dir: Path,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 5))
    plt.plot(prices, color="black", linewidth=1.2, label="Collateral Price")
    for level in metadata.get("first_loan_levels", []):
        plt.axhline(level["price_down"], color="steelblue", linestyle="--", alpha=0.35)
        plt.axhline(level["price_up"], color="teal", linestyle=":", alpha=0.35)
    plt.title("Synthetic/Historical Price Path with HF Trigger Tiers")
    plt.xlabel("Time Step")
    plt.ylabel("Collateral Price (USD)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_dir / "synthetic_price_grid.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 6))
    for loan_id, loan_df in path_df.groupby("loan_id"):
        plt.plot(loan_df["time"], loan_df["cr"], linewidth=1.1, label=f"Loan {loan_id}")
    for threshold in metadata.get("trigger_tier_hf_downs", []):
        plt.axhline(y=threshold, linestyle="--", color="gray", alpha=0.2)
    if not events_df.empty:
        sells = events_df[events_df["event"] == "SELL"]
        buys = events_df[events_df["event"] == "BUY"]
        if not sells.empty:
            cr_sells = path_df.merge(sells[["loan_id", "time"]], on=["loan_id", "time"], how="inner")
            plt.scatter(cr_sells["time"], cr_sells["cr"], color="red", marker="v", s=20, label="Sell")
        if not buys.empty:
            cr_buys = path_df.merge(buys[["loan_id", "time"]], on=["loan_id", "time"], how="inner")
            plt.scatter(cr_buys["time"], cr_buys["cr"], color="green", marker="^", s=20, label="Buy")
    plt.title("Collateral Ratio Paths with Trigger Events")
    plt.xlabel("Time Step")
    plt.ylabel("Collateral Ratio")
    plt.grid(alpha=0.2)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "cr_plot_tiers.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(prices, color="black", linewidth=1.2, label="Collateral Price")
    for idx, start_idx in enumerate(metadata.get("loan_indices", [])):
        plt.axvline(start_idx, color="gray", linestyle=":", alpha=0.45)
        plt.text(start_idx, prices[start_idx], f"L{idx}", fontsize=7, rotation=90, va="bottom")
    if not events_df.empty:
        sell_points = events_df[events_df["event"] == "SELL"]
        buy_points = events_df[events_df["event"] == "BUY"]
        if not sell_points.empty:
            plt.scatter(
                sell_points["time"],
                prices[sell_points["time"].astype(int)],
                color="red",
                marker="v",
                s=22,
                label="Sell",
            )
        if not buy_points.empty:
            plt.scatter(
                buy_points["time"],
                prices[buy_points["time"].astype(int)],
                color="green",
                marker="^",
                s=22,
                label="Buy",
            )
    plt.title("Price Path with Loan Starts and Partial Liquidation Operations")
    plt.xlabel("Time Step")
    plt.ylabel("Collateral Price (USD)")
    plt.grid(alpha=0.2)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "price_grid_loans_tiers.png", dpi=200)
    plt.close()


def latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    return df.to_latex(
        index=False,
        float_format="%.4f",
        caption=caption,
        label=label,
        escape=True,
        na_rep="-",
        column_format="l" + "r" * (len(df.columns) - 1),
    )


def save_tables(
    summary_df: pd.DataFrame,
    events_df: pd.DataFrame,
    table_dir: Path,
    source_tag: str,
    config: dict,
) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)

    agg = pd.DataFrame(
        [
            {
                "num_loans": int(len(summary_df)),
                "avg_impermanent_loss_pct": float(summary_df["impermanent_loss_pct"].mean()),
                "avg_protocol_profit_usd": float(summary_df["protocol_profit_usd"].mean()),
                "avg_min_cr": float(summary_df["min_cr"].mean()),
                "avg_min_hf": float(summary_df["min_hf"].mean()),
                "avg_max_ltv": float(summary_df["max_ltv"].mean()),
                "max_bad_debt_usd": float(summary_df["max_bad_debt_usd"].max()),
                "source": source_tag,
            }
        ]
    )

    summary_df.to_csv(table_dir / "loan_summary.csv", index=False)
    events_df.to_csv(table_dir / "event_log.csv", index=False)
    agg.to_csv(table_dir / "aggregate_metrics.csv", index=False)

    config_df = pd.DataFrame(
        [
            {"Parameter": "Price source", "Value": config["source_tag"]},
            {"Parameter": "Number of loans", "Value": config["num_loans"]},
            {"Parameter": "Loan value (USD)", "Value": config["loan_value"]},
            {"Parameter": "Initial CR", "Value": config["initial_cr"]},
            {"Parameter": "Liquidation threshold", "Value": config["liquidation_threshold"]},
            {"Parameter": "HF down triggers", "Value": config["hf_downs"]},
            {"Parameter": "Close factors", "Value": config["close_factors"]},
            {"Parameter": "Liquidation bonuses", "Value": config["liquidation_bonuses"]},
            {"Parameter": "Buyback ratios", "Value": config["buyback_ratios"]},
            {"Parameter": "Buyback bandwidth", "Value": config["buyback_bandwidth"]},
            {"Parameter": "Random seed", "Value": config["seed"]},
            {"Parameter": "Synthetic length", "Value": config["synthetic_length"]},
            {"Parameter": "Min tail", "Value": config["min_tail"]},
        ]
    )

    impermanent_table = summary_df[
        ["loan_id", "initial_collateral", "final_collateral", "impermanent_loss_pct", "restoration_ratio"]
    ].copy()
    impermanent_table.columns = ["Loan ID", "Initial Collateral", "Final Collateral", "Imp. Loss (\\%)", "Restoration Ratio"]

    pnl_table = summary_df[
        ["loan_id", "total_sell_usd", "total_buy_usd", "total_debt_repaid_usd", "protocol_profit_usd", "final_debt_usd"]
    ].copy()
    pnl_table.columns = [
        "Loan ID",
        "Total Sell (USD)",
        "Total Buy (USD)",
        "Debt Repaid (USD)",
        "Reserve PnL (USD)",
        "Final Debt (USD)",
    ]

    aggregate_table = agg[
        [
            "num_loans",
            "avg_impermanent_loss_pct",
            "avg_protocol_profit_usd",
            "avg_min_cr",
            "avg_min_hf",
            "avg_max_ltv",
            "max_bad_debt_usd",
            "source",
        ]
    ].copy()
    aggregate_table.columns = [
        "Loans",
        "Avg IL (\\%)",
        "Avg Reserve PnL",
        "Avg Min CR",
        "Avg Min HF",
        "Avg Max LTV",
        "Max Bad Debt",
        "Source",
    ]

    (table_dir / "impermanent_loss_table.tex").write_text(
        latex_table(
            impermanent_table,
            "Borrower collateral restoration and impermanent loss by loan.",
            "tab:impermanent_loss",
        ),
        encoding="utf-8",
    )
    (table_dir / "protocol_profit_table.tex").write_text(
        latex_table(
            pnl_table,
            "Per-loan partial-liquidation turnover, debt repayment, and reserve PnL.",
            "tab:protocol_profit",
        ),
        encoding="utf-8",
    )
    (table_dir / "aggregate_metrics_table.tex").write_text(
        latex_table(aggregate_table, "Aggregate metrics for the current experiment run.", "tab:aggregate_metrics"),
        encoding="utf-8",
    )
    (table_dir / "simulation_config.csv").write_text(config_df.to_csv(index=False), encoding="utf-8")
    (table_dir / "simulation_config_table.tex").write_text(
        latex_table(config_df, "Simulation configuration used to generate current artifacts.", "tab:simulation_config"),
        encoding="utf-8",
    )


def build_paper(project_root: Path, tex_rel_path: str) -> None:
    tex_path = project_root / tex_rel_path
    doc_dir = tex_path.parent
    tex_name = tex_path.name
    stem = tex_path.stem
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_name],
        cwd=doc_dir,
        check=True,
    )
    subprocess.run(["bibtex", stem], cwd=doc_dir, check=True)
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_name],
            cwd=doc_dir,
            check=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click HF/LTV-driven partial liquidation experiment pipeline")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--num-loans", type=int, default=5)
    parser.add_argument("--loan-value", type=float, default=100000.0)
    parser.add_argument("--initial-cr", type=float, default=1.3)
    parser.add_argument("--liquidation-threshold", type=float, default=0.85)
    parser.add_argument("--hf-downs", type=str, default=",".join(str(x) for x in DEFAULT_HF_DOWNS))
    parser.add_argument("--close-factors", type=str, default=",".join(str(x) for x in DEFAULT_CLOSE_FACTORS))
    parser.add_argument(
        "--liquidation-bonuses",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LIQUIDATION_BONUSES),
    )
    parser.add_argument("--buyback-ratios", type=str, default=",".join(str(x) for x in DEFAULT_BUYBACK_RATIOS))
    parser.add_argument("--buyback-bandwidth", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-length", type=int, default=720)
    parser.add_argument("--min-tail", type=int, default=120)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-tables", action="store_true")
    parser.add_argument("--build-paper", action="store_true")
    parser.add_argument("--paper-tex", type=str, default="doc/main.tex")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    csv_path = args.csv_path.resolve() if args.csv_path else infer_csv_path(project_root)

    hf_downs = parse_float_list(args.hf_downs)
    close_factors = parse_float_list(args.close_factors)
    bonuses = parse_float_list(args.liquidation_bonuses)
    buyback_ratios = parse_float_list(args.buyback_ratios)
    trigger_tiers = build_hf_trigger_tiers(
        hf_downs,
        close_factors,
        bonuses,
        buyback_ratios,
        args.buyback_bandwidth,
    )

    prices, source_tag = load_prices(csv_path, synthetic_length=args.synthetic_length, seed=args.seed)
    path_df, events_df, summary_df, metadata = simulate(
        prices=prices,
        num_loans=args.num_loans,
        loan_value=args.loan_value,
        initial_cr=args.initial_cr,
        liquidation_threshold=args.liquidation_threshold,
        trigger_tiers=trigger_tiers,
        seed=args.seed,
        min_tail=args.min_tail,
    )

    fig_dir = project_root / "doc" / "figures"
    table_dir = project_root / "doc" / "tables"

    if not args.skip_plots:
        save_plots(prices, path_df, events_df, metadata, fig_dir)

    if not args.skip_tables:
        config_payload = {
            "source_tag": source_tag,
            "num_loans": args.num_loans,
            "loan_value": args.loan_value,
            "initial_cr": args.initial_cr,
            "liquidation_threshold": args.liquidation_threshold,
            "hf_downs": ",".join(f"{x:.4f}" for x in hf_downs),
            "close_factors": ",".join(f"{x:.4f}" for x in close_factors),
            "liquidation_bonuses": ",".join(f"{x:.4f}" for x in bonuses),
            "buyback_ratios": ",".join(f"{x:.4f}" for x in buyback_ratios),
            "buyback_bandwidth": args.buyback_bandwidth,
            "seed": args.seed,
            "synthetic_length": args.synthetic_length,
            "min_tail": args.min_tail,
        }
        save_tables(summary_df, events_df, table_dir, source_tag, config_payload)

    print("Simulation completed")
    print(f"Price source: {source_tag}")
    print(f"Loans simulated: {len(summary_df)}")
    print(f"HF trigger tiers: {[tier['hf_down'] for tier in trigger_tiers]}")
    print(f"Buyback bandwidth: {args.buyback_bandwidth}")
    print(f"Figures directory: {fig_dir}")
    print(f"Tables directory: {table_dir}")

    if args.build_paper:
        build_paper(project_root, args.paper_tex)
        print(f"Paper build completed: {(project_root / 'doc' / 'main.pdf')}" )


if __name__ == "__main__":
    main()
