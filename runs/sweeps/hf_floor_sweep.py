"""HF-floor governance dial: restoration vs re-leverage trade-off.

Sweeps the buyback health-factor floor on the real-window stress batch (the
primary protocol) and checks safety on the synthetic deep-crash batch. For each
floor it reports, paired vs the same no-buyback baseline: collateral restoration
gain, borrower-loss reduction (+ Wilcoxon p), how often buyback *hurts*, induced
extra sells (ping-pong indicator), and worst bad debt. `None` = the naive
borrow-to-LLTV rule (no floor).
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

LT, DEBT, W = 0.83, 100_000.0, 180
CRS = [1.25, 1.35, 1.50]
N_WINDOWS = 200
FLOORS = [None, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]


def build_windows(series: np.ndarray, seed: int, tag: str, synth: bool) -> list[Path]:
    tmp = ROOT / "runs" / "hf_floor_sweep" / tag
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dirs = []
    if synth:
        idx = 0
        for run_i in range(120):
            x = np.arange(W); p0 = 3000.0
            depth = rng.uniform(0.18, 0.45); trough = int(rng.uniform(0.30, 0.70) * W)
            width = max(1.0, W * rng.uniform(0.18, 0.30))
            dip = -depth * np.maximum(0.0, 1.0 - np.abs(x - trough) / width)
            reb_frac = rng.uniform(0.3, 1.0)
            reb = np.where(x > trough, reb_frac * depth * (x - trough) / max(1.0, W - trough), 0.0)
            noise = rng.normal(0.0, 0.012, size=W)
            prices = np.clip(p0 * (1 + 0.04 * np.sin(x / 40.0)) * (1.0 + dip + reb + noise), 100.0, None)
            dirs.append(_write(tmp, idx, prices, 1.30)); idx += 1
    else:
        starts = sorted(int(s) for s in rng.choice(np.arange(len(series) - W), size=N_WINDOWS, replace=False))
        idx = 0
        for st in starts:
            for cr in CRS:
                dirs.append(_write(tmp, idx, series[st:st + W], cr)); idx += 1
    return dirs


def _write(tmp: Path, idx: int, prices: np.ndarray, cr: float) -> Path:
    ds = tmp / f"w{idx:04d}"
    (ds / "normalized").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"block_number": np.arange(len(prices)),
                  "timestamp": pd.date_range("2024-01-01", periods=len(prices), freq="h").astype(str),
                  "asset_symbol": "WETH", "price_usd": prices}).to_csv(ds / "normalized" / "prices.csv", index=False)
    pd.DataFrame([{"account": "x", "asset_symbol": "WETH", "collateral_amount": 1.0,
                   "debt_amount": DEBT, "liquidation_threshold": LT, "initial_cr": cr}]).to_csv(
        ds / "normalized" / "positions_initial.csv", index=False)
    return ds


def dyn(enable: bool, floor):
    d = {"lltv": 0.85, "target_hf": 1.05, "min_close_factor": 0.15, "max_close_factor": 0.60,
         "cf_slope": 1.6, "liquidation_bonus": 0.01, "buyback_ratio": 1.0, "buyback_funding": "reborrow",
         "enable_buyback": enable, "recovery_ltv_gap": 0.08, "sell_cooldown_steps": 1,
         "buy_cooldown_steps": 1 if enable else 1000000, "min_buyback_spread": 0.05}
    if floor is not None:
        d["buyback_hf_floor"] = floor
    return {"name": "baseline_dynamic" if enable else "target_hf_no_buyback", "dynamic": d}


def run_batch(dirs: list[Path], scn: list[dict], tag: str) -> pd.DataFrame:
    out = ROOT / "runs" / "hf_floor_sweep" / "_runs"
    sp = ROOT / "runs" / "hf_floor_sweep" / f"scn_{tag}.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(scn), encoding="utf-8")
    rows = []
    for i, ds in enumerate(dirs):
        rd = run_counterfactual(dataset_dir=ds, scenario_path=sp, output_dir=out / tag, run_id=f"w{i}")
        rows.append(pd.read_csv(rd / "scenario_metrics.csv"))
    return pd.concat(rows, ignore_index=True)


def evaluate(dirs: list[Path], label: str) -> list[dict]:
    no = run_batch(dirs, [dyn(False, None)], f"{label}_no").reset_index(drop=True)
    no_loss = no["avg_borrower_final_loss_usd"]; no_rho = no["avg_restoration_ratio"]; no_sells = no["total_sell_events"].mean()
    rows = []
    for floor in FLOORS:
        on = run_batch(dirs, [dyn(True, floor)], f"{label}_f{floor}").reset_index(drop=True)
        d = no_loss - on["avg_borrower_final_loss_usd"]; nz = d[d.abs() > 1e-9]
        p = stats.wilcoxon(nz, alternative="greater").pvalue if len(nz) > 5 else float("nan")
        rows.append({
            "regime": label, "floor": "LLTV" if floor is None else f"{floor:.2f}",
            "mean_loss_reduction": float(d.mean()),
            "buyback_win": int((d > 1e-9).sum()), "buyback_lose": int((d < -1e-9).sum()),
            "wilcoxon_p": float(p),
            "restoration_gain": float((on["avg_restoration_ratio"] - no_rho).mean()),
            "extra_sells": float(on["total_sell_events"].mean() - no_sells),
            "worst_bad_debt": float(on["max_bad_debt_usd"].max()),
            "mean_buy_events": float(on["total_buy_events"].mean()),
        })
        r = rows[-1]
        print(f"[{label}] floor={r['floor']:>4}: Δloss={r['mean_loss_reduction']:+8.1f} "
              f"win/lose={r['buyback_win']}/{r['buyback_lose']} p={r['wilcoxon_p']:.1e} "
              f"Δρ={r['restoration_gain']:+.4f} extra_sells={r['extra_sells']:+.2f} "
              f"bad_debt={r['worst_bad_debt']:.1f}")
    return rows


def main() -> None:
    series = pd.read_csv(ROOT / "data" / "aave" / "normalized" / "prices.csv").sort_values("block_number")["price_usd"].astype(float).to_numpy()
    real_dirs = build_windows(series, 20260529, "real", synth=False)
    crash_dirs = build_windows(series, 40001, "crash", synth=True)
    rows = evaluate(real_dirs, "real") + evaluate(crash_dirs, "crash")
    out = pd.DataFrame(rows)
    out_path = ROOT / "runs" / "sweeps" / "hf_floor_sweep_results.csv"
    out.to_csv(out_path, index=False)
    print("\nresults_csv", out_path)


if __name__ == "__main__":
    main()
