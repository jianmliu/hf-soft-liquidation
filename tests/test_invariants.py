"""Property tests for the paper's closed-form guarantees (\\S2--\\S3).

Each test asserts an identity the manuscript claims as a *guarantee*, against the
actual engine (not a re-derivation):

  1. target-HF sizing lands the post-trade health factor exactly on HF*.
  2. No action when the position is already at/above target.
  3. Infeasible denominator (HF* <= LT(1+b)) falls back to full repayment.
  4. Reserve identity: every sell credits exactly Delta U = b * Delta D.
  5. Re-leverage floor: post-buy HF >= HF_floor at every executed buy.
  6. Stress cap: post-buy HF >= 1 even after a further d price drop.
  7. Buyback equity identity: with identical sells (floor guard),
     Delta Eq_T = sum_i q_i (P_T - P_i^buy)  and  Delta C_T = sum_i q_i > 0,
     with zero bad debt on the test path.
  8. Confirmed-bounce gate actually gates (a huge min_bounce yields zero buys).

Run directly (no pytest needed):  PYTHONHASHSEED=0 ./.venv/bin/python tests/test_invariants.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import aave_counterfactual_pipeline as M

LT = 0.83
TOL = 1e-8


def make_policy(**over) -> M.DynamicPolicy:
    base = dict(
        lltv=0.85, min_close_factor=0.15, max_close_factor=0.60, cf_slope=1.6,
        liquidation_bonus=0.01, buyback_ratio=1.0, buyback_funding="reborrow",
        recovery_ltv_gap=0.08, sell_cooldown_steps=1, buy_cooldown_steps=1,
        enable_buyback=True, target_hf=1.05, buyback_hf_floor=1.20,
        min_buyback_spread=0.05, buyback_uptrend_lookback=0,
        buyback_min_bounce=0.0, buyback_stress_drawdown=0.25,
    )
    base.update(over)
    return M.DynamicPolicy(**base)


# ---------------------------------------------------------------- unit level --
def test_target_hf_exact():
    pol = make_policy()
    debt = 100_000.0
    price = 2_000.0
    collateral = debt / pol.lltv / price  # exactly at the LLTV trigger
    dd = M.target_hf_debt_repaid(collateral, debt, price, LT, pol)
    assert dd > 0
    sold = (1 + pol.liquidation_bonus) * dd / price
    post_hf = (collateral - sold) * price * LT / (debt - dd)
    assert abs(post_hf - pol.target_hf) < 1e-9, post_hf


def test_no_action_above_target():
    pol = make_policy()
    debt = 100_000.0
    price = 2_000.0
    collateral = pol.target_hf * debt / LT / price * 1.01  # HF just above HF*
    assert M.target_hf_debt_repaid(collateral, debt, price, LT, pol) == 0.0


def test_infeasible_full_repay():
    pol = make_policy(target_hf=LT * 1.01)  # HF* <= LT(1+b): denominator <= 0
    debt, price = 100_000.0, 2_000.0
    collateral = debt / pol.lltv / price
    assert M.target_hf_debt_repaid(collateral, debt, price, LT, pol) == debt


# ----------------------------------------------------------- end-to-end level --
def run_pair(tmp: Path, **policy_over):
    """Run buyback-on vs buyback-off on a deep decline-then-rebound path.

    The decline must be deep: under stress-sized buying (d=0.25) a buy requires
    LTV <= LT(1-d) ~ 0.62 while the price is still below an early band lot, which
    only happens after large sells near the bottom followed by a partial rebound.
    """
    down = np.linspace(3000, 1700, 60)
    up = np.linspace(1700, 2300, 40)
    prices = np.concatenate([down, up])
    n = len(prices)
    (tmp / "normalized").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "block_number": np.arange(n),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h").astype(str),
        "asset_symbol": "WETH", "price_usd": prices,
    }).to_csv(tmp / "normalized" / "prices.csv", index=False)
    pd.DataFrame([{
        "account": "0xtest", "asset_symbol": "WETH", "collateral_amount": 1.0,
        "debt_amount": 100_000.0, "liquidation_threshold": LT, "initial_cr": 1.30,
    }]).to_csv(tmp / "normalized" / "positions_initial.csv", index=False)

    def scn(enable):
        dyn = dict(
            lltv=0.85, target_hf=1.05, min_close_factor=0.15, max_close_factor=0.60,
            cf_slope=1.6, liquidation_bonus=0.01, buyback_ratio=1.0,
            buyback_funding="reborrow", enable_buyback=enable, recovery_ltv_gap=0.08,
            sell_cooldown_steps=1, buy_cooldown_steps=1 if enable else 1_000_000,
            buyback_hf_floor=1.20, min_buyback_spread=0.05,
            buyback_stress_drawdown=0.25,
        )
        dyn.update(policy_over)
        return [{"name": "dyn", "dynamic": dyn}]

    out = {}
    for enable in (True, False):
        sp = tmp / f"scn_{enable}.json"
        sp.write_text(json.dumps(scn(enable)), encoding="utf-8")
        rd = M.run_counterfactual(dataset_dir=tmp, scenario_path=sp,
                                  output_dir=tmp / "runs", run_id=f"r{enable}")
        out[enable] = {
            "events": pd.read_csv(rd / "event_log.csv"),
            "acct": pd.read_csv(rd / "account_outcomes.csv").iloc[0],
        }
    return prices, out


def test_end_to_end_guarantees():
    tmp = ROOT / "runs" / "test_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    prices, out = run_pair(tmp)
    p_T = float(prices[-1])
    ev_b, ev_n = out[True]["events"], out[False]["events"]
    sells_b = ev_b[ev_b.event == "SELL"].reset_index(drop=True)
    sells_n = ev_n[ev_n.event == "SELL"].reset_index(drop=True)
    buys = ev_b[ev_b.event == "BUY"].reset_index(drop=True)

    # (7a) the floor prevents induced re-liquidation: identical sells
    assert len(sells_b) == len(sells_n) > 0
    assert np.allclose(sells_b.debt_repaid, sells_n.debt_repaid)
    assert np.allclose(sells_b.collateral_amount, sells_n.collateral_amount)
    assert len(buys) > 0, "test path must activate the buyback"

    # (4) reserve identity Delta U = b * Delta D on every sell
    assert np.allclose(sells_b.reserve_change_usd, 0.01 * sells_b.debt_repaid)

    # (5)+(6) replay the ledger; check floor and stress floor after every buy
    coll = 100_000.0 * 1.30 / float(prices[0])
    debt = 100_000.0
    ev_by_t = {int(t): g for t, g in ev_b.groupby("block_number")}
    for t, p in enumerate(prices):
        for _, e in ev_by_t.get(t, pd.DataFrame()).iterrows():
            if e.event == "SELL":
                coll -= e.collateral_amount
                debt -= e.debt_repaid
            else:
                coll += e.collateral_amount
                debt += e.collateral_amount * p
                hf_post = coll * p * LT / debt
                assert hf_post >= 1.20 - 1e-9, f"floor breached: {hf_post}"
                p_s = p * (1 - 0.25)
                assert coll * p_s * LT / debt >= 1.0 - 1e-9, "stress floor breached"

    # (7b) equity identity and unconditional collateral gain; zero bad debt
    def equity(acct):
        return acct.final_collateral * acct.final_price_usd - acct.final_debt_usd
    d_eq = equity(out[True]["acct"]) - equity(out[False]["acct"])
    identity = float((buys.collateral_amount * (p_T - buys.price_usd)).sum())
    assert abs(d_eq - identity) < 1e-6 * max(1.0, abs(identity)), (d_eq, identity)
    assert (out[True]["acct"].restoration_ratio
            > out[False]["acct"].restoration_ratio)
    assert out[True]["acct"].max_bad_debt_usd == 0.0
    assert out[False]["acct"].max_bad_debt_usd == 0.0


def test_bounce_gate_blocks():
    tmp = ROOT / "runs" / "test_tmp_gate"
    shutil.rmtree(tmp, ignore_errors=True)
    _, out = run_pair(tmp, buyback_uptrend_lookback=5, buyback_min_bounce=10.0)
    buys = out[True]["events"].query("event == 'BUY'")
    assert len(buys) == 0, "a 1000% bounce requirement must block all buys"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
