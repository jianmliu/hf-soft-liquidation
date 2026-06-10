# Band-Based Soft Liquidation with Collateral Restoration

Simulation engine, reproduction scripts, and manuscript for a two-sided
soft-liquidation mechanism on Aave-style lending markets:

- **Sell side (baseline, Aave v4's design):** target-HF partial liquidations
  deleverage as price falls, recording an endogenous price band of
  (sell price, quantity) lots.
- **Buy side (the contribution):** an optional, per-borrower restoration leg
  that buys collateral back at the band bottom on rebounds, **reborrow-funded**
  and sized by solvency — a health-factor floor `B*` plus a stress-tested cap
  (post-buy HF ≥ 1 even after a further `d` price drop) — rather than matched
  one-to-one to the sold lots. Naive lot-matched buybacks re-lever the position
  into ping-pong re-liquidation; the solvency sizing is what makes the leg safe.

Motivating user: the BTC-denominated borrower who posts BTC for USD liquidity
but measures wealth in BTC — in their numeraire the restoration is
unconditionally a gain (it only ever adds collateral back).

The manuscript is `doc/main.tex` (build with `pdflatex` + `bibtex`).

## Setup

```sh
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

All experiments are deterministic; run them with `PYTHONHASHSEED=0`.

## Reproducing the paper

In order (or just `make paper`):

```sh
export PYTHONHASHSEED=0
./.venv/bin/python runs/sweeps/historical_window_sweep.py   # in-sample sweep -> scenario_candidate_best.json
./.venv/bin/python runs/sweeps/real_window_batch.py         # 600-run real-window stress batch
./.venv/bin/python runs/sweeps/synthetic_stress_batch.py    # 120-run synthetic deep-crash batch
./.venv/bin/python runs/sweeps/hf_floor_sweep.py            # HF-floor governance-dial frontier
./.venv/bin/python runs/sweeps/make_paper_tables.py         # regenerates the in-sample .tex tables + stats

./.venv/bin/python runs/sweeps/fetch_oos_data.py            # 15 assets, daily, 2019-2024 (Coinbase public API)
./.venv/bin/python runs/sweeps/oos_validation.py            # fixed candidate, no re-tuning, 95 independent windows
./.venv/bin/python runs/sweeps/sell_side_oos_sweep.py       # sell-side dial: borrower loss vs bad debt
./.venv/bin/python runs/sweeps/optimize_buy_principles.py   # buy-side dial: stress sizing vs timing
./.venv/bin/python runs/sweeps/make_oos_tables.py           # regenerates the OOS .tex tables
```

Sanity checks of the accounting guarantees: `make test`.

## Layout

| Path | What it is |
|---|---|
| `aave_counterfactual_pipeline.py` | The engine: target-HF sizing, HF-floor + stress-sized buyback, deterministic backtests. The paper's `\S Reproducibility` points here. |
| `runs/sweeps/*.py` | All reproduction scripts (tracked even though `runs/` outputs are git-ignored). |
| `runs/sweeps/scenario_candidate_best.json` | The fixed tuned candidate used everywhere downstream. |
| `tests/test_invariants.py` | Property tests for the closed-form guarantees (ΔD\* hits HF\*, ΔU = b·ΔD, post-buy HF ≥ floor, stress cap, buyback equity identity). |
| `doc/main.tex`, `doc/tables/`, `doc/figures/` | Manuscript and its generated inputs. |
| `band_based_lending_simulation.py` | Legacy side-track simulator; **not** used by the paper. |

## Data

Three price datasets coexist — they are different things:

| Path | Series | Role |
|---|---|---|
| `data/aave/normalized/prices.csv` | ETH, 1101 daily points, **2023-02 → 2026-02** | **In-sample**: sweep/tuning + in-sample batches (derived from the root CoinMarketCap CSV). |
| `ETH_1Y_graph_coinmarketcap.csv`, `BTC_1Y_graph_coinmarketcap.csv` | raw CoinMarketCap exports | Source for the above; BTC also feeds the independent-window check. |
| `data/oos/*.csv` (15 assets) | daily, **2019 → 2024**, Coinbase | **Out-of-sample** validation. Note: ETH 2023–24 overlaps the tuning series in time; the other 14 assets are fully disjoint (this is stated in the paper). |

## Key results (see the paper for the honest framing)

- Target-HF deleveraging Pareto-dominates legacy fixed-close-factor liquidation
  out of sample (lower bad debt, no-worse borrower loss) — Aave's result,
  used here as the baseline.
- "Zero bad debt" is a **sell-side governance frontier**, not an absolute:
  an early trigger (LLTV 0.70 / HF\* 1.25) drives OOS worst-window bad debt
  to a few hundred USD at a ~+27% borrower-loss cost.
- The guarded buyback restores strictly more collateral (accounting identity),
  adds no bad debt, and induces almost no extra liquidation; its USD benefit is
  modest and regime-dependent. **Conservative sizing beats clever timing.**
