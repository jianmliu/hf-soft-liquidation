# Submission-Style Final Checklist (One-Page)

Scope: all `.tex` files under `doc/`, with title/abstract from `doc/main.tex`.

## A. Title Check
- Title: "Dynamic LLTV/LTV Soft Liquidation for Collateralized Lending"
- Word count: 8
- Verdict: concise and mechanism-specific; keep as is.

## B. Abstract Length
- Word count: 119
- Verdict: compact and within common limits (150–250), no forced trimming needed.

## C. Caption Length Check
- Captions found: 14
- [1] 8 words (OK) | doc/main.tex | Single-position comparison: traditional liquidation vs dynamic soft liquidation.
- [2] 9 words (OK) | doc/main.tex | Synthetic collateral price path with HF-trigger bands and crossings.
- [3] 9 words (OK) | doc/main.tex | Collateral ratio trajectories with buy/sell markers across loans.
- [4] 10 words (OK) | doc/main.tex | Collateral price path with loan start times and partial-liquidation events.
- [5] 3 words (OK) | doc/main.tex | Baseline simulation template.
- [6] 7 words (OK) | doc/tables/aggregate_metrics_table.tex | Aggregate metrics for the current experiment run.
- [7] 11 words (OK) | doc/tables/baseline_batch_evidence_table.tex | Baseline-only batch evidence for the current partial-liquidation design (batch\_20260217\_104432).
- [8] 9 words (OK) | doc/tables/borrower_loss_batch_120_table.tex | Borrower final-loss statistics across 120 batch runs (baseline\_dynamic).
- [9] 8 words (OK) | doc/tables/impermanent_loss_table.tex | Borrower collateral restoration and impermanent loss by loan.
- [10] 8 words (OK) | doc/tables/liquidation_sample_stats_table.tex | Dune liquidation sample statistics (query 2104473, normalized dataset).
- [11] 8 words (OK) | doc/tables/protocol_profit_table.tex | Per-loan partial-liquidation turnover, debt repayment, and reserve PnL.
- [12] 7 words (OK) | doc/tables/simulation_config_table.tex | Simulation configuration used to generate current artifacts.
- [13] 10 words (OK) | doc/tables/soft_vs_traditional_borrower_loss_table.tex | Soft vs. traditional borrower final-loss comparison across 120 paired runs.
- [14] 9 words (OK) | doc/tables/three_strategy_borrower_loss_table.tex | Three-strategy comparison on borrower final loss (120 paired runs).
- Verdict: all captions are concise.

## D. Terminology Consistency
- target-HF: 26
- target HF: 0
- target_hf: 0
- close factor: 11
- buyback: 43
- soft liquidation: 11
- LLTV: 27
- LTV: 32
- Verdict: target-HF and buyback terminology is dominant and consistent with three-strategy framing.
- Note: keep "close factor" mentions explicitly scoped to traditional baseline comparisons.

## E. Final Submission Notes
- PDF build status: pass (`latexmk -g -pdf -interaction=nonstopmode main.tex`).
- Remaining warnings are non-blocking typography/float-level warnings.
- Ready for submission package export.
