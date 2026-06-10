# Portable: defaults to the project venv, override with e.g. `make PYTHON=python3 ...`.
PYTHON ?= ./.venv/bin/python
RUN := PYTHONHASHSEED=0 $(PYTHON)

.PHONY: paper paper-insample paper-oos tables test \
	cf-init cf-validate cf-sim cf-report cf-batch cf-batch-topn cf-batch-closures \
	cf-collect-subgraph cf-collect-dune cf-list-presets cf-bootstrap-market

# --- Paper reproduction pipeline (see README) ---------------------------------
paper: paper-insample paper-oos

paper-insample:
	$(RUN) runs/sweeps/historical_window_sweep.py
	$(RUN) runs/sweeps/real_window_batch.py
	$(RUN) runs/sweeps/synthetic_stress_batch.py
	$(RUN) runs/sweeps/hf_floor_sweep.py
	$(RUN) runs/sweeps/make_paper_tables.py

paper-oos:
	$(RUN) runs/sweeps/fetch_oos_data.py
	$(RUN) runs/sweeps/oos_validation.py
	$(RUN) runs/sweeps/sell_side_oos_sweep.py
	$(RUN) runs/sweeps/optimize_buy_principles.py
	$(RUN) runs/sweeps/make_oos_tables.py

tables:
	$(RUN) runs/sweeps/make_paper_tables.py
	$(RUN) runs/sweeps/make_oos_tables.py

test:
	$(RUN) tests/test_invariants.py

# --- Engine CLI helpers --------------------------------------------------------

cf-init:
	$(PYTHON) aave_counterfactual_pipeline.py init-dataset --dataset-dir data/aave

cf-validate:
	$(PYTHON) aave_counterfactual_pipeline.py validate-dataset --dataset-dir data/aave

cf-sim:
	$(PYTHON) aave_counterfactual_pipeline.py simulate --dataset-dir data/aave --scenario-file data/aave/config/scenarios.json --output-dir runs

cf-batch:
	$(PYTHON) aave_counterfactual_pipeline.py batch-simulate --dataset-dir data/aave --scenario-file data/aave/config/scenarios.json --output-dir runs --seeds="41,42,43" --price-shocks="-0.15,0.00,0.10" --debt-scales="0.95,1.00,1.05" --noise-scales="0.00,0.01" --max-runs 18

cf-batch-topn:
	@latest_batch=$$(ls -td runs/batch_* 2>/dev/null | head -n 1); \
	if [ -z "$$latest_batch" ]; then \
		echo "No batch directory found under runs/"; \
		exit 1; \
	fi; \
	$(PYTHON) aave_counterfactual_pipeline.py batch-topn --batch-dir $$latest_batch --top-n 5

cf-batch-closures:
	@latest_batch=$$(ls -td runs/batch_* 2>/dev/null | head -n 1); \
	if [ -z "$$latest_batch" ]; then \
		echo "No batch directory found under runs/"; \
		exit 1; \
	fi; \
	$(PYTHON) aave_counterfactual_pipeline.py batch-buy-closures --batch-dir $$latest_batch

cf-report:
	@latest_run=$$(ls -td runs/run_* 2>/dev/null | head -n 1); \
	if [ -z "$$latest_run" ]; then \
		echo "No run directory found under runs/"; \
		exit 1; \
	fi; \
	$(PYTHON) aave_counterfactual_pipeline.py report --run-dir $$latest_run

cf-collect-subgraph:
	$(PYTHON) aave_counterfactual_pipeline.py collect-aave-subgraph --endpoint "$(AAVE_SUBGRAPH_ENDPOINT)" --market "$(AAVE_MARKET)" --symbols "$(AAVE_SYMBOLS)" --dataset-dir data/aave --start "$(AAVE_START)" --end "$(AAVE_END)" --page-size 1000 --max-pages 25 --api-key "$(AAVE_API_KEY)"

cf-collect-dune:
	$(PYTHON) aave_counterfactual_pipeline.py collect-dune --dataset-dir data/aave --dune-api-key "$(DUNE_API_KEY)" --liquidation-query-id "$(DUNE_LIQ_QUERY_ID)" --positions-query-id "$(DUNE_POS_QUERY_ID)" --liquidation-threshold 0.83 --collect-prices

cf-list-presets:
	$(PYTHON) aave_counterfactual_pipeline.py list-presets

cf-bootstrap-market:
	$(PYTHON) aave_counterfactual_pipeline.py bootstrap-aave-market --preset "$(AAVE_PRESET)" --dataset-dir data/aave --symbols "$(AAVE_SYMBOLS)" --start "$(AAVE_START)" --end "$(AAVE_END)" --endpoint "$(AAVE_SUBGRAPH_ENDPOINT)" --page-size 1000 --max-pages 25 --api-key "$(AAVE_API_KEY)"
