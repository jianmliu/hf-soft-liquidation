# Aave 数据采集与反事实仿真快速开始

## 1) 初始化数据目录
```bash
make cf-init
```

会创建：
- `data/aave/raw/` 原始数据模板
- `data/aave/normalized/prices.csv` 样例价格数据
- `data/aave/normalized/positions_initial.csv` 样例账户头寸
- `data/aave/config/scenarios.json` 样例策略场景

## 2) 校验数据
```bash
make cf-validate
```

## 3) 运行反事实仿真
```bash
make cf-sim
```

运行后在 `runs/run_*/` 下输出：
- `event_log.csv`
- `account_outcomes.csv`
- `scenario_metrics.csv`
- `run_meta.json`

## 4) 生成报告
```bash
make cf-report
```

在最近一次 `runs/run_*/` 中生成 `report.md`。

## 5) 接入真实价格数据（可选）
```bash
/Volumes/T7-Data/D^2/band-soft-liquidation/.venv/bin/python aave_counterfactual_pipeline.py collect-prices \
  --asset-id ethereum \
  --symbol WETH \
  --start 2024-01-01T00:00:00+00:00 \
  --end 2024-03-01T00:00:00+00:00 \
  --out-csv data/aave/normalized/prices.csv
```

然后再次执行：
```bash
make cf-validate && make cf-sim && make cf-report
```

## 6) 接入 Aave 真实链上数据（头寸 + 清算事件）
```bash
/Volumes/T7-Data/D^2/band-soft-liquidation/.venv/bin/python aave_counterfactual_pipeline.py collect-aave-subgraph \
  --endpoint 'https://<your-graphql-endpoint>' \
  --market 'aave-v3-ethereum' \
  --symbols 'WETH,WBTC' \
  --dataset-dir data/aave \
  --start 2024-01-01T00:00:00+00:00 \
  --end 2024-03-01T00:00:00+00:00
```

会更新：
- `data/aave/raw/positions_raw.csv`
- `data/aave/raw/liquidations_raw.csv`
- `data/aave/raw/collection_meta.json`
- `data/aave/normalized/positions_initial.csv`
- `data/aave/normalized/liquidation_events.csv`

随后可继续运行：
```bash
make cf-validate && make cf-sim && make cf-report
```

## 6.1) 接入 Dune 查询结果（推荐作为链上数据替代通道）

最小命令（仅清算事件）：
```bash
/Volumes/T7-Data/D^2/band-soft-liquidation/.venv/bin/python aave_counterfactual_pipeline.py collect-dune \
  --dataset-dir data/aave \
  --dune-api-key '<your_dune_api_key>' \
  --liquidation-query-id 2104473
```

同时拉取头寸并补价格（可直接用于后续仿真）：
```bash
/Volumes/T7-Data/D^2/band-soft-liquidation/.venv/bin/python aave_counterfactual_pipeline.py collect-dune \
  --dataset-dir data/aave \
  --dune-api-key '<your_dune_api_key>' \
  --liquidation-query-id <liq_query_id> \
  --positions-query-id <positions_query_id> \
  --collect-prices
```

会写入：
- `data/aave/raw/dune_liquidations_raw.csv`
- `data/aave/raw/dune_positions_raw.csv`（若提供 `--positions-query-id`）
- `data/aave/raw/dune_collection_meta.json`
- `data/aave/normalized/liquidation_events.csv`
- `data/aave/normalized/positions_initial.csv`（若 positions 成功映射）

随后可继续运行：
```bash
make cf-validate && make cf-sim && make cf-report
```

## 7) 预置网络 + 资产映射（接近零参数启动）

查看内置预置：
```bash
make cf-list-presets
```

一键启动（默认 `ethereum-v3`，若不传时间窗口则自动使用最近 30 天）：
```bash
/Volumes/T7-Data/D^2/band-soft-liquidation/.venv/bin/python aave_counterfactual_pipeline.py bootstrap-aave-market
```

指定预置和资产子集：
```bash
/Volumes/T7-Data/D^2/band-soft-liquidation/.venv/bin/python aave_counterfactual_pipeline.py bootstrap-aave-market \
  --preset arbitrum-v3 \
  --symbols WETH,WBTC,USDC
```

说明：
- 预置同时包含 subgraph endpoint 与资产到 CoinGecko 源映射（如 `WETH -> ethereum`）。
- 若 endpoint 失效或需要私有网关，可通过 `--endpoint` 覆盖。
- 完成 bootstrap 后可直接运行：
```bash
make cf-sim && make cf-report
```

## 8) 不依赖链上接口的批量仿真

当暂时无法访问 The Graph 网关时，可先基于本地 normalized 数据批量压力仿真：

```bash
make cf-batch
```

输出：
- `runs/batch_*/batch_summary.csv`（每个批次 run 的场景指标）
- `runs/batch_*/batch_report.md`（批量汇总报告）

生成“最佳/最差参数组合 Top-N”摘要表：
```bash
make cf-batch-topn
```

输出：
- `runs/batch_*/topn_best_5.csv`
- `runs/batch_*/topn_worst_5.csv`
- `runs/batch_*/topn_summary_5.md`

导出“所有发生 BUY 的闭环案例表”（含 SELL→BUY 时差、数量回补率、净 reserve 变化）：
```bash
make cf-batch-closures
```

输出：
- `runs/batch_*/buy_closure_cases.csv`

可自定义参数（示例）：
```bash
/Volumes/T7-Data/D^2/band-soft-liquidation/.venv/bin/python aave_counterfactual_pipeline.py batch-simulate \
  --dataset-dir data/aave \
  --scenario-file data/aave/config/scenarios.json \
  --seeds 101,102,103 \
  --price-shocks -0.20,-0.10,0.00,0.10 \
  --debt-scales 0.90,1.00,1.10 \
  --noise-scales 0.00,0.02 \
  --max-runs 24
```

## 9) 当前默认策略：LLTV/LTV 内生化（无固定 tier）

`data/aave/config/scenarios.json` 现已支持并默认使用 `dynamic` 配置：

```json
[
  {
    "name": "baseline_dynamic",
    "dynamic": {
      "lltv": 0.82,
      "min_close_factor": 0.12,
      "max_close_factor": 0.55,
      "cf_slope": 1.8,
      "liquidation_bonus": 0.06,
      "buyback_ratio": 0.70,
      "recovery_ltv_gap": 0.04,
      "sell_cooldown_steps": 1,
      "buy_cooldown_steps": 1
    }
  }
]
```

说明：
- SELL 触发条件：`LTV >= LLTV`，close factor 按超阈程度连续计算（并裁剪到 `min/max_close_factor`）。
- BUY 触发条件：`LTV <= LLTV - recovery_ltv_gap`，按累计卖出量目标比例回补，受 reserve 约束。
