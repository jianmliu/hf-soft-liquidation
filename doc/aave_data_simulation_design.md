# Aave 真实数据采集与反事实清算仿真设计文档

## 1. 目标
- 建立可复现的数据管道，持续采集 Aave 历史状态与事件，用于清算风险评估。
- 在统一历史路径上进行反事实回放：对比 Baseline（历史参数）与 Candidate（新策略）的结果差异。
- 输出研究可用指标：清算触发率、坏账、协议收入、用户净损失、执行成功率、系统稳定性。
- 支持策略快速迭代：新增策略后可在标准数据集上一键重跑并生成对比报告。

## 2. 非目标
- 不构建生产级自动交易或清算机器人。
- 不覆盖所有 DeFi 协议，首期仅面向 Aave 指定网络与版本。
- 不追求 mempool 级微观复现，首期默认区块级回放。
- 不在首期引入复杂多智能体博弈行为。

## 3. 总体架构
- 数据层：Archive 节点、事件索引、价格源（Chainlink/TWAP）、参数快照。
- 标准化层：将头寸、价格、参数统一映射到区块时间轴。
- 仿真层：事件驱动回放引擎，按区块推进并执行 HF/LTV 触发判定。
- 策略层：插拔式策略接口（阈值、close factor、liquidation bonus、延迟、重试）。
- 评估层：指标聚合、统计检验、图表与报告输出。

## 4. 数据字段清单
- 头寸维度：address、asset、collateral_amount、debt_amount、rate_mode、indexes。
- 协议参数：LTV、liquidation threshold（LT）、liquidation bonus、reserve factor、eMode。
- 市场状态：block_number、timestamp、price、liquidity、utilization。
- 清算事件：liquidation_call、repay_amount、seized_collateral、executor、fee。
- 外生变量：gas_price、oracle_delay、price_deviation。

## 5. Schema 建议
- 分层表：raw_events、normalized_snapshots、simulation_runs、metrics_outputs。
- 关键主键：network + protocol_version + block_number + tx_index + log_index。
- 场景配置：scenario_config（JSON）+ config_hash（保证同参可复跑）。
- 结果输出：scenario_metrics（全局）+ account_outcomes（账户级）。
- 追踪标识：run_id 贯穿输入、日志、指标与图表产物。

## 6. 回放精度要求
- 默认区块级回放，关键窗口支持交易级增强回放。
- 状态对齐要求：价格、指数、协议参数必须与对应区块一致。
- 清算判定一致性：HF/LTV 计算与目标 Aave 版本保持一致。
- 偏差门限：核心指标相对历史误差按指标分级控制在 1%–5%。
- 执行模式：fast（抽样）与 audit（全量）双轨。

## 7. 策略接口
- 输入：block_state、account_state、market_state、strategy_params。
- 输出：是否触发、部分清算比例、资产路径、执行延迟、失败重试动作。
- 约束：纯函数优先，避免隐式全局状态；参数必须可序列化。
- 最小策略集：Baseline、Conservative、Aggressive、Latency-aware。
- 版本要求：每个策略绑定 strategy_version 与默认参数快照。

## 8. 指标体系
- 风险指标：坏账规模、清算账户占比、尾部 HF 分位数、最大单日损失。
- 效率指标：清算完成率、平均清算延迟、失败交易比例、资本回收率。
- 经济指标：协议收入、清算执行者收益、用户净损失、价格冲击成本。
- 稳定性指标：极端行情穿仓率、连续清算链长度、相关性放大效应。
- 对比口径：相对 Baseline 的绝对差、百分比差、置信区间。

## 9. 数据质量检查
- 完整性：区块连续、事件无缺口、主键去重。
- 一致性：多源价格偏差检查、参数快照与链上读取对齐。
- 合法性：数量非负、地址格式正确、关键指数单调约束。
- 对账：抽样核验历史清算事件与回放触发结果。
- 报警：缺失率、异常跳变率、对账偏差超过阈值自动告警。

## 10. 稳健性与统计
- 敏感性分析：价格冲击、oracle 延迟、gas 拥堵的单因子与多因子扰动。
- 重采样：按时间窗口进行 block bootstrap。
- 显著性：参数检验与非参数检验双路径验证。
- 极值处理：winsorize 与原值并行报告，避免单一口径偏差。
- 决策门槛：仅在效果方向一致且显著时给出策略推荐。

## 11. 可复现性
- 数据、配置、结果均做内容哈希并版本化。
- 固定依赖版本、容器镜像与随机种子。
- 提供统一入口命令执行拉数、回放、评估、报告生成。
- 每次实验按 run_id 归档参数、日志、指标、图表。
- 记录代码版本与配置差异，支持审计追踪。

## 12. 里程碑计划
- M1（1–2 周）：确定网络与版本范围，完成字段字典与最小采集链路。
- M2（2–4 周）：落地标准化 Schema 与区块级回放 Baseline。
- M3（2 周）：接入策略插件、核心指标面板、质量检查与告警。
- M4（2 周）：完成稳健性实验与统计报告，形成策略建议。
- M5（1 周）：复现打包、文档定稿、交付验收与后续迭代清单。

## 13. 术语对齐约定（与现有仓库一致）
- 使用 trigger tier，不使用 band 作为实现层主术语。
- 使用 HF/LTV 触发与 partial liquidation，不使用一次性 hard liquidation 作为默认执行语义。
- 使用 liquidation bonus buffer 描述清算缓冲收益来源。
- 使用 buyback（inventory-constrained）描述回补动作。
- 使用 permissionless liquidator 描述双向执行角色（sell-side / buy-side）。

## 14. 与当前仓库映射
- 论文术语与机制说明来源：doc/main.tex。
- 仿真与事件引擎实现来源：band_based_lending_simulation.py。
- 引用与背景文献来源：doc/references.bib。
- 本文档定位：作为研究执行层设计说明，连接论文叙事与仿真实现。