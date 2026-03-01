# 2026-02-28: atlasmtl 优化路线图实现记录（P0–P2 初版）

本次变更目标：在不破坏 `build_model()` / `predict()` / `TrainedModel` / `PredictionResult` 与 AnnData 写回契约的前提下，补齐可靠性（calibration / open-set / Unknown）、benchmark 可复现性、domain shift 适配入口、KNN/层级一致性与产物可追溯性。

## 1) 置信度与拒识（confidence / reject option）

- **Temperature scaling（post-hoc calibration）**（按 label head 拟合温度）  
  - 新增：`atlasmtl/mapping/calibration.py`  
  - 训练侧：`build_model(..., calibration_method="temperature_scaling", val_fraction>0)` 将温度参数写入 `train_config["calibration"]`。  
  - 推理侧：`predict(..., apply_calibration=...)` 在 softmax 前对 logits 做 `logits / T`，并在 `uns["atlasmtl"]` 记录 `calibration_applied`、`calibration_method`、`temperature_<col>`。

- **Open-set scoring（可选 Unknown 信号）**  
  - 新增：`atlasmtl/mapping/openset.py`（`openset_method="nn_distance"|"prototype"` + `openset_threshold`）。  
  - 推理侧：当 score 超过阈值时强制 `Unknown`，并记录 `openset_*` 元数据（包含 unknown rate / prototypes 数量等）。

## 2) Benchmark / evaluation “hardening”

- **扩展 `evaluate_predictions()`**：新增 `reject_rate`、并在存在 `conf_<level>` 时提供 `ECE`、`Brier`、`AURC`（selective classification）。  
  - 位置：`atlasmtl/core/evaluate.py`
- **按域分组评估**：新增 `evaluate_predictions_by_group()`，用于 domain/batch 子集的公平报告。  
  - 位置：`atlasmtl/core/evaluate.py`
- **benchmark runner 从 skeleton 变为可执行**（atlasmtl-only baseline，保留 comparator 扩展位）  
  - 位置：`benchmark/pipelines/run_benchmark.py`  
  - 产物：`metrics.json`、`summary.csv`、（可选）`summary_by_domain.csv`、`run_manifest.json`。  
  - 支持（可选）坐标指标：当 manifest 提供 `query_coord_targets` 时输出 coordinate `RMSE` 与 `trustworthiness`。

## 3) Domain shift / batch robustness（opt-in）

- **domain_key 接入**：`build_model(..., domain_key="batch")` 读取 `adata.obs[domain_key]` 并写入 `train_config`。  
- **轻量对齐损失（mean-matching）**：`domain_loss_weight>0` 时对 shared encoder 表征加入跨域均值匹配惩罚（默认关闭，避免改变基线）。  
  - 位置：`atlasmtl/core/train.py`

## 4) 结构性增强（hierarchy / KNN / topology）

- **Hierarchy enforcement（可选）**：推理后按 parent-child 映射检查路径一致性，不一致的 child 置 `Unknown` 并记录不一致率。  
  - 新增：`atlasmtl/mapping/hierarchy.py`  
  - 推理侧参数：`predict(..., enforce_hierarchy=True, hierarchy_rules=...)`

- **KNN 增强**（保持默认行为不变，全部为 opt-in）  
  - distance-weighted vote：`knn_vote_mode="distance_weighted"`  
  - prototype reference：`knn_reference_mode="prototypes"`（每个 label level 计算 centroid）  
  - approximate NN：`knn_index_mode="pynndescent"`（如环境提供 `pynndescent`）  
  - 位置：`atlasmtl/mapping/knn.py` + `atlasmtl/core/predict.py`

- **Topology-aware coordinate loss（训练可选）**：在 mini-batch 内对坐标邻域距离保持加入额外损失项。  
  - 参数：`topology_loss_weight`、`topology_k`、`topology_coord`  
  - 位置：`atlasmtl/core/train.py`

## 5) Traceability / packaging（run manifests / checksums / compression / presets）

- **manifest checksums**：`model_manifest.json` 新增 `checksums` 字段（SHA-256），便于产物审计与复现。  
  - 新增：`atlasmtl/models/checksums.py`，并在 `atlasmtl/models/serialization.py` 写入 manifest。
- **reference 压缩（opt-in）**：reference store 支持 `.pkl.gz` 读写。  
  - 位置：`atlasmtl/models/reference_store.py`
- **model presets（显式）**：新增 `preset="small|default|large"`（仅在用户未显式提供对应参数时生效）。  
  - 新增：`atlasmtl/models/presets.py`，接入：`atlasmtl/core/train.py`
- **CLI / benchmark 运行清单**：  
  - `scripts/train_atlasmtl.py` 写 `train_run_manifest.json`  
  - `scripts/predict_atlasmtl.py` 写 `predict_run_manifest.json`  
  - `benchmark/pipelines/run_benchmark.py` 写 `run_manifest.json`

## 6) 测试覆盖

新增/扩展 unit + integration tests，覆盖 calibration、open-set、benchmark runner、domain_key、topology loss config、hierarchy enforcement、reference gzip、presets，以及 KNN ANN 分支等。

## 7) Benchmark comparator 进度同步

截至 `2026-02-28`，benchmark 已不再是 atlasmtl-only。当前可运行的方法包括：

- `atlasmtl`
- `reference_knn`
- `celltypist`
- `scanvi`
- `singler`
- `symphony`
- `azimuth`

其中：

- Python comparator 环境：`/home/data/fhz/.local/share/mamba/envs/atlasmtl-env`
- 推荐 `NUMBA_CACHE_DIR=/tmp/numba_cache`
- 原生 `Azimuth` / `Seurat v5` R library：`/home/data/fhz/seurat_v5`
- repo-local comparator R library：`/home/data/fhz/project/phmap_package/atlasmtl/.r_libs`

实现状态说明：

- `celltypist`、`scanvi` 直接使用 Python 环境
- `singler`、`symphony` 使用 R bridge
- `azimuth` 优先走 native `Azimuth` backend
- 对极小 toy smoke dataset，`azimuth` 允许带标签的 fallback backend，以避免原生 Azimuth 在极小数据上的数值不稳定影响集成测试；正式 benchmark 主表应优先使用 native backend

当前 comparator benchmark 的公平比较边界也已明确：

- 主比较任务是共享目标标签层级上的 `sc -> sc` label transfer
- 主指标是 `accuracy`、`macro_f1`、`balanced_accuracy`、`coverage`、`reject_rate`、`ece`、`brier`、`aurc`
- `hierarchy`、`KNN rescue`、`open-set`、`coordinate/topology` 主要作为 atlasmtl 特色分析，不强行要求所有 comparator 具备完全同构能力

## 兼容性说明

- 默认行为保持：calibration、open-set、domain loss、hierarchy enforcement、distance-weighted vote、prototype reference、ANN index、topology loss 都是 **opt-in**。  
- `uns["atlasmtl"]` 新增字段为加法；现有字段与写回模式保持不变。

## 对照 roadmap 仍未完成的部分

- **Benchmark protocol 仍未完全固化**：当前已有 runner 与核心指标，但 dataset manifest schema、split 记录、输出 schema、以及文档化协议仍需继续补齐。
- **Domain shift 仍偏“入口完成”而非“协议完成”**：已有 `domain_key` 与分组评估，但 in-domain / cross-domain / held-out 场景定义及失败分析报表仍未标准化。
- **Hierarchy 仍缺 benchmark 指标闭环**：推理侧已有 enforcement，但 full-path accuracy、path consistency 等尚未稳定进入 benchmark 输出。
- **KNN 变体仍缺系统 benchmark**：vote mode、reference mode、ANN index 已有接口，但尚未形成标准 ablation matrix 与统一报表。
- **Topology-aware 评估仍不完整**：训练项已加，benchmark 目前主要只有 RMSE / trustworthiness，continuity、neighbor overlap 等仍未进入稳定协议。
- **Traceability 仍需继续增强**：checksums 与 run manifests 已实现，但 split / seed / protocol version 的统一约定仍需继续收敛。
- **P2 项目仍未真正落地**：query-time adaptation 尚未实现；compressed reference 与 presets 也尚未完成系统 benchmark 验证。
