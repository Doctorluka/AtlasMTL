# 2026-02-28: PH-Map -> atlasmtl 方法学继承与变更清单 + atlasmtl 核心审查

本文档目的：
1) 明确 `phmap` 论文方法学与旧包实现的关键要点。
2) 对照 `atlasmtl` 当前实现，列出“继承点 / 变更点 / 缺口与风险”。
3) 按功能审查 `atlasmtl` 核心模块，给出优化建议与优先级。

参考材料：
- `vendor/phmap_methodlogy.md`：PH-Map 论文方法学摘要（MTL、输入二值化、基因对齐、概率输出、可视化）。
- `vendor/phmap_snapshot/`：PH-Map 旧包代码快照（训练/预测/预训练模型注册/plotting）。
- `atlasmtl/core/`：当前新包核心训练与推理实现。

---

## A. 方法学继承与变更清单（phmap -> atlasmtl）

### A1. 继承（保持一致或同源演进）

- 多任务学习（MTL）层级注释：共享特征提取 backbone + 每一层级独立分类头（hard parameter sharing）。
- `AnnData in -> AnnData out` 使用模式：训练基于 `AnnData.X` + `AnnData.obs`，推理结果可回写到 `AnnData`。
- 基因对齐（gene alignment）：推理时将 query 映射到训练基因空间，不存在的基因用 0 填充。
- `TrainedModel` + metadata 打包思路：保存模型参数、label encoder、训练基因等，用于可移植推理。

### A2. 变更（atlasmtl 相比 phmap 的关键扩展/偏离）

1) 任务集合从“纯层级分类”扩展为“层级分类 + 坐标回归”
- `phmap`：仅输出各层级分类 logits/probabilities。
- `atlasmtl`：新增 `coord_heads`，同时预测参考坐标（默认 `latent` 与 `UMAP`），训练损失为分类 + 坐标回归（Huber）。
- 含义：方法从“层级分类器”演进为“参考感知的映射/定位 + 分类”。

2) 推理从“概率输出”扩展为“置信门控 + 低置信纠错 + Unknown 拒识”
- `phmap`：可输出概率与可视化，但不包含 KNN 纠错与 abstention 策略。
- `atlasmtl`：对每个层级计算 `max_prob` 与 `margin`，低置信触发 KNN（默认仅对低置信子集），并用阈值输出 `Unknown`。
- 默认阈值在 `configs/thresholds/default.yaml` 中定义（`confidence_high=0.7`, `margin_threshold=0.2`, `confidence_low=0.4`）。

3) 输入表征假设发生偏离（重要）
- `phmap` 方法学与实现均强调输入二值化（`X = (X > 0)`）。
- `atlasmtl` 当前实现直接使用 `float32`（未二值化），这会改变模型学习问题与与旧结果的可比性。
- 结论：如果后续论文叙事仍以 PH-Map 方法学为基线，必须明确声明这一偏离，或提供可配置的二值化开关并在 benchmark 中做对照。

4) 训练策略差异：早停/验证集与稳定性
- `phmap`：训练包含 train/val 切分与 early stopping（patience），强调可复现训练流程。
- `atlasmtl`：当前为固定 epoch 训练、无验证/早停逻辑（更像可运行原型）。
- 影响：结果稳定性与过拟合风险更高，也更难与 `phmap` 报告方式对齐。

5) 预训练模型注册与 plotting 支持
- `phmap`：包含预训练模型注册表与加载逻辑，以及概率图、Sankey 图等可视化工具。
- `atlasmtl`：目前对外 API 精简，plotting/模型注册尚未实装（目录存在但为空）。

### A3. 缺口与风险（影响可复现、benchmark、公允比较）

- `Unknown` 策略目前只依赖 MTL 的 `max_prob < confidence_low`，未实现“MTL 低置信 -> KNN 修正 -> 若仍低置信再 Unknown”的闭环置信再评估。
- KNN 修正后不重新计算置信度指标（例如基于邻居投票一致性或距离加权概率），因此无法在方法学上支撑“修正后仍低置信则 Unknown”这一设计目标。
- 缺少 `knn_correction="off"` 的显式模式（计划中要求 `off/low_conf_only/all`）。
- `phmap` 的 `evaluate()`（基础评估能力）在 `atlasmtl` 中尚未出现，benchmark 阶段会需要这一层能力或等价替代。

---

## B. atlasmtl 核心模块按功能审查（总结 + 评价 + 优化建议）

### B1. 输入与基因对齐（`_extract_matrix`）

现状：
- 支持稀疏矩阵 `.toarray()`，并提供训练基因对齐（missing gene -> 0 padding）。
- 直接使用数值表达（float32），未二值化。

评价：
- 基因对齐是正确且必要的。
- `.toarray()` 在大数据上可能造成内存峰值；二值化与否应当显式化，否则难以复现 `phmap` 方法学与结果。

建议（高优先级）：
- 增加可配置的输入变换策略（至少支持 `binary` 与 `float`），并在 `uns["atlasmtl"]` 记录该策略，保证可追溯。
- 避免无条件 densify：优先支持稀疏到 torch 的高效路径或 chunk 化转换（至少在推理路径）。

### B2. 模型结构（`AtlasMTLModel`）

现状：
- MLP encoder + 多分类 head + 多坐标 head，结构清晰。
- 默认隐藏层大小与 dropout 与 `phmap` 不同（`atlasmtl` 默认 `[256, 128]`, `0.3`；`phmap` 默认 `[200, 100]`, `0.4`）。

评价：
- 作为可运行基线完全足够，结构表达了方法演进方向。
- 若目标是与 `phmap` 基线严格对照，需要提供“对齐的默认超参 preset”，否则对照实验会混入架构差异。

建议（中高优先级）：
- 在 `configs/model/` 中提供一套 `phmap_compatible` 预设（hidden sizes、dropout、输入二值化、epoch、batch 等），使 ablation 更干净。

### B3. 训练循环（`build_model`）

现状：
- 仅训练集循环，无 val/early stopping。
- 坐标目标标准化（mean/std），训练预测的是标准化坐标，推理再反标准化。
- 保存参考坐标与标签用于 KNN。

评价：
- 坐标标准化是正确的工程细节。
- 无早停导致“方法学对齐 phmap”存在缺口，也使得超参对比更难。

建议（高优先级）：
- 增加可选的验证集切分与 early stopping（默认可关闭，保持当前简单流程）。
- `task_weights` 默认值当前为全 1.0；若论文叙事沿用“细粒度层级权重大”的 `phmap` 观点，建议提供推荐权重并在配置中明确。

### B4. 推理、置信门控与 Unknown（`predict`）

现状：
- 为每个 level 计算 `max_prob` 与 `margin`，并产出：
  - `pred_<level>_raw`
  - `conf_<level>`
  - `margin_<level>`
  - `is_low_conf_<level>`
  - `pred_<level>`
  - `is_unknown_<level>`
- `Unknown` 判定仅依赖 `max_prob < confidence_low`（MTL 的原始概率）。

评价：
- 输出契约基本符合计划文档，且字段命名一致性较好。
- 但“Unknown 与 KNN 纠错策略”目前并未形成闭环：纠错改变了 label，但不会改变 `conf/margin/is_unknown` 的判定依据，方法学上不够自洽。

建议（最高优先级）：
- 引入“纠错后置信再评估”的最小实现（例如：邻居投票占比作为 `knn_conf`，或距离加权投票），并让 `Unknown` 能基于 MTL 与 KNN 的综合置信来决定。
- 增加 `knn_correction="off"` 模式，满足计划验收标准。
- 在 `uns["atlasmtl"]` 中记录更多运行信息（如使用的空间 `latent/umap`、是否发生 KNN、每层 KNN 覆盖率等），便于 benchmark 分析。

### B5. KNN 修正（当前在 `core/api.py` 内部实现）

现状：
- 以 `X_pred_latent` 优先，否则用 `X_pred_umap` 作为 query 空间；reference 空间用保存的 `X_ref_latent` / `X_ref_umap`。
- KNN 输出为多数投票标签。

评价：
- “只对低置信子集做 KNN”符合计划，也能省计算。
- 但 KNN 投票没有置信输出；也没有处理类不平衡、距离权重等细节。

建议（中高优先级）：
- KNN 返回同时返回 `knn_label` 与 `knn_vote_frac`（或 top1-top2 vote margin），作为纠错置信的基础。
- 将 KNN 与 Unknown 逻辑迁移到 `atlasmtl/mapping/`，使 `core/api.py` 只负责编排（更符合仓库设计）。

### B6. 序列化与可复现性（`TrainedModel.save/load` + metadata）

现状：
- 保存 `.pth` 与 `_metadata.pkl`，metadata 含 label encoders、genes、coord stats、reference coords/labels 等。

评价：
- 可用且简单，能满足原型迭代。
- metadata 包含 `reference_coords/reference_labels`，文件可能很大；如果 reference 很大，会显著影响模型可分发性。

建议（中优先级）：
- 将 reference 数据持久化策略做成可选：例如只存 KNN index 所需的信息、或存子采样 reference、或把 reference 单独文件/路径化。
- 记录训练配置（hidden sizes、dropout、epoch、输入变换策略等）到 metadata，便于追溯与论文写作。

---

## C. 建议的优化优先级（面向当前开发阶段）

P0（直接影响方法学自洽与 benchmark 公允性）：
- 输入表征策略显式化（至少支持 binary vs float），并记录到 `uns/metadata`。
- `Unknown` 与 KNN 纠错闭环：纠错后需要可解释的置信再评估；Unknown 不应只看纠错前的 MTL 概率。
- `knn_correction` 增加 `off` 模式，满足计划验收项。

P1（提高可复现与可比性）：
- 可选 early stopping + 验证集切分。
- 提供 `phmap_compatible` 的超参 preset，支持干净 ablation。
- 推理/训练避免无条件 densify（至少推理分批、稀疏友好）。

P2（工程化与论文资产准备）：
- 将 mapping/unknown/IO 拆到对应模块目录，降低 `core/api.py` 复杂度。
- 补齐 `evaluate()` 或 benchmark 所需的最小评估工具链。
- 增补模型注册与 plotting（若论文/用户需要）。
