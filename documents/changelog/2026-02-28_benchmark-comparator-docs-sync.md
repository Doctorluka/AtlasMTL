# 2026-02-28: benchmark comparator 与文档同步增量记录

本记录用于补充 `documents/changelog/2026-02-28_atlasmtl_optimization_roadmap_implementation.md` 之后的新增进度，重点是 benchmark comparator 落地、环境路径同步、协议约束更新，以及全局设计总结文档补齐。

## 1) Benchmark comparator 实现进度补充

当前 benchmark 已可运行的方法：

- `atlasmtl`
- `reference_knn`
- `celltypist`
- `scanvi`
- `singler`
- `symphony`
- `azimuth`

与前一版记录相比，这里的关键变化不是 atlasmtl 主模型接口，而是 comparator benchmark 已从“预留扩展位”变为“已有一批可运行 published baselines”。

## 2) 原生 Azimuth 路径补充

`azimuth` comparator 当前实现为：

- 正式路径：native `Azimuth` + `Seurat v5`
- 元数据中记录 `implementation_backend`
- 极小 toy smoke dataset 下允许使用显式标记的 fallback backend

这样做的原因是：

- 原生 `Azimuth` 在正式 benchmark 场景下应作为主实现
- 但极小 synthetic 测试常出现数值不稳定、neighbor graph 或 anchor 搜索异常
- 为了保证集成测试稳定性，允许 fallback，但不应把 fallback 结果与正式 native benchmark 混为一谈

## 3) 环境与依赖路径已纳入文档约束

此次同步新增了 benchmark 相关运行环境记录：

- Python env：`/home/data/fhz/.local/share/mamba/envs/atlasmtl-env`
- 推荐 `NUMBA_CACHE_DIR=/tmp/numba_cache`
- native `Azimuth` / `Seurat v5` R library：`/home/data/fhz/seurat_v5`
- repo-local comparator R library：`/home/data/fhz/project/phmap_package/atlasmtl/.r_libs`

这些路径现已写入：

- `README.md`
- `AGENTS.md`
- `benchmark/README.md`
- `documents/protocols/experiment_protocol.md`

## 4) Comparator benchmark fairness contract 已明确

本次更新把 comparator benchmark 的公平比较边界正式写入协议：

- 主比较任务是共享目标标签层级上的 `sc -> sc` label transfer
- 主指标是：
  - `accuracy`
  - `macro_f1`
  - `balanced_accuracy`
  - `coverage`
  - `reject_rate`
  - `ece`
  - `brier`
  - `aurc`
- `hierarchy`、`KNN rescue`、`open-set`、`coordinate/topology` 主要作为 atlasmtl 的特色分析，不强行要求外部 comparator 完全同构

这意味着 benchmark 叙事已经从“能跑”推进到“知道该如何公平解释结果”。

## 5) 新增 protocol template

新增：

- `documents/protocols/comparator_benchmark_result_template.md`

其作用是为正式 benchmark 结果记录提供统一模板，包括：

- run identity
- comparator 列表
- main comparison table
- domain-wise table
- atlasmtl-specific analysis
- coordinate/topology diagnostics
- fairness checklist
- environment record

这为后续真实数据集 benchmark 和论文结果表整理提供了直接起点。

## 6) 新增全局设计总结

新增：

- `documents/design/overall_summary.md`

该文档从全局角度总结 atlasmtl，包括：

- 框架定位
- 核心目的
- 模块结构
- 涉及算法
- 训练 / 推理 / artifact 实现途径
- benchmark 设计
- comparator 现状
- 环境依赖
- 当前优势与局限

这份文档可直接作为论文写作、方法总览和后续架构决策的参考基线。

## 7) 本次同步涉及的主要文件

- `README.md`
- `AGENTS.md`
- `benchmark/README.md`
- `documents/protocols/experiment_protocol.md`
- `documents/protocols/comparator_benchmark_result_template.md`
- `documents/design/overall_summary.md`
- `documents/changelog/2026-02-28_benchmark-comparator-docs-sync.md`

## 8) 当前状态判断

截至本次更新：

- comparator benchmark 已具备第一阶段可比性
- benchmark 环境与路径约束已被文档化
- comparator 公平比较协议已明确
- 全局设计总结文档已补齐

下一步最自然的工作是：

- 将 comparator protocol template 用到真实 benchmark 结果
- 开始系统化 comparator matrix / 多数据集 benchmark 执行

## 9) Feature panel artifact decoupling

模型序列化现在会在 preprocessing 元数据包含 `feature_panel` 时额外写出
`*_feature_panel.json`，并在 `model_manifest.json` 中记录
`feature_panel_path`。模型加载时会优先从这个独立 artifact 回填
`train_config["preprocess"]["feature_panel"]`，保留兼容性的同时降低
对 `train_config` 和 metadata pickle 的单点耦合。
# 2026-02-28 Gene namespace and feature policy sync

- clarified that atlasmtl currently keeps phmap-consistent `input_transform="binary"` by default
- formalized versionless Ensembl IDs as the recommended internal gene namespace
- documented that readable symbols should be preserved as metadata rather than used as the sole formal alignment key
- added protocol requirements to record:
  - original `var_names` type
  - species
  - mapping table/resource
  - duplicate and unmapped gene handling
- recorded the preferred feature policy for formal experiments:
  - canonicalize genes first
  - use a reference-derived HVG panel second
  - treat whole-matrix training as an explicit ablation or special-case run
- bundled a repackaged BioMart mapping file at `atlasmtl/resources/gene_id/biomart_human_mouse_rat.tsv.gz`
  with explicit human/mouse/rat column names for future preprocessing support
- added a dedicated preprocessing package under `atlasmtl/preprocess/`
- added explicit preprocessing APIs for:
  - reference canonicalization
  - query canonicalization and feature alignment
  - HVG or whole-matrix feature-panel selection
  - metadata traceability through CLI, model artifacts, and benchmark manifests
