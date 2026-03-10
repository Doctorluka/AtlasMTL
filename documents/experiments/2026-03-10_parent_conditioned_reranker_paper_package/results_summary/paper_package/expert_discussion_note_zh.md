# Parent-Conditioned Reranker 章节讨论稿

## 章节定位

这一章节更适合被定位为一个面向 difficult deep-hierarchy case 的局部结构化 refinement 模块，而不是 AtlasMTL 的第二主模型。当前最稳妥的主张是：

在 difficult deep-hierarchy reference mapping 中，error-driven parent-conditioned hotspot reranking 可以在最佳基础多层 AtlasMTL 配置之上，同时改善 finest-level annotation 和 full-path hierarchy recovery。

## 实验设计摘要

### PH-Map 设计

PH-Map 这条线已经完整收口。设计上先从原始 reference 数据重做 `study`-isolated split，排除原有 `sample`-split 的组泄漏风险；随后确认 finest-level 强化和 finest-head class weighting 是 base model 上最有价值的改动；再通过 parent-child error decomposition 识别剩余 tradeoff 的主来源；接着验证简单的 local fixes，如 hotspot thresholding 与 shared temperature scaling，并确认它们不能解决主问题；最后把 auto parent-conditioned reranker 提升为正式 operational path，并通过 `top6 vs top8` 多 seed 稳定性确认最终默认规则。

### HLCA 设计

HLCA 使用新提供的原始数据 [hlca_clean.h5ad](/home/data/fhz/project/phmap_package/data/real_test/HLCA/hlca_clean.h5ad)，按 `obs.study` 重做了一套 `study`-grouped split，而不是直接复用旧 benchmark 抽样子集。由于 HLCA 有 5 层注释，因此没有继承 PH-Map 的 4 维权重，而是单独做了 5 维 weighting confirmation，比对 `uniform`、`mild_lv5` 和 `strong_lv5` 三组权重。之后仅在 HLCA 的 best base config 上做了第一版 AutoHotspot reranker 机制验证，目标不是复制 PH-Map 的完整探索史，而是确认“dataset-specific weighting + hotspot discovery + parent-conditioned reranker”这条机制链能否迁移到第二个 deep-hierarchy 数据集。

## 当前结果摘要

### PH-Map

PH-Map 现在已经是明确的正结果，并且可作为正文主证据。当前最终 operational path 为：

`lv4strong + per-class weighting + auto parent-conditioned reranker_top8`

在 `predict_100000_10000 + hierarchy_on` 上，多 seed 汇总结果为：

- base + class weighting：`macro_f1 = 0.587177 ± 0.005695`，`full_path_accuracy = 0.43836 ± 0.01042`，`parent_correct_child_wrong_rate = 0.12348 ± 0.00986`
- + auto reranker_top8：`macro_f1 = 0.588557 ± 0.002093`，`full_path_accuracy = 0.47216 ± 0.00453`，`parent_correct_child_wrong_rate = 0.08926 ± 0.00310`

这说明 reranker_top8 不只是提高 finest-level，也同时回补了 full-path，并显著压低主错误模式。`top8` 也已经通过多 seed 默认规则确认，可以正式替代 `top6` 成为 PH-Map 默认 hotspot rule。

### HLCA

HLCA 目前提供的是“部分支持 + mixed first-pass evidence”。

第一，HLCA 的 weighting confirmation 很清楚：`uniform` 是当前最佳 base config，说明 HLCA 不应继承 PH-Map 的 finest-level upweighting schedule，这支持了“最佳 schedule 是 dataset-specific”这一判断。

第二，HLCA 的 first-pass auto reranker 结果是 mixed 的。在 `predict_100000_10000 + hierarchy_on` 上：

- baseline uniform：`macro_f1 = 0.688732`，`full_path_accuracy = 0.8239`，`parent_correct_child_wrong_rate = 0.0334`
- + auto reranker_top6：`macro_f1 = 0.693015`，`full_path_accuracy = 0.8200`，`parent_correct_child_wrong_rate = 0.0371`

也就是说，HLCA 这版 reranker 提高了 finest-level `macro_f1`，但没有保住 `full_path_accuracy`，同时 `parent_correct_child_wrong_rate` 变差，因此没有通过 PH-Map 风格的 guardrail。当前它还不能被表述为 HLCA 上的正向 operational upgrade。

## 当前可支持的论文叙事

从论文角度，当前最稳妥的叙事是：

1. PH-Map 提供一个完整而强的正结果，证明 hardest deep-hierarchy case 上，parent-conditioned reranking 可以同时修复 finest-level 与 full-path tradeoff。
2. HLCA 作为第二个 deep-hierarchy 数据集，目前已经支持 dataset-specific weighting 这一点，但对 reranker 的证据仍是 mixed，尚不足以写成“跨数据集稳定正向复现”。
3. 因此，这一章节当前最稳妥的定位是：
   - PH-Map：正文主证据
   - HLCA：第二数据集 stress test / partial external validation

## 建议带给专家讨论的核心问题

1. 当前 HLCA 的 mixed 结果是否已经足以作为补充材料中的“第二数据集压力测试”，还是仍需要再补一轮 targeted reranker refinement 才能作为更强的 generalization evidence。
2. 正文是否应只用 PH-Map 做主 Figure，而将 HLCA 放入补充材料中作为 dataset-specific weighting + first-pass reranker transfer 的验证。
3. 如果还要继续补实验，下一步最值的方向是否应聚焦于 HLCA 的 targeted reranker refinement，而不是再回到 PH-Map 扩展更多分支。
