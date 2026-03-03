下面汇总了上文涉及的“论文级指标补齐”相关**方法学引用文献/资源**（按主题分组，Markdown 链接可直接点开）：

## 校准 Calibration（ECE / reliability diagram / temperature scaling）

* Guo et al., 2017 — *On Calibration of Modern Neural Networks*（提出并系统化 ECE、reliability diagram、temperature scaling）
  [https://arxiv.org/pdf/1706.04599](https://arxiv.org/pdf/1706.04599)
* Minderer et al., 2021 — *Revisiting the Calibration of Modern Neural Networks*（讨论现代模型校准现象、ECE/可靠性图等）
  [https://openreview.net/pdf?id=QRBvLayFXI](https://openreview.net/pdf?id=QRBvLayFXI)

## 选择性预测 / 拒识 Selective prediction（risk–coverage / AURC）

* Zhou et al., 2024/2025 — *A Novel Characterization of the Population Area Under the Risk Coverage Curve (AURC)…*（AURC 的统计定义与估计）
  [https://arxiv.org/abs/2410.15361](https://arxiv.org/abs/2410.15361)
* Zhu et al., 2022（补充材料）— *Rethinking Confidence Calibration for Failure Prediction*（对 AURC 的定义/解释：risk-coverage 视角）
  [https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850512-supp.pdf](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850512-supp.pdf)
* 2024 — *Overcoming Common Flaws in the Evaluation of Selective Classification*（讨论 selective classification 评估缺陷，并提出广义风险-覆盖指标 AUGRC）
  [https://arxiv.org/html/2407.01032v1](https://arxiv.org/html/2407.01032v1)

## 层级选择性预测 Hierarchical selective classification（与多层级标签/层级一致性强相关）

* NeurIPS 2024 Poster — *Hierarchical Selective Classification*（正式化 hierarchical risk/coverage 与层级 risk–coverage 曲线）
  [https://neurips.cc/virtual/2024/poster/93110](https://neurips.cc/virtual/2024/poster/93110)

## Conformal prediction（可作为“带统计保证”的置信/拒识替代路线）

* Shafer & Vovk, 2008 (JMLR) — *A Tutorial on Conformal Prediction*（经典教程；分类预测集/覆盖率保证）
  [https://jmlr.csail.mit.edu/papers/volume9/shafer08a/shafer08a.pdf](https://jmlr.csail.mit.edu/papers/volume9/shafer08a/shafer08a.pdf)
* arXiv 版本（同主题，便于引用/检索）
  [https://arxiv.org/abs/0706.3188](https://arxiv.org/abs/0706.3188)

## 单细胞自动注释/label transfer 基准评估（对照“点指标：accuracy/macro-F1/balanced acc”等）

* Abdelaal et al., 2019 — *A comparison of automatic cell identification methods for single-cell RNA-sequencing data*（经典 scRNA 自动细胞鉴定方法 benchmark）
  [https://pmc.ncbi.nlm.nih.gov/articles/PMC6734286/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6734286/)
* Xu et al., 2022 — *Comparison of cell type annotation algorithms…*（比较多种注释算法；适合补充材料/背景引用）
  [https://www.frontiersin.org/journals/systems-biology/articles/10.3389/fsysb.2022.1026686/full](https://www.frontiersin.org/journals/systems-biology/articles/10.3389/fsysb.2022.1026686/full)


