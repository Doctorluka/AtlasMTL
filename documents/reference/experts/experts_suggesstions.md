从“顶刊标准论文”的角度看，你们当前的指标体系**方向是对的，但还不够“论文级够用”**：

* **分类性能（accuracy / macro-F1 / balanced acc）**是必要项，且宏平均指标在单细胞长尾细胞类型场景里是主流（macro-F1 被明确用于强调稀有类表现）([Nature][1])。
* 但如果你们把定位写成“可靠 label transfer（校准+拒识+层级一致性+可审计）”，那么顶刊审稿通常会要求更完整的**可靠性/开放集/泛化**证据链，而不仅是三四个点指标。

下面按“是否够用”拆成三层：**必须补齐（否则容易被认为证据不足）**、**强烈建议（显著抬升可信度）**、**加分项（冲顶刊更稳）**。

---

## 1) 目前已有指标：哪些在顶刊语境下“够用”

你们现有的核心集合大体是：

* 单标签性能：accuracy、macro-F1、balanced accuracy（你们报告已用）
* 多层级：per-level 指标 + hierarchy consistency、fulage/covered acc（你们报告已用）
* 拒识：coverage / covered accuracy / unknown rat
* 校准：temperature scaling 支持（但你们目前更像“实现了”，而不是“论

**结论：**

* 若目标是 **Bioinformatics 类（如 Briefings，再补齐 2–3 个可靠性评价模块后就接近“够用”。
* 若目标是 **Nat Biotech / Nat Methods / Nat Protocols** 这种“方法学顶级”，现在的指标更像“内部工程报告级”，需要把可靠性、泛化和开放集评估做成**成体系的 benchmark story**。

---

## 2) 必须补齐的“论文级”评价模块（强建议当作门槛）

### A. 校准（Calibration）必须从“有实现”变成“有系统证据”

顶刊会希望看到：

* **ECE / MCE**（至少 ECE）
* reliability diagram（可视化）
* 校准前后对比（含阈值拒识场景）

理由：你们定位强调 calibrated confidence。许多工作把 ECE/MCE + selective prediction 作为可靠性评估标准组件。([ScienceDirect][2])

> 你们现在只有“temperature scaling exists”，在论文里通常不够，需要定量展示“校准改善且不牺牲覆盖/性能”。

### B. Selective prediction / Abstention 需要完整曲线，而不只是单点 coverage

建议加入：

* **Risk–coverage curve**（或 accuracy–*AURC**（area under risk-coverage curve）或同类汇总指标
* “在相同 coverage 下比较各方法的 covered accuracy / risk”

这会把你们的“拒识”从 heuristic 变成可对比的论文证据链。([ScienceDirect][2])

### C. 泛化与稳健性：必须跨数据集/跨条件，而不是单一 sampled bundle

你们自己也写了“Second dataset validation 是缺口”。顶刊方法学审稿几乎必问：

* 跨组织 / 跨平台 / 跨批次
* label set shift（参考缺少某些细胞/或 query 出现新群）
* 同类工具在不同场景下的稳定性（均值±方差）

至少要做到：**2–3 个独立 query/reference 组合**，并包含一个明显 shift 场景。

---

## 3) 强烈建议补的指标与分析（显著提升说服力）

### D. 长尾与稀有类性能的专门评估

你们当前用 macro-F1、balanced a往还希望看到更细的分解：

* 按 cell type 频率分箱（head/mid/tail）的 F1/recall
* “最难 K 个 cell types”的性能对比
* confusion 的层级分解（跨父类错误率 vs 同父类内部混淆）

macro-F1 用来强调稀有类更敏感是合理的，但最好加这些分解更“论文级”。([Nature][1])

### E. 资源与效率：CPU-first 的定位必须有“系统 benchmark 表”

如果你们强调资源受限用户：

* 训练/推理 wall time、峰值 RSS、吞吐（cells/s）
* 与 cell number / gene number 的 scaling 曲线（至少 2–3 个规模点）
* CPU vs GPU 的加速比（你们工程报告已有时间/内存片段，但论文需要更系统）

这部分在 Nat Biotech/Nat Methods 的 methods paper 里经常是重要的审稿点（“能不能用、跑不跑得起”）。

---

## 4) 加分项（冲 Nat Biotech/Nat Protocols 更稳）

### F. 开放集/OOD 检测的标准指标

如果要把 Unknown/abstention 讲成“open-set mapping”，建议加入：

* Unknown 作为正类的 AUROC / AUPRC（在已知/未知标签的构造实验里）
* 在“query 含 novel cell states”的设定下，比较误标率 vs unknown 召回率

### G.

* calibration 之后，置信度分箱的实际准确率单调性更好
* margin 与错误率的相关性（你们已输出 margin）

### H. 对照基线的公平性与协议清晰度

你们已经做了 comparator closure，但论文还会被追问：

* 每个对照方法的最佳实践参数是否合理
* 训练数据泄漏/同源数据的控制
* 评估协议是否对所有方法一致

---

## 5) 回答你的核心问题：现在“够用吗”？

**如果以顶刊方法论文标准：不够。**
原因不是你们现有指标“错”，而是缺少三类关键闭环：

1. **校准/拒识的曲线化与汇总指标**（ECE + risk-coverage/AURC）
2. **跨数据集与 shift 场景的泛化证据**（至少 2–3 套）
3. **CPU-first 的系统资时间/内存片段）

补齐后，你们的定位（reliable multi-level label transfer）会更像“顶刊可发表的系统方法主张”。

---

## 6) 最小增量补齐清单（建议你们下一轮实验直接纳入）

1. 在现有输出里新增并固定：**ECE** + reliability diagram（校准前/后）
2. 对 abstention：输出 **risk–coverage curve** + **AURC**
3. 增加 **第二数据集**（你们报告中也标为缺口）
4. 资源表：CPU/GPU 的 time/RSS/throughput + 至少 2 个规模点 scaling
5. 长尾分解：按频率分箱的 F1/recall + “跨父类错误率”




[1]: https://www.nature.com/articles/s41588-024-01993-3?utm_source=chatgpt.com "Consensus prediction of cell type labels in single-cell data ..."
[2]: https://www.sciencedirect.com/science/article/pii/S2666827025002002?utm_source=chatgpt.com "DefMoN: A reproducible framework for theory-grounded ..."

