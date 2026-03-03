下面我按“顶刊方法学审稿人”（Nat Biotech / Nat Methods / Nat Protocols / Briefings in Bioinformatics）的标准来审视，并把“论文级指标补齐”**直接并入你们之前的实验矩阵**，形成一个可执行、可交付的 **paper-grade benchmark plan**。我会尽量保持你们现有定位（reliable multi-level label transfer：校准/拒识/层级一致性/可审计，CPU-first）不变，只补齐审稿最容易卡住的证据链。

---

## 1) 顶刊审稿人视角：目前最可能被抓住的“证据链缺口”

### 缺口 A：你们声称“calibrated confidence + abstention”，但目前指标更像“点状”

* 需要从“有实现”（temperature scaling exists）提升到“有系统证据”（ECE、可靠性曲线、risk–coverage/AURC）。否则审稿人会认为你们的“可靠性”是叙述而不是结论。

### 缺口 B：泛化（跨数据集/shift）证据不足

* 单一 sampled bundle 的结论不足以支撑“框架有效/稳健”。你们报告也明确写了 second dataset 是缺口。
* 顶刊通常至少要：2–3 个独立 dataset pairs + 1 个明确 distribution shift（平台/组织/病种/批次/label-set shift）。

### 缺口 C：CPU-first 的定位需要“系统资源表 + scaling”

* 目前资源指标是片段式（例如某些 variant 的 RSS、训练时间），论文里需要**系统表格**和**随规模变化的曲线**（cells、genes/HVG）。

### 缺口 D：公平对照协议（comparators）要更“论文级”

* 你们 comparator closure 已做，但审稿会追问：每个方法是否按 best practice、是否存在参数不公平、Azimuth fallback 如何表述、资源统计是否完整。

---

## 2) 统一“论文级输出规范”（把补齐内容变成每次实验的强制产物）

从现在开始，每个实验 run 产出统一四类 artifact（便于写论文/补充材料）：

### 2.1 必须的指标 JSON（所有方法/所有实验一致）

**分类性能（每层级 lv1–lv4）：**

* accuracy, macro-F1, balanced accuracy
* per-class F1/recall（至少 lv4） + 按频率分箱（head/mid/tail）

**层级一致性（全路径）：**

* edge consistency
* full-path accuracy / coverage / covered full-path accuracy
* cross-parent error rate（新增：lv4 预测落在错误 lv3 父类的比例）

**校准（calibration）：**

* ECE（至少 lv4；最好 per-level）
* Brier score（可选但强烈建议）

**拒识/选择性预测（selective prediction）：**

* risk–coverage curve（点序列）
* AURC（area under risk-coverage curve）
* 在固定 coverage（如 0.90/0.80）下的 covered accuracy（便于表格比较）

**开放集/OOD（若做 novel-type 设定）：**

* Unknown detection AUROC/AUPRC（构造 known vs novel）

### 2.2 必须的图（paper-ready）

* lv4 reliability diagram（校准前/后）
* risk–coverage 曲线（含 AURC 标注）
* 资源 scaling：cells×time / cells×RSS（CPU与GPU分开）
* confusion 的层级分解图（至少 cross-parent 错误占比）

### 2.3 必须的资源报告（方法学顶刊很看重）

* wall time（train/predict分开）
* peak RSS（CPU）
* peak VRAM（GPU）
* 吞吐：cells/s（predict）
* 若对照方法资源统计缺失，必须在表格里显式标注 “not reported by wrapper” 并给出计划补齐。

### 2.4 必须的协议声明（写进 Methods / Supplement）

* 数据拆分、seed、gene ID canonicalization、counts layer契约、HVG策略、阈值选择规则（例如基于 val set 的 coverage 目标选择阈值）
* 对照方法的参数策略（best practice 依据、是否调参、调参范围与预算）

---

## 3) 把“论文级指标补齐”并入实验矩阵（升级版）

我保留你们原矩阵的 Phase A→F，但每个 Phase 增加“paper-grade必做项”。重点是：**把曲线化（calibration + selective prediction）、跨数据集泛化、资源 scaling**变成矩阵的一部分，而不是额外工作。

---

# Phase 0：数据与评估基座（新增，必须先做）

| Exp ID | 目的                           | 内容                                                                                   | 通过判据（审稿门槛）                       |
| ------ | ---------------------------- | ------------------------------------------------------------------------------------ | -------------------------------- |
| P0     | 数据集套件扩展（泛化闭环）                | 至少 2–3 个 dataset pairs（ref/query），并包含 1 个 shift 场景（如跨组织/跨平台/跨病种/跨批次明显）               | 每个 pair 都能跑通全流程；所有指标/图/资源输出完整    |
| P1     | Novel/Unknown 构造协议           | 构造 open-set：从 reference 移除一组 cell types 或在 query 引入 novel labels（可基于 coarse→fine 子集） | 能计算 Unknown AUROC/AUPRC 与风险-覆盖曲线 |
| P2     | 统一阈值选择规则（避免审稿质疑 cherry-pick） | Unknown/abstention 阈值：基于 val set 选择，使 coverage 达到预设目标（如 0.90）或最小化 AURC               | 阈值选择可复现、可解释，并对所有方法/变体一致          |

> 注：你们报告里已指出缺 second dataset，这是顶刊最致命缺口之一；这一步必须前置。

---

# Phase A：基线门禁 + 论文级曲线（原A阶段增强）

| Exp ID | 目的              | Profile      | 变量                    | 新增“论文级输出”                                                                        | 通过/失败判据                                |
| ------ | --------------- | ------------ | --------------------- | -------------------------------------------------------------------------------- | -------------------------------------- |
| A1     | 基线（CPU-light）门禁 | P0-CPU-light | 无                     | **ECE + reliability diagram**；**risk–coverage + AURC**；资源表（train/predict/RSS/吞吐） | 输出完整；作为后续对比基线（均值±std, seeds=5）         |
| A2     | GPU-fast 门禁     | P1-GPU-fast  | 无                     | 同 A1，另加 VRAM 与 speedup                                                           | GPU 仅作为加速，不允许指标明显退化（>0.5%）             |
| A3     | 校准策略闭环          | P0/P1        | temp scaling {off,on} | 必出“校准前后 reliability diagram + ECE 表”                                             | **ECE显著下降**且 risk–coverage 不变差（AURC不升） |
| A4     | 阈值扫描→阈值选择规则固化   | P0           | threshold grid        | 输出 coverage-accuracy tradeoff 曲线与选择理由                                            | 用 val rule 固定默认阈值（避免后续每次手调）            |

---

# Phase B：决策鲁棒性（SWA/EMA）（原B阶段增强）

| Exp ID | 目的                      | Profile | 变量                   | 新增输出                       | 通过判据（按顶刊/产品两条线）                                |
| ------ | ----------------------- | ------- | -------------------- | -------------------------- | ---------------------------------------------- |
| B1     | SWA                     | P0      | SWA on/off + start比例 | 同 A1 的所有曲线 + std 报告        | **std↓≥25%**（lv4 mF1/bAcc），且 AURC 不变差；推理成本不增加  |
| B2     | EMA备选                   | P0      | EMA on/off + decay   | 同上                         | 若 SWA不通过，用 EMA；判据同 B1                          |
| B3     | SWA 在 shift dataset 上验证 | P0      | 最优 SWA vs baseline   | 在 P0 阶段定义的 shift pair 上完整跑 | **泛化：均值不退化 + std下降**（关键是 shift 场景）             |

---

# Phase C：类不平衡（原C阶段增强：增加长尾分解 + 统一表格）

| Exp ID | 目的               | Profile | 变量                              | 新增输出                                        | 通过判据（更贴合审稿）                                |
| ------ | ---------------- | ------- | ------------------------------- | ------------------------------------------- | ------------------------------------------ |
| C1     | class weights    | P0      | none / inv_freq / effective_num | **按频率分箱(head/mid/tail) 的 F1/recall**；最差K类对比 | tail-bin F1/recall 提升；overall AURC/ECE 不恶化 |
| C2     | focal loss       | P0      | γ/α 网格                          | 同 C1                                        | 同上；额外检查训练稳定性（loss不发散）                      |
| C3     | balanced sampler | P0      | sampler 网格                      | 同 C1                                        | 同上                                         |
| C4     | 选默认不平衡策略（单一机制）   | P0      | pick best                       | 生成“主文表格”：baseline vs best imbalance         | 在 ≥2 个 dataset pairs 上稳定提升（避免单数据集过拟合）      |

---

# Phase D：层级训练（原D阶段增强：新增 cross-parent 错误率）

| Exp ID | 目的                          | Profile | 变量                         | 新增输出                                            | 通过判据                                             |
| ------ | --------------------------- | ------- | -------------------------- | ----------------------------------------------- | ------------------------------------------------ |
| D1     | parent-conditioned decoding | P0      | flat vs parent-conditioned | **cross-parent error rate** + full-path 曲线化（可选） | cross-parent error 明显下降；full-path covered acc 上升 |
| D2     | hierarchy loss              | P0      | λ_hier 网格                  | 同 D1                                            | full-path accuracy/covered acc 提升且 ECE/AURC 不恶化  |
| D3     | shift 场景下的层级鲁棒性             | P0      | best hierarchy vs baseline | 在 shift dataset 上跑                              | shift 中 cross-parent error 不飙升（审稿重点）             |

---

# Phase E：置信度质量（原E阶段增强：把“可靠性定位”写实）

| Exp ID | 目的              | Profile | 变量                    | 新增输出                                               | 通过判据                         |
| ------ | --------------- | ------- | --------------------- | -------------------------------------------------- | ---------------------------- |
| E1     | label smoothing | P0      | ε 网格                  | reliability diagram + ECE/Brier；risk–coverage/AURC | 校准改善且 selective 风险更低（AURC下降） |
| E2     | 校准默认策略（论文可写规则）  | P0      | always vs only-if-val | 输出策略对比 + 失败案例分析                                    | 给出可复现实用默认，不依赖人工调参            |

---

# Phase F：表征增强与蒸馏（原F阶段增强：强调 CPU-first 的落地性）

| Exp ID | 目的                           | Profile | 变量                    | 新增输出                                 | 通过判据                                |
| ------ | ---------------------------- | ------- | --------------------- | ------------------------------------ | ----------------------------------- |
| F1     | 轻量 contrastive/metric 正则     | P0      | λ_sc 网格               | lv4 + tail-bin + ECE/AURC + CPU time | 性能提升≥1% 且 CPU predict 吞吐下降<20%      |
| F2     | GPU teacher → CPU student 蒸馏 | P1→P0   | teacher强模型；student轻模型 | 同上 + 学生模型资源表                         | CPU student 接近 teacher（差<1%）且资源显著更优 |

---

## 4) Comparator（对照方法）论文级补齐：并入每个关键 Phase 的“对照面板”

为了避免审稿人质疑“只对 AtlasMTL 做了曲线/校准/资源表”，建议在关键里程碑（A1、C4、D2、E1、F2）同步跑 **对照方法面板**，至少包括：

* celltypist、scanvi/scANVI、singler、symphony、azimuth（若仍 fallback 必须明确）
  并对每个方法给出：
* 点指标（accuracy/mF1/bAcc）
* selective：如果方法自身不支持 abstention，则用统一方式从其置信度/score 构造（需在协议里写清；否则只报告点指标）
* 资源：train/predict time 与 RSS（缺失必须标注并计划补齐）

> 关键：**协议一致** + **阈值选择一致**（P2）+ **预算一致**（不允许对一个方法调参到极致、另一个用默认）。

---

## 5) 作为审稿人的“最低可接受交付物清单”（你们做完就基本能过方法学审稿门槛）

1. **≥2–3 数据集 pairs + 1 shift 场景**的主表（点指标 + full-path + 资源）
2. lv4 的 **校准证据链**：ECE + reliability diagram（校准前后）
3. **selective prediction**：risk–coverage + AURC，并在固定 coverage 点做比较
4. **长尾分解**：head/mid/tail 的 F1/recall + 最差K类
5. **CPU-first 资源表 + scaling**：cells/genes/HVG 规模变化下的 time/RSS
6. 对照公平性声明：参数策略、阈值策略、预算、Azimuth fallback 的措辞边界

---

## 6) 你们原优化方案是否需要调整（以“顶刊审稿”驱动的矩阵层面结论）

* **合理**：聚焦 MTL classifier（lv4 headroom）、SWA 稳定化、KNN 默认 off。
* **需要调整的是“证据呈现形式”**：把“可靠性”从实现变成曲线与统计；把“单数据集”变成多数据集与 shift；把“资源片段”变成系统表格与 scaling。

---


