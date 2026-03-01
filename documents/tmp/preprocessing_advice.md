# Reference Mapping / Label Transfer 预处理方案（多组织通用版）
**目标**：为 label transfer / label prediction 构建稳健的 reference–query 共享表示空间与特征集，降低“不可迁移的样本特异性/质量轴”对邻域结构与分类器的干扰。  
**核心原则**：**不默认硬删基因**；默认采用 **exclude-from-HVG / downweight** 来抑制 nuisance 模块主导表示；所有决策可审计、可消融、可配置。

---

## 0. 输入与输出
### 输入
- `adata_ref`: reference AnnData（带 label：`obs[label_key]`）
- `adata_qry`: query AnnData（无 label 或待预测 label）
- `batch_key_ref`（可选）：ref 中批次/样本字段
- `batch_key_qry`（可选）：qry 中批次/样本字段（仅用于报告，不用于选 HVG）

### 输出
- `ref_proc`, `qry_proc`：处理后的 AnnData（共享相同基因子集与同构预处理）
- `feature_genes`: 用于表示学习/邻域/分类的基因列表（HVG 子集）
- `prep_report`: 可审计报告（JSON/表格/图均可）：
  - HVG 模块占比（处理前后）
  - PC1/PC2 loading 的模块富集（处理前后）
  - ref vs qry 的模块 score 分布差异（KS/EMD 等）
  - 被 exclude/drop/downweight 的基因列表与规则

---

## 1. 基因模块定义（nuisance modules）
> 用“模块”驱动规则，而不是固定黑名单；默认仅影响 HVG/表示，不从矩阵删除。

### 1.1 Regex-based modules（跨组织通用）
- **MT**：`^MT-`（人）/`^mt-`（鼠）
- **Ribo**：`^RPL|^RPS`
- **HB**：`^HBA|^HBB`
- **IG**：`^IGH|^IGK|^IGL`
- **TCR**：`^TRAV|^TRBV|^TRAJ|^TRBJ|^TRAC|^TRBC`

### 1.2 List-based modules（可选，建议提供内置表）
- **Stress/Immediate early**：`FOS, JUN, HSPA*, DUSP*, ATF3, EGR* ...`
- **Cell-cycle**：S/G2M 基因表（Scanpy/Seurat 常用表）

> 默认启用：MT/Ribo/HB/IG/TCR  
> Stress/Cell-cycle 默认不硬处理：仅报告与触发 warning（除非用户选择启用）。

---

## 2. 总体流程概览（强制对齐：ref 主导）
**关键点**：**HVG 在 reference 上选定**，query 仅对齐到同一套基因；不要在 query 上重新选 HVG。

1) QC / 基础过滤（两者一致）  
2) 归一化与 log1p（两者一致）  
3) 在 ref 上选 HVG（`seurat_v3` + 可选 `batch_key_ref`），同时对 nuisance modules 进行 **exclude-from-HVG**  
4) 得到 `feature_genes`，ref/qry 同时子集化到该基因集  
5) `downweight`（可选）用于表示学习阶段：对 nuisance 模块在构建距离/embedding时降权（不改原矩阵）  
6) 训练/映射：基于 `feature_genes` 的 PCA/邻域/分类器进行 label transfer  
7) 输出报告与消融接口

---

## 3. Step-by-step 可执行细节

### 3.1 QC 与基础过滤（建议“温和”，避免过拟合到某组织）
**推荐默认（UMI数据）**：
- 过滤细胞：`min_genes >= 200`（可配）
- 过滤基因：`min_cells >= 3`（可配）
- 不做过强的 `pct_mt` 硬阈值（跨组织差异太大），只做报告 + 可选阈值

**必做指标**
- `n_genes_by_counts`
- `total_counts`
- `pct_counts_mt`（用 MT module 计算）
-（可选）`pct_counts_ribo`, `pct_counts_hb`, `pct_counts_ig_tcr`

> 若你要支持多平台（Smart-seq vs UMI），把阈值作为 profile/preset，而不是写死。

---

### 3.2 归一化与 log1p（ref/qry 一致）
建议采用 Scanpy 常用的：
- `sc.pp.normalize_total(target_sum=1e4)`
- `sc.pp.log1p()`

> **不要默认 regress out**（侵入性高、跨组织易伤信号）。作为可选 ablation：仅对 `log1p_total_counts` 或 `pct_mt` 提供触发式回归（默认关闭）。

---

### 3.3 在 reference 上选 HVG（Seurat v3 风格）
**默认推荐**：`flavor="seurat_v3"`，`n_top_genes=2000~4000`（建议默认 3000）

- 若 `batch_key_ref` 存在：必须传入 `batch_key_ref`（降低 ref 内部批次污染）
- 得到 `ref.var["highly_variable"]`

---

### 3.4 nuisance-aware feature selection（核心：exclude-from-HVG）
**目标**：让 HVG 尽量代表“可迁移的细胞身份/状态差异”，而不是质量轴或克隆轴。

#### 3.4.1 默认规则（Level 1：exclude-from-HVG，不删表达）
- 先按 `seurat_v3` 得到候选 HVGs（例如 top 5000）
- 将其中属于以下模块的基因移出 HVG 列表：
  - MT / Ribo / HB / IG / TCR

#### 3.4.2 例外策略（推荐支持的两种模式）
- **strict_exclude（默认）**：无条件排除上述模块基因进入 HVG
- **adaptive_exclude（更稳健，适合写进论文）**：
  - 若模块基因在 HVG 候选中的占比 > `p_thresh`（如 5%）或在 PC1 loadings 前 `k` 名富集显著，则排除
  - 否则仅标记 warning，不排除

> 对 label transfer 来说，strict_exclude 通常更不容易被 IG/TCR（PBMC）或 HB（污染）击穿。

---

### 3.5 ref/qry 对齐到同一 feature space
- `feature_genes = final_HVG_list`
- `ref_proc = ref[:, feature_genes].copy()`
- `qry_proc = qry[:, feature_genes].copy()`
- 确保基因顺序完全一致（相同 var_names 顺序）

---

### 3.6 Downweight（可选）：在“表示/距离层”降低 nuisance 影响
> 不改变表达矩阵，仅改变用于 PCA/距离计算的输入权重。这样可逆、可审计。

**两种实现思路**
1) **gene-level weighting**：对模块内基因设置权重 `w < 1`（如 0.2–0.5）  
2) **module-orthogonalization**（更强）：计算模块 score 向量，在 PCA/embedding 前做正交化（侵入性更强，默认不启用）

**默认建议**
- 只对极端场景启用 downweight（例如 PBMC 的 IG/TCR 即便 exclude-from-HVG 后仍影响邻域，或你允许其进入 features 时）

---

## 4. 报告与自动诊断（强烈建议作为框架默认输出）
### 4.1 必备报告项
1) `feature_genes` 的模块组成占比（MT/Ribo/HB/IG/TCR）
2) ref vs qry 的模块 score 分布差异（KS/EMD）
3) ref PCA 前两主成分与 `total_counts/pct_mt` 的相关性
4) mapping 风险提示：
   - “IG/TCR dominates HVG candidates” → 建议 strict_exclude
   - “HB module high in query only” → 建议检查红细胞污染/ambient RNA
   - “MT module shift in query” → 建议 QC 或触发式回归（可选）

### 4.2 最小可复现的消融接口
- `nuisance_mode = {none, strict_exclude, adaptive_exclude, strict_exclude+downweight}`
- 输出每种模式下的 label transfer 指标（见下）

---

## 5. 面向 label transfer 的评估建议（作为算法开发闭环）
> 你做的是算法/框架，评估要覆盖“准确率”与“校准/拒识”能力。

### 5.1 基础指标
- Accuracy / macro-F1（按 cell type）
- balanced accuracy（类别不平衡时）
- top-k accuracy（若输出概率分布）

### 5.2 邻域一致性与可迁移性（建议至少一个）
- kNN label agreement（query 在 ref 邻域的 label 一致性）
- LISI/kBET（如你同时做整合）
- entropy of predicted labels（过度自信/塌缩预警）

### 5.3 置信度与拒识（强烈建议）
- 设置 `unknown`：低置信度/高 entropy 的 query 不强行赋值
- 报告 calibration：ECE 或 reliability curve（可选）

---

## 6. 默认参数建议（可直接作为你的方法默认值）
- `n_top_genes = 3000`
- `hvg_flavor = "seurat_v3"`
- `batch_key_ref = "batch"`（若存在）
- `nuisance_mode = "strict_exclude"`（默认）
- `modules_enabled = ["MT","Ribo","HB","IG","TCR"]`
- `drop_genes = False`（默认永不硬删）
- `downweight = False`（默认关闭；作为高级选项）

---

## 7. 设计取舍的说明（写进 Methods 的关键句）
- 我们保留所有基因用于解释与 marker 验证，但在构建用于 label transfer 的表示空间时，通过对已知 nuisance 模块进行 variable feature exclusion（并可选地进行权重抑制），避免质量轴/克隆轴主导邻域结构，从而提升跨组织与跨队列的可迁移性与稳健性。