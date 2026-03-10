# AtlasMTL Methods Draft v1

## Method Overview

AtlasMTL is a framework for multi-level single-cell reference mapping. Given a
labeled reference dataset and an unlabeled query dataset, both represented as
`AnnData` objects, AtlasMTL transfers cell annotations from the reference to
the query while preserving the multi-level structure of the annotation task.
The method is intended for use cases in which the practical objective is not
only to maximize label accuracy, but also to produce predictions that are
reliable, hierarchically coherent, and operationally interpretable under
uncertainty.

In this study, AtlasMTL is framed as a reference-mapping and label-transfer
tool rather than as a method primarily optimized for generic integrated
embedding benchmarks. Its central outputs are multi-level predicted labels,
probability-derived confidence signals, abstention-aware predictions, and
traceable metadata that can be audited after model training and deployment.
This framing matches the intended role of AtlasMTL as a practical annotation
workflow for cross-study and cross-cohort label transfer, where uncertainty
handling and reproducibility are as important as nominal classification
accuracy.

## Framework Design And Learning Objective

AtlasMTL uses a hard parameter-sharing multi-task neural architecture. Let
\(x \in \mathbb{R}^G\) denote the aligned input vector for one cell after
preprocessing over a fixed feature panel of size \(G\). A shared encoder
\(f_{\theta}\) maps the input into a latent representation
\(z = f_{\theta}(x)\). For each annotation level \(\ell \in \{1, \dots, L\}\),
AtlasMTL defines a task-specific classification head \(g_{\ell}\) that produces
logits \(h_{\ell} = g_{\ell}(z)\), followed by a softmax distribution
\(p_{\ell} = \mathrm{softmax}(h_{\ell})\). Thus, all annotation levels share a
common backbone representation while preserving level-specific output spaces.

The core training objective is a weighted sum of per-level cross-entropy
losses,
\[
\mathcal{L}_{\mathrm{cls}} = \sum_{\ell=1}^{L} \alpha_{\ell}
\mathrm{CE}\left(y_{\ell}, p_{\ell}\right),
\]
where \(y_{\ell}\) is the ground-truth label at level \(\ell\), \(p_{\ell}\) is
the predicted categorical distribution for that level, and \(\alpha_{\ell}\)
is the task weight assigned to that level. This formulation allows AtlasMTL to
learn coarse and fine annotation tasks jointly instead of training independent
models for each level. In the benchmark-facing core path, this weighted
classification objective is the primary training target. The framework-level
weighted formulation is stable, but the best benchmark-default non-uniform
task-weight schedule for the multi-level setting should not yet be treated as
fully fixed across all datasets.

AtlasMTL also supports optional auxiliary components for extended
configurations, including coordinate heads and lightweight geometric or
domain-alignment regularization. These extensions are part of the broader
framework but are not required to describe the stable benchmark-facing core of
the method.

## Data Contract And Preprocessing

AtlasMTL uses `AnnData` as the primary input and output container throughout
training and prediction. Formal preprocessing assumes that raw expression
counts are available in `adata.layers["counts"]`, which serves as the stable
count-layer contract for benchmarked and reproducible runs. If the working
expression matrix differs from the raw-count representation, the counts layer
remains the reference point for feature selection and related preprocessing
steps.

The canonical internal gene namespace is versionless Ensembl identifiers. In
formal training and prediction workflows, reference and query datasets are
expected to be aligned in this namespace rather than by symbol alone. If an
input dataset is provided with gene symbols only, an explicit preprocessing
step is required to map features into the canonical identifier space and to
record the mapping resource and species context used for that conversion. This
constraint improves reproducibility and avoids silent ambiguity in
cross-dataset feature alignment.

Feature alignment is performed relative to the training feature panel. During
model construction, AtlasMTL derives a reference-side feature set that defines
the input space of the trained model. During prediction, query features are
reordered to match this panel, and missing features are padded so that the
query matrix conforms to the model input contract. This alignment strategy
allows the framework to consume heterogeneous datasets while preserving a
stable feature order across training, serialization, and inference.

The current default training input transform is binary encoding of gene
detection (`input_transform="binary"`). Under this default, the model is
trained on a binarized feature matrix derived from the aligned input panel
rather than on a continuously scaled expression matrix. This choice was fixed
through the benchmark-facing parameter-lock process and retained in the
subsequent default-optimization round.

## Training Procedure And Software Defaults

AtlasMTL model training begins from preprocessed reference data paired with one
or more annotation columns defining the target prediction levels. The model is
optimized jointly across levels rather than through independent per-level
training runs, allowing the shared backbone to learn a common representation
that supports both coarse and fine annotation tasks while preserving
level-specific decoding at the output heads. Model selection uses
validation-aware training with early stopping to reduce overfitting and to
preserve a lightweight, reproducible training workflow.

The benchmark-facing training skeleton was fixed in the `2026-03-07`
parameter-lock round, which established a stable training envelope before any
optimizer-default refinement was considered. A later low-cost optimization
round evaluated only a narrow optimizer-level search space and promoted a new
software default without reopening the larger architectural search space. The
current AtlasMTL software default is therefore defined by the following
training path: `input_transform="binary"`, `optimizer_name="adamw"`,
`weight_decay=5e-5`, `scheduler_name=None`, and
`reference_storage="external"`.

The promoted optimizer default was supported by the completed optimization
dossier, primarily through non-degraded Stage B GPU confirmation. However, the
later fifth-round formal refresh did not provide sufficient evidence to replace
the retained third-wave manuscript-grade AtlasMTL rows in the formal
comparison tables. Accordingly, the current software default and the retained
manuscript-grade formal benchmark rows should be treated as separate records:
the software default has been updated to `AdamW + weight_decay=5e-5`, whereas
the paper-grade formal comparison tables continue to use the retained
third-wave AtlasMTL baseline rows.

## Prediction, Confidence, Abstention, Hierarchy, And Optional KNN Refinement

During prediction, a query dataset is first aligned to the training feature
panel and then passed through the trained AtlasMTL model to obtain one
probability distribution per annotation level. For each level, the predicted
label is obtained by decoding the top-probability class after softmax. AtlasMTL
retains the per-level probability outputs long enough to derive
confidence-related quantities that are exposed in the prediction results and in
benchmark summaries.

AtlasMTL uses two simple but operationally useful confidence summaries for each
prediction: the maximum class probability and the top-two probability margin.
If \(p_{\ell}\) is the predicted distribution for level \(\ell\), then the
confidence score is
\[
c_{\ell} = \max_k p_{\ell,k},
\]
and the margin is
\[
m_{\ell} = p_{\ell,(1)} - p_{\ell,(2)},
\]
where \(p_{\ell,(1)}\) and \(p_{\ell,(2)}\) denote the largest and
second-largest class probabilities at that level. Low-confidence predictions
can then be identified through thresholding on \(c_{\ell}\) and \(m_{\ell}\),
which forms the basis of the built-in low-confidence routing and abstention
logic.

Abstention is treated as part of the method contract rather than as an
afterthought. In the default closed-loop logic, predictions below a lower
confidence threshold can be marked as `Unknown`, and low-confidence cases can
optionally be routed to an auxiliary post-prediction refinement step rather
than being accepted directly as final labels. For hierarchical annotation
tasks, AtlasMTL can enforce parent-child consistency after per-level
predictions are generated, making hierarchy consistency a first-class property
of prediction quality. In the current benchmark path, abstention and
hierarchy-aware consistency enforcement are part of the active method contract,
whereas KNN-assisted refinement should be treated as an optional framework
extension rather than as a core benchmark-default component.

## Model Artifacts And Reproducibility

AtlasMTL is designed to export portable trained-model artifacts rather than
only in-memory training results. The trained bundle stores model weights,
metadata, reference linkage, and a manifest describing how the model should be
reloaded. This artifact design makes model reuse explicit and improves
reproducibility by preserving the feature panel, label metadata, configuration
choices, and runtime context that define the trained model.

As a result, AtlasMTL should be described not only as a neural architecture but
also as a reproducible model-building and model-deployment workflow. This
artifact contract is also relevant to the benchmark design because it allows
later optimization and formal-refresh rounds to distinguish clearly between
current software defaults and retained manuscript-grade benchmark records.

## Benchmark Design And Evaluation

The AtlasMTL benchmark is framed around the task of `sc -> sc reference
mapping`, with emphasis on multi-level annotation quality and reliability under
uncertainty. Comparator selection is therefore restricted to methods that act
in the same task family, rather than expanding the main benchmark to spatial
deconvolution, gene imputation, or generic integrated embedding objectives.

Primary evaluation metrics focus on label quality and uncertainty-aware
behavior. These include overall accuracy, macro-F1, balanced accuracy,
coverage, reject rate, covered accuracy, risk, expected calibration error,
Brier score, and the area under the risk-coverage curve. AtlasMTL-specific
behavior metrics such as Unknown rate and hierarchy path consistency are
treated as method-relevant supporting analyses because they reflect whether the
framework behaves reliably in difficult or ambiguous cases. Optional KNN
rescue-related metrics should be treated as supplementary framework behaviors
rather than as part of the current main evidence path unless a later benchmark
round restores them to the active protocol.

Resource benchmarking is part of the formal evaluation contract. Training time,
prediction time, peak process RSS, and peak GPU memory are recorded so that the
practical cost of model construction and deployment can be assessed alongside
annotation quality. CPU and GPU tracks are reported separately, and fairness
metadata are recorded to distinguish non-degraded runs from runs executed under
restricted conditions.

Finally, the benchmark record must distinguish between software-default updates
and manuscript-grade formal comparison rows. The low-cost optimization round
was sufficient to promote `AdamW + weight_decay=5e-5` as the current software
default. However, the fifth-round AtlasMTL-only formal refresh did not justify
replacing the retained third-wave manuscript-grade AtlasMTL baseline rows.
Accordingly, the present implementation should be described using the current
software default, whereas formal comparison tables should continue to use the
retained third-wave AtlasMTL rows unless a later completed refresh provides
stronger replacement evidence. The completed sixth-round multi-level benchmark
should likewise be interpreted as a capability study for hierarchy-aware
annotation rather than as a direct replacement for the retained single-level
formal comparison rows.
