# atlasmtl Optimization Roadmap

Date: 2026-02-28
Project root: `/home/data/fhz/project/phmap_package/atlasmtl`
Package root: `/home/data/fhz/project/phmap_package/atlasmtl/atlasmtl`

## Goal
Define a priority-ranked optimization roadmap for `atlasmtl` based on the current strengths and known limitations of modern reference mapping methods. The roadmap should help the project improve methodological rigor, benchmark fairness, engineering usability, and long-term extensibility without destabilizing the public API.

## Planning Principles

### Principle 1: Preserve the public contract unless there is a versioned reason to break it
- Keep `build_model()`, `predict()`, `TrainedModel`, and `PredictionResult` stable.
- Preserve the `AnnData in -> AnnData out` workflow.
- Preserve writeback field names under `obs`, `obsm`, and `uns["atlasmtl"]`.
- Treat new outputs as additive by default.

### Principle 2: Prioritize methodological self-consistency before adding model complexity
- Fix confidence, rejection, and benchmark semantics before introducing larger architectures.
- Prefer improvements that make current outputs more trustworthy over improvements that only increase model capacity.

### Principle 3: Optimize for fair comparison against current reference mapping baselines
- Add metrics and controls that allow `atlasmtl` to be compared against nearest-neighbor, anchor-based, and probabilistic methods on equal footing.
- Separate algorithm improvements from changes in preprocessing, thresholds, and data splits.

### Principle 4: Keep implementation modular
- Keep orchestration in `atlasmtl/core/`.
- Keep confidence and KNN logic in `atlasmtl/mapping/`.
- Keep writeback and export logic in `atlasmtl/io/`.
- Keep model serialization and manifest logic in `atlasmtl/models/`.

### Principle 5: Every optimization should have a benchmark or regression check
- No roadmap item is complete until there is at least one unit, integration, or regression test covering it.
- For methodology-facing changes, add benchmark-facing metrics or protocol notes.

## Non-goals for This Roadmap
- Replacing the current `atlasmtl` stack with a full generative model framework.
- Covering all spatial tasks from the literature such as deconvolution or gene imputation in the same release cycle.
- Making large-scale GPU-first training a hard dependency.

## Priority Summary

### P0: Must do next
- Improve confidence calibration and open-set behavior.
- Strengthen domain-shift robustness in the current architecture.
- Make benchmark and evaluation outputs methodologically defensible.

### P1: Should do after P0 stabilizes
- Improve hierarchical consistency and reference representation efficiency.
- Make coordinate prediction more topology-aware.
- Expand benchmark coverage and artifact traceability.

### P2: Nice to have after the core is stable
- Add more advanced adaptation modes and compressed reference strategies.
- Add optional higher-capacity variants and registry-style model packaging.
- Extend toward broader atlas and spatial workflows.

## P0 Roadmap

## P0-1. Confidence Calibration and Reject Option

### Why this is P0
Modern reference mapping methods increasingly emphasize uncertainty, abstention, and calibration. `atlasmtl` already exposes `conf_*`, `margin_*`, `used_knn_*`, and `is_unknown_*`, but the confidence system is still mostly heuristic. This is the highest-leverage place to improve reliability without changing the external workflow.

### Problems to solve
- Raw softmax confidence is not guaranteed to be calibrated.
- `Unknown` currently behaves like thresholded abstention, not a full open-set decision layer.
- KNN correction changes labels, but the project still needs stronger statistical support for when corrected labels should be trusted.

### Actionable principles
- Keep current confidence fields, but make their semantics stronger.
- Prefer calibration layers that can be added post hoc without retraining the full model.
- Keep `Unknown` decision logic interpretable.
- Record all confidence-related controls in `uns["atlasmtl"]`.

### Implementation plan
1. Add calibration support for MTL outputs.
   - Candidate first step: temperature scaling per label head.
   - Store fitted calibration parameters in model metadata.
2. Extend confidence metadata.
   - Record whether calibration was applied.
   - Record calibration method and calibration split.
3. Separate confidence sources explicitly.
   - MTL confidence
   - KNN vote confidence
   - final decision confidence
4. Add an explicit open-set scoring path.
   - Start with distance-to-reference or prototype-distance scoring in latent space.
   - Keep threshold-based fallback for backward compatibility.
5. Add benchmark metrics for calibration and abstention quality.

### Deliverables
- Calibration helper module under `atlasmtl/mapping/` or `atlasmtl/core/`.
- Metadata extension in saved artifacts and `uns["atlasmtl"]`.
- New benchmark metrics for ECE, Brier score, coverage-risk, and reject accuracy.

### Acceptance criteria
- `predict()` can report whether outputs were calibrated.
- `Unknown` decisions can be explained using stored thresholds and scores.
- At least one regression test covers calibrated vs. uncalibrated confidence behavior.
- At least one benchmark report includes calibration metrics.

## P0-2. Domain Shift and Batch Robustness

### Why this is P0
Compared with reference compression and probabilistic latent methods, the current `atlasmtl` stack is relatively lightweight but also more exposed to domain shift. This is one of the clearest weaknesses against stronger modern baselines.

### Problems to solve
- Query distributions may shift because of batch, platform, tissue context, or disease state.
- Coordinate-guided correction works only if the predicted space remains meaningful under shift.
- The model currently relies more on supervised fit than explicit alignment.

### Actionable principles
- Start with lightweight alignment mechanisms before considering heavy generative redesigns.
- Keep the default path simple and opt-in for more advanced adaptation.
- Measure robustness separately from in-domain accuracy.

### Implementation plan
1. Add domain metadata support.
   - Accept optional batch/domain covariates during training.
   - Record them in `train_config`.
2. Add a lightweight alignment loss or adaptation mode.
   - Candidate first step: domain-adversarial head or MMD-style penalty on shared encoder outputs.
   - Keep disabled by default.
3. Add robustness benchmark splits.
   - In-domain
   - cross-batch
   - cross-platform
   - held-out cohort or disease scenario when available
4. Add failure analysis views.
   - Performance by source batch
   - `Unknown` rate by query subset
   - KNN rescue rate under shift

### Deliverables
- Optional adaptation controls in `build_model()`.
- Benchmark split definitions and reporting format.
- Documentation explaining when to enable adaptation.

### Acceptance criteria
- There is a reproducible benchmark showing performance under at least one domain-shift scenario.
- The adaptation mode is optional and does not break the current API contract.
- Resource and runtime overhead is recorded.

## P0-3. Evaluation and Benchmark Protocol Hardening

### Why this is P0
Method improvements are not credible without a benchmark protocol that distinguishes label accuracy, abstention behavior, calibration quality, and compute cost. This is required for paper claims and for fair comparison with the literature.

### Problems to solve
- Current tests check behavior, but not enough methodology-facing metrics.
- The project needs a stable benchmark contract before larger algorithmic changes are added.

### Actionable principles
- Define benchmark metrics before large refactors.
- Evaluate final labels and rejection behavior together.
- Report compute and artifact costs alongside predictive performance.

### Implementation plan
1. Add a minimal evaluation module.
   - Accuracy
   - Macro-F1
   - Balanced accuracy
   - reject rate
   - coverage-risk summaries
2. Add calibration metrics.
   - ECE
   - Brier score
3. Add engineering metrics.
   - train time
   - inference time
   - peak memory
   - artifact size
4. Standardize output report formats.
   - CSV summary table
   - JSON machine-readable metrics
5. Add a benchmark protocol document under `benchmark/` or `documents/protocols/`.

### Deliverables
- `atlasmtl/core/evaluate.py` or equivalent.
- Protocol document for fair comparison.
- Regression/benchmark scaffolding.

### Acceptance criteria
- A single benchmark run can produce predictive, uncertainty, and runtime metrics together.
- Metrics are reproducible from saved artifacts and fixed splits.

## P1 Roadmap

## P1-1. Hierarchy-Aware Prediction Consistency

### Why this is P1
`atlasmtl` already uses multi-head prediction for multiple annotation levels, but the heads are still mostly parallel. This is useful, but it does not fully exploit hierarchy information.

### Problems to solve
- Parent-child inconsistency can appear across levels.
- Per-level confidence may not reflect whether the full path is biologically coherent.

### Actionable principles
- Preserve per-level outputs for backward compatibility.
- Add hierarchy-aware logic as an enhancement, not as a replacement.
- Keep the path from coarse to fine interpretable.

### Implementation plan
1. Add optional hierarchy definitions.
   - Accept label tree or ontology mapping.
2. Add consistency-aware training or inference.
   - Candidate first step: penalize impossible parent-child combinations.
   - Candidate second step: top-down decoding.
3. Add hierarchy-aware metrics.
   - parent-path consistency
   - ontology-aware accuracy

### Deliverables
- Hierarchy config format.
- Optional consistency module.
- Hierarchy-aware benchmark outputs.

### Acceptance criteria
- The model can detect or reduce impossible label paths.
- Full-path accuracy and path consistency are reported.

## P1-2. Better KNN and Reference Representation

### Why this is P1
The current low-confidence-gated KNN correction is a good design choice, but the underlying vote is still simple. Improving the reference side can make the existing rescue path more robust and scalable.

### Problems to solve
- Majority vote is sensitive to class imbalance and sparse neighborhoods.
- Reference storage can grow with atlas size.
- Exact neighbor lookup may become slow or heavy at larger scale.

### Actionable principles
- Keep low-confidence gating as the default.
- Improve KNN internals without changing user-facing defaults unless justified.
- Make reference storage more explicit and benchmarkable.

### Implementation plan
1. Add distance-weighted or margin-aware KNN voting.
2. Add reference prototypes or cluster centroids as an optional mode.
3. Add approximate nearest-neighbor support when scale requires it.
4. Add artifact metadata for reference index type and size.

### Deliverables
- Extended KNN strategies in `atlasmtl/mapping/knn.py`.
- Optional prototype/reference index format in `atlasmtl/models/`.
- Benchmark comparison for exact vs. weighted vs. prototype rescue.

### Acceptance criteria
- At least one improved KNN mode outperforms plain majority vote on difficult-cell subsets or compute efficiency.
- Artifact metadata clearly records which reference representation was used.

## P1-3. Topology-Aware Coordinate Prediction

### Why this is P1
Current coordinate heads solve a practical problem, but simple regression to latent or UMAP coordinates is only a proxy for reference mapping quality. Preserving neighborhood structure is often more important than matching raw coordinates.

### Problems to solve
- Coordinate loss alone may not preserve local topology.
- UMAP coordinates are useful for visualization but are not a stable geometric target by themselves.

### Actionable principles
- Keep current coordinate heads because they are useful and already integrated.
- Add neighborhood-preserving objectives around them instead of replacing them immediately.
- Evaluate topology explicitly.

### Implementation plan
1. Add optional topology-aware losses.
   - neighborhood consistency
   - pairwise distance preservation
   - anchor/prototype attraction
2. Add structure-focused metrics.
   - trustworthiness
   - continuity
   - neighbor overlap
3. Distinguish visualization coordinates from correction coordinates in metadata.

### Deliverables
- Optional topology-aware training controls.
- Additional coordinate and neighborhood metrics.

### Acceptance criteria
- The project can report whether coordinate improvements also improve neighbor preservation.

## P1-4. Stronger Experiment Traceability

### Why this is P1
The project already saves model, metadata, reference data, and manifest. The next step is to make experiments easier to compare across roadmap stages.

### Implementation plan
1. Add richer train/predict config snapshots.
2. Add optional artifact checksums.
3. Add benchmark run manifests and result manifests.
4. Standardize seed recording and split recording.

### Acceptance criteria
- A saved result can be traced back to code settings, artifact bundle, split definition, and benchmark protocol.

## P2 Roadmap

## P2-1. Lightweight Query-Time Adaptation

### Why this is P2
This is a useful extension, but it should come after benchmark and confidence semantics are stable.

### Implementation plan
1. Add optional limited query-time adaptation modes.
2. Keep the default frozen-reference path unchanged.
3. Benchmark whether adaptation helps more than it harms reproducibility.

### Acceptance criteria
- Adaptation is clearly opt-in and benchmarked against the frozen baseline.

## P2-2. Compressed Reference Assets

### Why this is P2
This becomes important at larger scale, but first the project should define what correctness and confidence mean.

### Implementation plan
1. Add compressed reference storage formats.
2. Compare artifact size, load time, and accuracy retention.
3. Keep manifest-based loading stable.

### Acceptance criteria
- The compressed path yields a meaningful reduction in artifact size or load cost without unacceptable accuracy loss.

## P2-3. Optional Higher-Capacity Model Variants

### Why this is P2
Model complexity is not the main bottleneck today. Larger variants should only be introduced after the evaluation protocol is stable.

### Implementation plan
1. Add model presets rather than changing the default model silently.
2. Keep a small default model for usability.
3. Compare higher-capacity variants only within the benchmark protocol.

### Acceptance criteria
- Model presets are explicit, benchmarked, and optional.

## Recommended Execution Order

### Stage A: reliability first
- P0-1 confidence calibration and reject option
- P0-3 evaluation and benchmark protocol hardening

### Stage B: robustness next
- P0-2 domain shift and batch robustness

### Stage C: structural improvements
- P1-1 hierarchy-aware prediction consistency
- P1-2 better KNN and reference representation
- P1-3 topology-aware coordinate prediction

### Stage D: scaling and advanced options
- P1-4 stronger experiment traceability
- P2-1 lightweight query-time adaptation
- P2-2 compressed reference assets
- P2-3 optional higher-capacity variants

## Release Planning Suggestion

### Near-term release
- Focus on P0 only.
- Goal: make `atlasmtl` trustworthy and benchmarkable.

### Mid-term release
- Add P1 items that improve structure and scale.
- Goal: strengthen the method without changing the core user workflow.

### Long-term release
- Add P2 items for larger atlas scenarios and optional advanced modes.
- Goal: broaden applicability while keeping the default path simple.

## Risks and Controls

### Risk 1: roadmap items inflate API complexity
- Control: keep new capabilities optional and additive.

### Risk 2: confidence changes alter downstream expectations
- Control: version defaults explicitly and record them in metadata.

### Risk 3: adaptation features make comparisons unfair
- Control: always benchmark against the frozen-reference baseline.

### Risk 4: engineering work outruns evaluation rigor
- Control: no optimization counts as complete without benchmark-facing evidence.

## Exit Criteria for the Roadmap
The roadmap should be considered successful when:
- `atlasmtl` has a calibrated and documented confidence story.
- `Unknown` behavior is benchmarked, not just thresholded.
- domain-shift scenarios are part of the benchmark protocol.
- hierarchy, KNN rescue, and coordinate mapping are each evaluated with fit-for-purpose metrics.
- artifact and experiment traceability are strong enough to support manuscript claims and reproducible comparisons.
