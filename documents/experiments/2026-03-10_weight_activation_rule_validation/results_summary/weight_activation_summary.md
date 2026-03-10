# Weight Activation Rule Validation

- rule version: `activation_rule_v1`
- policy target: decide whether a dataset should leave `uniform` task weights

## Decisions

### PHMap_Lung_Full_v43_light

- activate non-uniform weighting: `True`
- recommended schedule name: `needs_candidate_test`
- candidate space: `uniform, mild_lv4, strong_lv4`
- finest macro_f1: `0.541423`
- full_path_accuracy: `0.4570`
- parent_correct_child_wrong_rate: `0.1003`
- decision note: activate non-uniform weighting because baseline shows fine-level difficulty (coarse_to_fine_headroom_gap=0.2704; finest_balanced_accuracy=0.5355; finest_macro_f1=0.5414) and structural tradeoff (parent_correct_child_wrong_rate=0.1003; hotspot_concentration_score=0.6333 with parent_correct_child_wrong_rate=0.1003)

### HLCA_Core

- activate non-uniform weighting: `False`
- recommended schedule name: `uniform`
- candidate space: `uniform`
- finest macro_f1: `0.688732`
- full_path_accuracy: `0.8239`
- parent_correct_child_wrong_rate: `0.0334`
- decision note: keep uniform weighting because no strong structural tradeoff trigger

### mTCA

- activate non-uniform weighting: `False`
- recommended schedule name: `uniform`
- candidate space: `uniform`
- finest macro_f1: `0.848143`
- full_path_accuracy: `0.9334`
- parent_correct_child_wrong_rate: `0.0299`
- decision note: keep uniform weighting because no strong structural tradeoff trigger

## Interpretation

- `PH-Map` is correctly classified as an activation case.
- `HLCA` is correctly classified as a stay-uniform case.
- `mTCA` also behaves as a stay-uniform sanity-check case under the current rule.
- this supports the framework policy claim that non-uniform weighting should be treated as an error-driven, dataset-adaptive option rather than a universal default.
