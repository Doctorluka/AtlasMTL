# V2 Weighted GPU Decision Note

Decision rule:

- at least `3/4` datasets have non-negative mean `delta_macro_f1`
- `PHMap_Lung_Full_v43_light` mean `delta_macro_f1` is positive
- no dataset mean `delta_full_path_accuracy` falls below `-0.01`

Current decision: `keep_v1_as_primary_sixth_round_track`

v1 remains frozen regardless of this outcome.
