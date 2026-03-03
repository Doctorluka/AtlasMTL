# Manifest templates and naming

This directory stores benchmark manifests that are ready to feed into
`benchmark/pipelines/run_benchmark.py`.

Use scenario-specific manifests. Do not try to force every dataset through one
universal YAML file.

## Scenario classes

- `reference_heldout`
  - formal quantitative benchmark from a reference dataset with accepted truth
    labels
- `external_query_validation`
  - deployment-style transfer into an unlabeled or not-yet-approved query
    dataset

## Naming rule

Store manifests under:

- `manifests/reference_heldout/`
- `manifests/external_query_validation/`
- `manifests/templates/`

Recommended file name:

- `<dataset_id>__<target_label>__<split_name>.yaml`

Examples:

- `PHMap_Lung_Full_v43_light__anno_lv4__group_split_v1.yaml`
- `HLCA_Core__ann_level_5__gse302339_marker_review_v1.yaml`

## Templates

Start from:

- `manifests/templates/reference_heldout_template.yaml`
- `manifests/templates/external_query_validation_template.yaml`

Then fill in:

- dataset paths
- target labels
- split metadata
- counts semantics
- preprocessing metadata
- method-specific settings only when they differ from defaults
