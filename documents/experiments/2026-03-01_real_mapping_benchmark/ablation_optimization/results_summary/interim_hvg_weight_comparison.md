# Interim HVG and Weight Comparison

This note is an internal checkpoint for the current AtlasMTL ablation round.
It is not the final benchmark conclusion and should not be cited directly in
the paper as the finalized result.

## Scope

This comparison fixes:

- `input_transform = binary`
- hierarchy-enabled multi-level prediction
- the current sampled reference/query dataset

It compares:

- feature space: `whole`, `hvg3000`, `hvg6000`
- task weights: `uniform` vs `phmap`
- device: `cpu` and `cuda`

## CUDA runs

| config | lv1 | lv2 | lv3 | lv4 | lv4 macro-F1 | full-path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `whole + uniform` | 0.9883 | 0.9390 | 0.8733 | 0.7470 | 0.6234 | 0.7377 |
| `whole + phmap` | 0.9880 | 0.9360 | 0.8717 | 0.7657 | 0.6601 | 0.7550 |
| `hvg3000 + uniform` | 0.9883 | 0.9410 | 0.8773 | 0.7277 | 0.6066 | 0.7213 |
| `hvg3000 + phmap` | 0.9887 | 0.9393 | 0.8787 | 0.7530 | 0.6432 | 0.7440 |
| `hvg6000 + uniform` | 0.9903 | 0.9443 | 0.8813 | 0.7513 | 0.6331 | 0.7427 |
| `hvg6000 + phmap` | 0.9880 | 0.9403 | 0.8807 | 0.7730 | 0.6720 | 0.7617 |

## CPU runs

| config | lv1 | lv2 | lv3 | lv4 | lv4 macro-F1 | full-path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `whole + uniform` | 0.9897 | 0.9413 | 0.8767 | 0.7593 | 0.6693 | 0.7520 |
| `whole + phmap` | 0.9870 | 0.9383 | 0.8723 | 0.7663 | 0.6673 | 0.7603 |
| `hvg3000 + uniform` | 0.9880 | 0.9387 | 0.8727 | 0.7343 | 0.6164 | 0.7267 |
| `hvg3000 + phmap` | 0.9870 | 0.9387 | 0.8750 | 0.7497 | 0.6410 | 0.7423 |
| `hvg6000 + uniform` | 0.9897 | 0.9440 | 0.8760 | 0.7407 | 0.6172 | 0.7340 |
| `hvg6000 + phmap` | 0.9890 | 0.9443 | 0.8767 | 0.7677 | 0.6618 | 0.7600 |

## Current reading

- `phmap` weights improve `lv4` relative to `uniform` in all three feature
  settings.
- `hvg3000` is currently weaker than both `whole` and `hvg6000`.
- `hvg6000 + binary + phmap` is the strongest current tradeoff candidate on
  this dataset.
- `whole` remains the stronger stability baseline and should stay in the next
  search grid.

## Boundary

These results are useful for choosing the next benchmark direction, but they
are still intermediate:

- they come from the current sampled dataset only
- they do not yet establish cross-dataset stability
- they should be treated as internal decision support, not as the final paper
  claim
