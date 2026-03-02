# AtlasMTL Ablation Discussion

This ablation round answers the main questions from the follow-up plan with a
cleaner separation between accuracy effects and resource effects.

## 1. Task-weighting matters

Using PH-Map-style weights improved the fine-grained target level compared with
uniform weighting.

Observed pattern:

- average `anno_lv4` accuracy increased from `0.6964` to `0.7205`
- the best-performing runs all used `phmap` weights

Current interpretation:

- the fine level should not be treated as just one more symmetric head
- giving more weight to `anno_lv4` is justified for this mapping task because
  the formal benchmark target is the most detailed retained label level

What remains open:

- the exact gain may still be dataset-dependent
- this should be validated on additional datasets before turning one weight
  schedule into a universal default

## 2. Binary encoding is not only a resource trick

`binary` substantially outperformed `float` on this dataset.

Observed pattern:

- average `anno_lv4` accuracy: `0.7530` vs `0.6639`
- best overall runs all used `binary`

Current interpretation:

- binary encoding is not just reducing compute; it is materially changing the
  inductive bias of the model
- on this sampled benchmark, that inductive bias improves label transfer at the
  fine level

Resource interpretation:

- training time was similar on average, but binary kept the best accuracy
- binary therefore improves the accuracy/resource tradeoff rather than merely
  reducing runtime

## 3. Whole matrix is not automatically better

The mean accuracy over all `whole` runs was slightly above the combined HVG
runs, but the best single run came from `hvg6000`.

Observed pattern:

- average `anno_lv4` accuracy:
  - `whole`: `0.7121`
  - combined `hvg`: `0.7066`
- best run:
  - `cuda + hvg6000 + binary + phmap`: `0.7730`

Resource pattern:

- `whole` training was materially heavier than HVG
- `hvg6000` reduced peak RSS to about `3.42 GB` while keeping top accuracy

Current interpretation:

- `whole` remains a meaningful ablation and can still be competitive
- `hvg6000` is currently the best practical configuration because it preserves
  accuracy while reducing memory and training time
- `hvg3000` appears too aggressive for this dataset

## 4. CPU efficiency is real, GPU speedup is also real

AtlasMTL does not require GPU to be competitive on this sampled dataset.

Observed pattern:

- average accuracy was nearly identical:
  - `cpu`: `0.7081`
  - `cuda`: `0.7087`
- GPU reduced average train time from `10.7532 s` to `3.4301 s`

Current interpretation:

- AtlasMTL is small enough and structured enough that CPU execution remains
  viable for ordinary users
- GPU remains valuable for throughput, repeated tuning, and larger runs
- this is a practical advantage, not a contradiction of the model being a
  deep-learning method

## 5. Relation to the earlier PH-Map comparison

This ablation narrows the explanation gap for the earlier AtlasMTL vs PH-Map
difference.

Supported conclusions from this round:

- part of the earlier accuracy gap was due to task-weight mismatch
- part of the earlier accuracy gap was due to using `float`-style inputs rather
  than the stronger binary configuration
- feature-space choice also matters, and `hvg6000` is currently stronger than
  the previously tested settings

What this round does not yet prove:

- whether the same ranking holds across multiple datasets
- whether the final production default should always be `hvg6000 + binary + phmap`

## Current recommendation

For the next formal AtlasMTL benchmark round, the strongest candidate default
is:

- `feature_space = hvg`
- `n_top_genes = 6000`
- `input_transform = binary`
- `task_weights = [0.3, 0.8, 1.5, 2.0]`

This should be treated as the next benchmark default candidate, not yet as an
unchangeable project-wide invariant.

The key decision rule is also now explicit:

- the benchmark target is not the numerically highest accuracy alone
- the preferred choice is the best accuracy-resource balance
- if multiple settings are close in quality, the lower-resource setting should
  be preferred as the operational recommendation

Under that rule, `hvg6000 + binary + phmap` is currently favored because it
beats the current `whole` baseline at the top end while materially reducing
memory and training cost.
