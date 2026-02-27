# atlasmtl Architecture

## Core path
1. AnnData input
2. Shared encoder + multi-task heads + coordinate heads
3. Confidence estimation per level
4. Low-confidence-gated KNN correction
5. Unknown assignment with dual thresholds
6. AnnData output (obs, obsm, uns)

## Default confidence policy
- high confidence: max_prob >= 0.7 and margin >= 0.2
- low confidence: max_prob < 0.4 -> Unknown
- middle band: optional KNN correction
