# ProjectSVR vs atlasmtl

This note records how `ProjectSVR` should be used as a paper-design reference
for `atlasmtl`.

## Comparison table

| Dimension | `ProjectSVR` | `atlasmtl` | Implication for atlasmtl |
|---|---|---|---|
| Core positioning | reference embedding projection + downstream label transfer | reliable multi-level label transfer | atlasmtl should not be written as a pure projection paper |
| Main question | can query cells be projected into the reference space well | can query cells receive accurate, reliable, abstention-aware labels | atlasmtl should keep labels and reliability as the main story |
| Main model | ensemble SVR from gene-set scores to embeddings | shared encoder + multi-task label heads + optional coordinate heads | atlasmtl has a richer prediction contract |
| Primary output | projected embedding, then k-NN label transfer | labels + confidence + margin + optional coordinates + Unknown | atlasmtl output is broader and more decision-oriented |
| Label structure | mostly single-level | native multi-level | this is one of atlasmtl’s strongest differentiators |
| Continuous state handling | strong, because projection is central | present but secondary | atlasmtl can report this, but should not lead with it |
| Uncertainty modeling | minimal | calibration / reject / open-set / Unknown | atlasmtl should emphasize this in the paper |
| KNN role | main label-transfer mechanism | rescue / correction mechanism | atlasmtl KNN is supportive, not the main classifier |
| Hierarchy support | absent | present | hierarchy consistency is atlasmtl-specific value |
| Domain robustness | shown through cross-condition applications | explicit protocol direction with `domain_key` | atlasmtl still needs stronger formal domain protocol |
| Benchmark main metrics | Accuracy, ARI, runtime | Accuracy, Macro-F1, Balanced accuracy, Coverage, Reject rate, ECE, Brier, AURC | atlasmtl should not collapse back to only Accuracy + ARI |
| Geometry analysis | strong emphasis | auxiliary diagnostics | atlasmtl should keep geometry as supporting evidence |
| Case studies | strong | not yet fully formalized | atlasmtl should learn from this structure |
| Reproducibility packaging | good | stronger manifest/checksum/run-manifest system | atlasmtl has engineering advantages worth highlighting |

## What atlasmtl should borrow

- multi-dataset benchmark organization
- runtime comparison
- biological case-study structure
- reference-mapping style presentation of experiments

## What atlasmtl should not borrow

- treating projection geometry as the primary success criterion
- using only `Accuracy + ARI` as the core evaluation
- weakening the uncertainty / abstention story
- presenting the method as embedding-first rather than label-first

## Practical conclusion

`ProjectSVR` is a useful experiment-design reference, but not a direct metric
template for atlasmtl. atlasmtl should borrow the experimental organization and
case-study logic, while preserving its own main claim around reliable label
transfer.
