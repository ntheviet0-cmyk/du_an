# GraphSAGE Error Audit

- Evaluation threshold: `0.80`
- Confusion matrix: `[[235, 12], [25, 22]]`
- False positives: `12`
- False negatives: `25`
- Local SHAP explanations already persisted: `2/37`

## Interpretation rule

`observed_profile` is descriptive context, not a model attribution and not a causal reason for attrition.
Only rows marked `local_shap_verified_inductive` have a persisted local SHAP explanation. Those SHAP
values come from the inductive wrapper and can differ slightly from the transductive evaluation probability.
Neighbor labels are joined only after prediction for audit; labels were not used to construct cosine edges.

## Outputs

- `graph_error_cases.csv`: all 37 errors
- `graph_false_positives.csv`: 12 false positives
- `graph_false_negatives.csv`: 25 false negatives
- `graph_error_neighbors.csv`: top-15 cosine neighbors for each error
- `graph_error_overview.png`: confusion matrix, probabilities and confidence margins
- `graph_error_feature_map.png`: observed feature map
- `graph_error_neighbors.png`: probability versus top-15 neighborhood labels
- `graph_false_positives.png`: false-positive casebook
- `graph_false_negatives_01.png`, `graph_false_negatives_02.png`: false-negative casebook
