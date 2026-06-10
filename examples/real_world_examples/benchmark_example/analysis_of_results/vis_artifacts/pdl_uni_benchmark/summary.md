# Benchmark Result Analysis: pdl_ucr_full

- Task type: `ts_classification`
- Metric: `accuracy`
- Metric direction: `higher`

## Coverage

| source_label | coverage_unit | expected_dataset_count | observed_dataset_count | coverage_pct | missing_dataset_count | missing_datasets | status  |
| ------------ | ------------- | ---------------------- | ---------------------- | ------------ | --------------------- | ---------------- | ------- |
| pdl_ucr_full | dataset_name  | 225                    | 111                    | 49.3333      | 0                     |                  | partial |

## Model Diagnostics

- Diagnostic rows: `448`
- Datasets with diagnostics: `111`
- Models with diagnostics: `4`

## Mean Rank

| model_name                                            | mean_rank | dataset_count |
| ----------------------------------------------------- | --------- | ------------- |
| KernelEnsembleClassifier_shapelet_motif_rbf           | 1.88739   | 111           |
| KernelEnsembleClassifier_embedding_nystrom            | 2.59459   | 111           |
| KernelEnsembleClassifier_score_baseline_summary       | 2.7027    | 111           |
| KernelEnsembleClassifier_adaptive_all_non_topological | 2.81532   | 111           |

## Top-K Summary

| model_name                                            | top_1 | top_3 | top_5 | top_half | dataset_count |
| ----------------------------------------------------- | ----- | ----- | ----- | -------- | ------------- |
| KernelEnsembleClassifier_shapelet_motif_rbf           | 61    | 104   | 111   | 90       | 111           |
| KernelEnsembleClassifier_embedding_nystrom            | 29    | 90    | 111   | 62       | 111           |
| KernelEnsembleClassifier_adaptive_all_non_topological | 27    | 79    | 111   | 55       | 111           |
| KernelEnsembleClassifier_score_baseline_summary       | 23    | 85    | 111   | 55       | 111           |

## Target Delta

No target delta rows.