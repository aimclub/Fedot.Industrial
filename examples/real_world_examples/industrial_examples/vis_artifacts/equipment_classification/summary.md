# Benchmark Result Analysis: equipment_classification_current_api_preview

- Task type: `ts_classification`
- Metric: `accuracy`
- Metric direction: `higher`

## Mean Rank

| model_name                                            | mean_rank | dataset_count |
| ----------------------------------------------------- | --------- | ------------- |
| KernelEnsembleClassifier_score_baseline_summary       | 1         | 2             |
| KernelEnsembleClassifier_adaptive_all_non_topological | 2         | 2             |
| KernelEnsembleClassifier_shapelet_motif_rbf           | 3         | 2             |
| KernelEnsembleClassifier_embedding_nystrom            | 4         | 2             |
| KernelEnsembleClassifier                              | 5         | 2             |
| PDLClassifier                                         | 6         | 2             |
| NearestCentroid                                       | 7         | 2             |

## Top-K Summary

| model_name                                            | top_1 | top_3 | top_5 | top_half | dataset_count |
| ----------------------------------------------------- | ----- | ----- | ----- | -------- | ------------- |
| KernelEnsembleClassifier_score_baseline_summary       | 2     | 2     | 2     | 2        | 2             |
| KernelEnsembleClassifier_adaptive_all_non_topological | 0     | 2     | 2     | 2        | 2             |
| KernelEnsembleClassifier_shapelet_motif_rbf           | 0     | 2     | 2     | 2        | 2             |
| KernelEnsembleClassifier                              | 0     | 0     | 2     | 0        | 2             |
| KernelEnsembleClassifier_embedding_nystrom            | 0     | 0     | 2     | 0        | 2             |
| NearestCentroid                                       | 0     | 0     | 0     | 0        | 2             |
| PDLClassifier                                         | 0     | 0     | 0     | 0        | 2             |

## Target Delta

| dataset_name                       | target_model                                    | target_metric | best_reference_model                                  | best_reference_metric | improvement | relative_improvement_pct |
| ---------------------------------- | ----------------------------------------------- | ------------- | ----------------------------------------------------- | --------------------- | ----------- | ------------------------ |
| equipment_classification_preview_1 | KernelEnsembleClassifier_score_baseline_summary | 0.9           | KernelEnsembleClassifier_adaptive_all_non_topological | 0.86                  | 0.04        | 4.65116                  |
| equipment_classification_preview_2 | KernelEnsembleClassifier_score_baseline_summary | 0.88          | KernelEnsembleClassifier_adaptive_all_non_topological | 0.84                  | 0.04        | 4.7619                   |