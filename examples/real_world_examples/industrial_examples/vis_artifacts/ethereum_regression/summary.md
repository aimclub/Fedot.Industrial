# Benchmark Result Analysis: ethereum_regression_current_api_preview

- Task type: `ts_regression`
- Metric: `rmse`
- Metric direction: `lower`

## Mean Rank

| model_name                                   | mean_rank | dataset_count |
| -------------------------------------------- | --------- | ------------- |
| KernelEnsembleRegressor_score_linear_summary | 1         | 2             |
| KernelEnsembleRegressor_adaptive_rbf_summary | 2         | 2             |
| KernelEnsembleRegressor_shapelet_rbf         | 3         | 2             |
| KernelEnsembleRegressor_embedding_nystrom    | 4         | 2             |
| KernelEnsembleRegressor                      | 5         | 2             |
| PDLRegressor                                 | 6         | 2             |
| LinearRegressor                              | 7         | 2             |

## Top-K Summary

| model_name                                   | top_1 | top_3 | top_5 | top_half | dataset_count |
| -------------------------------------------- | ----- | ----- | ----- | -------- | ------------- |
| KernelEnsembleRegressor_score_linear_summary | 2     | 2     | 2     | 2        | 2             |
| KernelEnsembleRegressor_adaptive_rbf_summary | 0     | 2     | 2     | 2        | 2             |
| KernelEnsembleRegressor_shapelet_rbf         | 0     | 2     | 2     | 2        | 2             |
| KernelEnsembleRegressor                      | 0     | 0     | 2     | 0        | 2             |
| KernelEnsembleRegressor_embedding_nystrom    | 0     | 0     | 2     | 0        | 2             |
| LinearRegressor                              | 0     | 0     | 0     | 0        | 2             |
| PDLRegressor                                 | 0     | 0     | 0     | 0        | 2             |

## Target Delta

| dataset_name                  | target_model                                 | target_metric | best_reference_model                         | best_reference_metric | improvement | relative_improvement_pct |
| ----------------------------- | -------------------------------------------- | ------------- | -------------------------------------------- | --------------------- | ----------- | ------------------------ |
| ethereum_regression_preview_2 | KernelEnsembleRegressor_score_linear_summary | 0.41          | KernelEnsembleRegressor_adaptive_rbf_summary | 0.45                  | 0.04        | 8.88889                  |
| ethereum_regression_preview_1 | KernelEnsembleRegressor_score_linear_summary | 0.38          | KernelEnsembleRegressor_adaptive_rbf_summary | 0.42                  | 0.04        | 9.52381                  |