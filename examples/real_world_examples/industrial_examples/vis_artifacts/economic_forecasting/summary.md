# Benchmark Result Analysis: economic_forecasting_current_api_preview

- Task type: `forecasting`
- Metric: `smape`
- Metric direction: `lower`

## Mean Rank

| model_name                                      | mean_rank | dataset_count |
| ----------------------------------------------- | --------- | ------------- |
| NaiveLastValue                                  | 1         | 2             |
| LaggedRidgeForecaster                           | 2         | 2             |
| KernelEnsembleForecaster_identity_shapelet      | 3         | 2             |
| KernelEnsembleForecaster_embedding_nystrom_okhs | 4         | 2             |
| KernelEnsembleForecaster                        | 5         | 2             |

## Top-K Summary

| model_name                                      | top_1 | top_3 | top_5 | top_half | dataset_count |
| ----------------------------------------------- | ----- | ----- | ----- | -------- | ------------- |
| NaiveLastValue                                  | 2     | 2     | 2     | 2        | 2             |
| KernelEnsembleForecaster_identity_shapelet      | 0     | 2     | 2     | 0        | 2             |
| LaggedRidgeForecaster                           | 0     | 2     | 2     | 2        | 2             |
| KernelEnsembleForecaster                        | 0     | 0     | 2     | 0        | 2             |
| KernelEnsembleForecaster_embedding_nystrom_okhs | 0     | 0     | 2     | 0        | 2             |

## Target Delta

| dataset_name                   | target_model   | target_metric | best_reference_model  | best_reference_metric | improvement | relative_improvement_pct |
| ------------------------------ | -------------- | ------------- | --------------------- | --------------------- | ----------- | ------------------------ |
| economic_forecasting_preview_2 | NaiveLastValue | 0.41          | LaggedRidgeForecaster | 0.45                  | 0.04        | 8.88889                  |
| economic_forecasting_preview_1 | NaiveLastValue | 0.38          | LaggedRidgeForecaster | 0.42                  | 0.04        | 9.52381                  |