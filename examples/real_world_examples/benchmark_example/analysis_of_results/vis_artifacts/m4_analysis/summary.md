# Benchmark Result Analysis: m4_full

- Task type: `forecasting`
- Metric: `owa`
- Metric direction: `lower`

## Coverage

| source_label | coverage_unit | expected_dataset_count | observed_dataset_count | coverage_pct | missing_dataset_count | missing_datasets | status  |
| ------------ | ------------- | ---------------------- | ---------------------- | ------------ | --------------------- | ---------------- | ------- |
| m4_full      | series_id     | 48000                  | 5                      | 0.0104       | 0                     |                  | partial |

## Model Diagnostics

- Diagnostic rows: `20`
- Datasets with diagnostics: `1`
- Models with diagnostics: `4`

## Mean Rank

| model_name                                      | mean_rank | dataset_count |
| ----------------------------------------------- | --------- | ------------- |
| LaggedRidgeForecaster                           | 7.8       | 1             |
| NaiveLastValue                                  | 10.2      | 1             |
| KernelEnsembleForecaster_identity_shapelet      | 10.8      | 1             |
| KernelEnsembleForecaster_embedding_nystrom_okhs | 13.2      | 1             |

## Top-K Summary

| model_name                                      | top_1 | top_3 | top_5 | top_half | dataset_count |
| ----------------------------------------------- | ----- | ----- | ----- | -------- | ------------- |
| LaggedRidgeForecaster                           | 1     | 2     | 3     | 2        | 1             |
| KernelEnsembleForecaster_identity_shapelet      | 0     | 1     | 1     | 0        | 1             |
| NaiveLastValue                                  | 0     | 0     | 1     | 0        | 1             |
| KernelEnsembleForecaster_embedding_nystrom_okhs | 0     | 0     | 0     | 0        | 1             |

## Target Delta

| dataset_name               | target_model                                    | target_metric | best_reference_model | best_reference_metric | improvement | relative_improvement_pct |
| -------------------------- | ----------------------------------------------- | ------------- | -------------------- | --------------------- | ----------- | ------------------------ |
| m4_monthly_kernel_learning | KernelEnsembleForecaster_embedding_nystrom_okhs | 1.76605       | NaiveLastValue       | 0.79718               | -0.968874   | -121.538                 |