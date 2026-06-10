# Benchmark Result Analysis: m4_sample

- Task type: `forecasting`
- Metric: `owa`
- Metric direction: `lower`

## Mean Rank

| model_name                                      | mean_rank | dataset_count |
| ----------------------------------------------- | --------- | ------------- |
| KernelEnsembleForecaster_embedding_nystrom_okhs | 1         | 3             |
| Fedot_Industrial_current                        | 2         | 3             |
| KernelEnsembleForecaster_identity_shapelet      | 3         | 3             |
| NBEATS                                          | 4.33333   | 3             |
| ESRNN                                           | 4.66667   | 3             |
| Theta                                           | 6         | 3             |
| LaggedRidgeForecaster                           | 7         | 3             |
| NaiveLastValue                                  | 8         | 3             |

## Top-K Summary

| model_name                                      | top_1 | top_3 | top_5 | top_half | dataset_count |
| ----------------------------------------------- | ----- | ----- | ----- | -------- | ------------- |
| KernelEnsembleForecaster_embedding_nystrom_okhs | 3     | 3     | 3     | 3        | 3             |
| Fedot_Industrial_current                        | 0     | 3     | 3     | 3        | 3             |
| KernelEnsembleForecaster_identity_shapelet      | 0     | 3     | 3     | 3        | 3             |
| ESRNN                                           | 0     | 0     | 3     | 1        | 3             |
| NBEATS                                          | 0     | 0     | 3     | 2        | 3             |
| LaggedRidgeForecaster                           | 0     | 0     | 0     | 0        | 3             |
| NaiveLastValue                                  | 0     | 0     | 0     | 0        | 3             |
| Theta                                           | 0     | 0     | 0     | 0        | 3             |

## Target Delta

| dataset_name | target_model             | target_metric | best_reference_model                            | best_reference_metric | improvement | relative_improvement_pct |
| ------------ | ------------------------ | ------------- | ----------------------------------------------- | --------------------- | ----------- | ------------------------ |
| Monthly      | Fedot_Industrial_current | 0.82          | KernelEnsembleForecaster_embedding_nystrom_okhs | 0.81                  | -0.01       | -1.23457                 |
| Daily        | Fedot_Industrial_current | 0.89          | KernelEnsembleForecaster_embedding_nystrom_okhs | 0.88                  | -0.01       | -1.13636                 |
| Weekly       | Fedot_Industrial_current | 0.86          | KernelEnsembleForecaster_embedding_nystrom_okhs | 0.85                  | -0.01       | -1.17647                 |