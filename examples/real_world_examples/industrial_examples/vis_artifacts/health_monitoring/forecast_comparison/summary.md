# health_monitoring: multi-model forecast preview

- Series: `health_monitoring_forecast_preview`
- Compared models: `5`

## Metrics

| model_name                                      | mae       | rmse      | smape    |
| ----------------------------------------------- | --------- | --------- | -------- |
| KernelEnsembleForecaster                        | 0.0711833 | 0.0848939 | 0.476926 |
| KernelEnsembleForecaster_embedding_nystrom_okhs | 0.0842583 | 0.0990499 | 0.575226 |
| KernelEnsembleForecaster_identity_shapelet      | 0.132333  | 0.170241  | 0.912578 |
| LaggedRidgeForecaster                           | 0.206033  | 0.256937  | 1.41574  |
| NaiveLastValue                                  | 0.301383  | 0.34687   | 2.04187  |