# Benchmark Result Analysis: tser_full

- Task type: `ts_regression`
- Metric: `rmse`
- Metric direction: `lower`

## Coverage

| source_label | coverage_unit | expected_dataset_count | observed_dataset_count | coverage_pct | missing_dataset_count | missing_datasets | status  |
| ------------ | ------------- | ---------------------- | ---------------------- | ------------ | --------------------- | ---------------- | ------- |
| tser_full    | dataset_name  | 219                    | 63                     | 28.7671      | 0                     |                  | partial |

## Model Diagnostics

- Diagnostic rows: `12`
- Datasets with diagnostics: `3`
- Models with diagnostics: `4`

## Mean Rank

| model_name                                   | mean_rank | dataset_count |
| -------------------------------------------- | --------- | ------------- |
| Unnamed: 31                                  | 1.54762   | 63            |
| Unnamed: 32                                  | 2.56349   | 63            |
| DrCIF                                        | 8.79032   | 62            |
| FreshPRINCE                                  | 9.09677   | 62            |
| RIST                                         | 9.87273   | 55            |
| RotF                                         | 11.4355   | 62            |
| InceptionT                                   | 11.5806   | 62            |
| TSF                                          | 12.629    | 62            |
| RDST                                         | 12.9636   | 55            |
| MultiROCKET                                  | 13.0645   | 62            |
| Fedot_Industrial_legacy_baseline_features    | 14.1349   | 63            |
| RandF                                        | 14.3333   | 63            |
| SingleInception                              | 14.4603   | 63            |
| KernelEnsembleRegressor_shapelet_rbf         | 14.5      | 3             |
| Ind_bf_place                                 | 15.0317   | 63            |
| ResNet                                       | 15.873    | 63            |
| ROCKET                                       | 16.5079   | 63            |
| Ind_baseline_place                           | 16.5714   | 63            |
| XGBoost                                      | 16.8095   | 63            |
| 5NN-DTW                                      | 17.3016   | 63            |
| KernelEnsembleRegressor_adaptive_rbf_summary | 17.5      | 3             |
| KernelEnsembleRegressor_embedding_nystrom    | 18        | 3             |
| FPCR                                         | 18.1905   | 63            |
| 5NN-ED                                       | 18.3333   | 63            |
| Grid-SVR                                     | 18.4762   | 63            |
| Fedot_Industrial_legacy_baseline             | 18.5159   | 63            |
| FCN                                          | 18.746    | 63            |
| FPCR-Bs                                      | 19.1746   | 63            |
| Ridge                                        | 19.7097   | 62            |
| CNN                                          | 21.1452   | 62            |
| KernelEnsembleRegressor_score_linear_summary | 21.6667   | 3             |
| 1NN-DTW                                      | 22.3651   | 63            |
| 1NN-ED                                       | 22.746    | 63            |

## Top-K Summary

| model_name                                   | top_1 | top_3 | top_5 | top_half | dataset_count |
| -------------------------------------------- | ----- | ----- | ----- | -------- | ------------- |
| Unnamed: 31                                  | 62    | 63    | 63    | 63       | 63            |
| Unnamed: 32                                  | 50    | 61    | 61    | 61       | 63            |
| Ind_bf_place                                 | 6     | 27    | 31    | 31       | 63            |
| Ind_baseline_place                           | 6     | 15    | 28    | 29       | 63            |
| Fedot_Industrial_legacy_baseline_features    | 1     | 3     | 20    | 33       | 63            |
| InceptionT                                   | 0     | 7     | 15    | 42       | 63            |
| RDST                                         | 0     | 4     | 10    | 31       | 63            |
| MultiROCKET                                  | 0     | 4     | 8     | 37       | 63            |
| DrCIF                                        | 0     | 3     | 11    | 55       | 63            |
| FreshPRINCE                                  | 0     | 2     | 15    | 52       | 63            |
| RIST                                         | 0     | 2     | 11    | 47       | 63            |
| ResNet                                       | 0     | 2     | 5     | 29       | 63            |
| FPCR                                         | 0     | 2     | 2     | 17       | 63            |
| Fedot_Industrial_legacy_baseline             | 0     | 1     | 7     | 18       | 63            |
| RotF                                         | 0     | 1     | 4     | 47       | 63            |
| ROCKET                                       | 0     | 1     | 3     | 29       | 63            |
| TSF                                          | 0     | 1     | 3     | 42       | 63            |
| Ridge                                        | 0     | 1     | 2     | 15       | 63            |
| XGBoost                                      | 0     | 1     | 2     | 21       | 63            |
| SingleInception                              | 0     | 0     | 7     | 35       | 63            |
| Grid-SVR                                     | 0     | 0     | 4     | 19       | 63            |
| FPCR-Bs                                      | 0     | 0     | 2     | 15       | 63            |
| RandF                                        | 0     | 0     | 2     | 32       | 63            |
| 5NN-ED                                       | 0     | 0     | 1     | 14       | 63            |
| FCN                                          | 0     | 0     | 1     | 23       | 63            |
| 1NN-DTW                                      | 0     | 0     | 0     | 6        | 63            |
| 1NN-ED                                       | 0     | 0     | 0     | 3        | 63            |
| 5NN-DTW                                      | 0     | 0     | 0     | 20       | 63            |
| CNN                                          | 0     | 0     | 0     | 11       | 63            |
| KernelEnsembleRegressor_adaptive_rbf_summary | 0     | 0     | 0     | 2        | 63            |
| KernelEnsembleRegressor_embedding_nystrom    | 0     | 0     | 0     | 1        | 63            |
| KernelEnsembleRegressor_score_linear_summary | 0     | 0     | 0     | 1        | 63            |
| KernelEnsembleRegressor_shapelet_rbf         | 0     | 0     | 0     | 2        | 63            |

## Target Delta

| dataset_name              | target_model                                 | target_metric | best_reference_model | best_reference_metric | improvement | relative_improvement_pct |
| ------------------------- | -------------------------------------------- | ------------- | -------------------- | --------------------- | ----------- | ------------------------ |
| NaturalGasPricesSentiment | KernelEnsembleRegressor_adaptive_rbf_summary | 0.108889      | Unnamed: 31          | 0                     | -0.108889   | -10.8889                 |
| AppliancesEnergy          | KernelEnsembleRegressor_adaptive_rbf_summary | 3.44675       | Unnamed: 31          | 0                     | -3.44675    | -344.675                 |
| ElectricityPredictor      | KernelEnsembleRegressor_adaptive_rbf_summary | 511.329       | Ind_baseline_place   | 1                     | -510.329    | -51032.9                 |