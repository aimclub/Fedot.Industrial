# Benchmark Result Analysis: ucr_multivariate_full

- Task type: `ts_classification`
- Metric: `accuracy`
- Metric direction: `higher`

## Coverage

| source_label          | coverage_unit | expected_dataset_count | observed_dataset_count | coverage_pct | missing_dataset_count | missing_datasets | status |
| --------------------- | ------------- | ---------------------- | ---------------------- | ------------ | --------------------- | ---------------- | ------ |
| ucr_multivariate_full | dataset_name  | 26                     | 26                     | 100          | 0                     |                  | full   |

## Model Diagnostics

No model diagnostics rows.

## Mean Rank

| model_name       | mean_rank | dataset_count |
| ---------------- | --------- | ------------- |
| HC2              | 4.96154   | 26            |
| ROCKET           | 6.36538   | 26            |
| Arsenal          | 7.46154   | 26            |
| DrCIF            | 7.67308   | 26            |
| CIF              | 8.44231   | 26            |
| MUSE             | 8.65      | 20            |
| HC1              | 9.13462   | 26            |
| Fedot_Industrial | 9.21154   | 26            |
| TDE              | 9.69231   | 26            |
| ResNet           | 10.2692   | 26            |
| mrseql           | 10.625    | 20            |
| STC              | 10.9423   | 26            |
| InceptionTime    | 11.375    | 20            |
| gRSF             | 11.3846   | 26            |
| DTW_A            | 11.86     | 25            |
| DTW_D            | 12.9038   | 26            |
| CBOSS            | 13.0385   | 26            |
| RISE             | 13.6346   | 26            |
| TSF              | 14.0577   | 26            |
| TapNet           | 14.55     | 20            |
| DTW_I            | 17        | 26            |

## Top-K Summary

| model_name       | top_1 | top_3 | top_5 | top_half | dataset_count |
| ---------------- | ----- | ----- | ----- | -------- | ------------- |
| HC2              | 6     | 15    | 19    | 24       | 26            |
| DrCIF            | 6     | 11    | 14    | 19       | 26            |
| ROCKET           | 6     | 7     | 15    | 22       | 26            |
| Fedot_Industrial | 4     | 10    | 11    | 15       | 26            |
| ResNet           | 4     | 8     | 10    | 16       | 26            |
| Arsenal          | 4     | 6     | 11    | 21       | 26            |
| HC1              | 4     | 6     | 8     | 18       | 26            |
| MUSE             | 4     | 5     | 8     | 15       | 26            |
| TapNet           | 4     | 4     | 4     | 5        | 26            |
| CIF              | 3     | 6     | 11    | 18       | 26            |
| TDE              | 3     | 5     | 7     | 17       | 26            |
| InceptionTime    | 3     | 4     | 4     | 10       | 26            |
| DTW_A            | 2     | 3     | 5     | 12       | 26            |
| CBOSS            | 2     | 3     | 4     | 8        | 26            |
| RISE             | 2     | 2     | 5     | 11       | 26            |
| mrseql           | 2     | 2     | 4     | 14       | 26            |
| DTW_D            | 1     | 4     | 4     | 9        | 26            |
| STC              | 1     | 2     | 6     | 16       | 26            |
| gRSF             | 1     | 2     | 4     | 12       | 26            |
| TSF              | 1     | 2     | 3     | 9        | 26            |
| DTW_I            | 0     | 0     | 1     | 2        | 26            |

## Target Delta

| dataset_name              | target_model     | target_metric | best_reference_model | best_reference_metric | improvement | relative_improvement_pct |
| ------------------------- | ---------------- | ------------- | -------------------- | --------------------- | ----------- | ------------------------ |
| FaceDetection             | Fedot_Industrial | 0.877         | HC2                  | 0.66                  | 0.217       | 32.8788                  |
| LSST                      | Fedot_Industrial | 0.688         | HC2                  | 0.643                 | 0.045       | 6.99844                  |
| Cricket                   | Fedot_Industrial | 1             | Arsenal              | 1                     | 0           | 0                        |
| BasicMotions              | Fedot_Industrial | 1             | Arsenal              | 1                     | 0           | 0                        |
| PenDigits                 | Fedot_Industrial | 0.986         | InceptionTime        | 0.988                 | -0.002      | -0.202429                |
| Heartbeat                 | Fedot_Industrial | 0.78          | DrCIF                | 0.79                  | -0.01       | -1.26582                 |
| FingerMovements           | Fedot_Industrial | 0.59          | DrCIF                | 0.6                   | -0.01       | -1.66667                 |
| ArticularyWordRecognition | Fedot_Industrial | 0.977         | Arsenal              | 0.993                 | -0.016      | -1.61128                 |
| NATOPS                    | Fedot_Industrial | 0.961         | ResNet               | 0.978                 | -0.017      | -1.73824                 |
| RacketSports              | Fedot_Industrial | 0.908         | MUSE                 | 0.928                 | -0.02       | -2.15517                 |
| Epilepsy                  | Fedot_Industrial | 0.978         | CBOSS                | 1                     | -0.022      | -2.2                     |
| PEMS-SF                   | Fedot_Industrial | 0.954         | CIF                  | 1                     | -0.046      | -4.6                     |
| HandMovementDirection     | Fedot_Industrial | 0.541         | CIF                  | 0.595                 | -0.054      | -9.07563                 |
| Libras                    | Fedot_Industrial | 0.877         | ResNet               | 0.944                 | -0.067      | -7.09746                 |
| ERing                     | Fedot_Industrial | 0.926         | DrCIF                | 0.993                 | -0.067      | -6.74723                 |
| SelfRegulationSCP2        | Fedot_Industrial | 0.5           | ROCKET               | 0.572                 | -0.072      | -12.5874                 |
| PhonemeSpectra            | Fedot_Industrial | 0.222         | HC1                  | 0.321                 | -0.099      | -30.8411                 |
| UWaveGestureLibrary       | Fedot_Industrial | 0.833         | ROCKET               | 0.941                 | -0.108      | -11.4772                 |
| MotorImagery              | Fedot_Industrial | 0.49          | HC1                  | 0.61                  | -0.12       | -19.6721                 |
| AtrialFibrillation        | Fedot_Industrial | 0.267         | MUSE                 | 0.4                   | -0.133      | -33.25                   |
| SelfRegulationSCP1        | Fedot_Industrial | 0.785         | TapNet               | 0.935                 | -0.15       | -16.0428                 |
| EigenWorms                | Fedot_Industrial | 0.786         | HC2                  | 0.947                 | -0.161      | -17.0011                 |
| StandWalkJump             | Fedot_Industrial | 0.333         | Arsenal              | 0.533                 | -0.2        | -37.5235                 |
| DuckDuckGeese             | Fedot_Industrial | 0.4           | ResNet               | 0.62                  | -0.22       | -35.4839                 |
| Handwriting               | Fedot_Industrial | 0.408         | InceptionTime        | 0.642                 | -0.234      | -36.4486                 |
| EthanolConcentration      | Fedot_Industrial | 0.281         | STC                  | 0.821                 | -0.54       | -65.7734                 |