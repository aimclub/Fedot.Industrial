import numpy as np
from sklearn.metrics import classification_report

from fedot_ind.api.main import FedotIndustrial

if __name__ == "__main__":
    # Options to generate time series
    real_ts = np.random.rand(1000)

    # synthetic anomaly configuration
    MAX_LENGTH = 20
    MIN_LENGTH = 10
    NUMBER = 10
    anomaly_config = {'dip': {'level': 20,
                              'number': NUMBER,
                              'min_anomaly_length': MIN_LENGTH,
                              'max_anomaly_length': MAX_LENGTH},
                      'add_noise': {'level': 80,
                                    'number': NUMBER,
                                    'noise_type': 'uniform',
                                    'min_anomaly_length': MIN_LENGTH,
                                    'max_anomaly_length': MAX_LENGTH}
                      }

    ts_smooth_normal = {'ts_type': 'smooth_normal',
                        'length': 1000,
                        'window_size': 30}
    ts_sin = {'length': 1000,
              'amplitude': 10,
              'period': 500}
    ts_random_walk = {'length': 1000,
                      'start_val': 36.6}
    ts_auto_regression = {'length': 1000,
                          'ar_params': [0.5, -0.3, 0.2],
                          'initial_values': None}
    # anomaly labels
    anomaly_labels = {}
    # Create FedotIndustrial object
    industrial = FedotIndustrial(task='ts_classification',
                                 strategy='quantile',  # or 'fedot_preset'
                                 dataset='your_dataset_name',
                                 # if 'fedot_preset' then you need to specify the following three parameters,
                                 # otherwise they will be ignored:
                                 branch_nodes=[
                                     # 'fourier_basis',
                                     # 'wavelet_basis',
                                     'data_driven_basis'
                                 ],
                                 tuning_iterations=1,
                                 tuning_timeout=1,
                                 # next are default for every solver
                                 use_cache=False,
                                 timeout=1,
                                 n_jobs=2)

    # use your own ts or let Industrial method generate_anomaly_ts produce it with anomalies
    # ts = np.random.rand(1000)

    # Get synthetic anomalies
    clean_ts, anomaly_ts, anomaly_intervals = industrial.generate_anomaly_ts(ts_data=ts_smooth_normal,
                                                                             # or ts=your own ts as np.array
                                                                             anomaly_config=anomaly_config,
                                                                             plot=True,
                                                                             overlap=0.1)

    train_data, test_data = industrial.split_ts(time_series=anomaly_ts,
                                                anomaly_dict=anomaly_intervals,
                                                binarize=False,
                                                strategy='frequent',
                                                # of strategy='unique'
                                                plot=True)

    # Fit model
    industrial.fit(features=train_data[0], target=train_data[1])

    # Predict
    predicted = industrial.predict(features=test_data[0], target=test_data[1])
    proba = industrial.predict_proba(features=test_data[0], target=test_data[1])

    industrial.get_metrics(target=test_data[1], metric_names=['f1', 'roc_auc'])

    print(classification_report(test_data[1], predicted))
    _ = 1
