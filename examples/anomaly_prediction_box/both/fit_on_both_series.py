import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from examples.anomaly_detection.detection_classification import split_series, convert_anomalies_dict_to_points
from fedot_ind.api.main import FedotIndustrial


def get_anomaly_subsequences(anomalies: np.ndarray, min_anomaly_len = 5):
    # Finding the indices of ones in the boolean array
    unique_anomalies = np.unique(anomalies[anomalies != 'no_anomaly'])
    anomaly_dict = {}
    for anomaly_class in unique_anomalies:
        anomaly_indices = np.where(anomalies == anomaly_class)[0]

        # Initializing variables
        start = None
        end = None
        subsequences = []

        # Iterating through the ones indices
        for index in anomaly_indices:
            if start is None:  # Starting a new subsequence
                start = index
                end = index
            elif index == end + 1:  # Continuing the current subsequence
                end = index
            else:  # Ending the current subsequence
                if end - start > min_anomaly_len:
                    subsequences.append((start, end))
                start = index
                end = index

        if start is not None and end is not None:  # Adding the last subsequence
            subsequences.append((start, end))
        anomaly_dict[anomaly_class] = subsequences
    return anomaly_dict


def split_time_series(series, features_columns: list, target_column: str):
    series[target_column] = series[target_column].replace('Норма', 'no_anomaly')
    anomaly_unique = get_anomaly_subsequences(series[target_column].values)

    series_train, anomaly_train, series_test, anomaly_test = split_series(series[features_columns].values,
                                                                          anomaly_unique,
                                                                          test_part=2000)
    return series_train, anomaly_train, series_test, anomaly_test


def main():
    cols = ['Power', 'Sound', 'Class']
    series = pd.read_csv('../../data/power_cons_anomaly.csv')[cols]
    series_train, anomaly_train, series_test, anomaly_test = split_time_series(series, ['Power', 'Sound'], 'Class')
    point_test = convert_anomalies_dict_to_points(series_test, anomaly_test)

    industrial = FedotIndustrial(task='anomaly_detection',
                                 dataset='custom_dataset',
                                 strategy='fedot_preset',
                                 branch_nodes=['fourier_basis'],
                                 use_cache=False,
                                 timeout=1,
                                 n_jobs=2,
                                 logging_level=20,
                                 output_folder='.')

    model = industrial.fit(features=series_train,
                           anomaly_dict=anomaly_train)

    labels = industrial.predict(features=series_test)
    probs = industrial.predict_proba(features=series_test)

    industrial.solver.get_metrics(target=point_test,
                                  metric_names=['f1', 'roc_auc'])

    print(classification_report(point_test, labels))

    industrial.solver.plot_prediction(series_test, point_test)


if __name__ == '__main__':
    main()
