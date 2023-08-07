import matplotlib.pyplot as plt
import numpy as np

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.operation.transformation.splitter import TSSplitter


def generate_time_series(to_plot: bool = True,
                         ts_length: int = 500,
                         num_anomaly_classes: int = 4,
                         num_of_anomalies: int = 20,
                         min_anomaly_length: int = 5,
                         max_anomaly_length: int = 15):
    import matplotlib.patches as mpatches

    np.random.seed(42)

    time_series = np.random.normal(0, 1, ts_length)
    anomaly_classes = [f'anomaly{i + 1}' for i in range(num_anomaly_classes)]

    anomaly_intervals = {}

    for i in range(num_of_anomalies):
        anomaly_class = np.random.choice(anomaly_classes)

        start_idx = np.random.randint(max_anomaly_length,
                                      ts_length - max_anomaly_length)

        end_idx = start_idx + np.random.randint(min_anomaly_length,
                                                max_anomaly_length + 1)

        anomaly = np.random.normal(5, 1, end_idx - start_idx)

        time_series[start_idx:end_idx] += anomaly

        if anomaly_class in anomaly_intervals:
            anomaly_intervals[anomaly_class].append([start_idx, end_idx])
        else:
            anomaly_intervals[anomaly_class] = [[start_idx, end_idx]]

    if to_plot:
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(time_series)
        ax.set_title('Time Series with Anomalies')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')

        cmap = generate_colors(len(anomaly_intervals.keys()))
        color_dict = {cls: color for cls, color in zip(anomaly_intervals.keys(), cmap)}

        legend_patches = [mpatches.Patch(color=color_dict[cls],
                                         label=cls) for cls in anomaly_intervals.keys()]

        for anomaly_class, intervals in anomaly_intervals.items():
            for interval in intervals.split(', '):
                start_idx, end_idx = map(int, interval.split(':'))
                ax.axvspan(start_idx, end_idx, alpha=0.3, color=color_dict[anomaly_class])

        plt.legend(handles=set(legend_patches))
        plt.show()

    return time_series, anomaly_intervals


def generate_colors(num_colors):
    colormap = plt.cm.get_cmap('tab10')
    colors = [colormap(i) for i in range(num_colors)]
    return colors


if __name__ == "__main__":
    time_series, anomaly_intervals = generate_time_series(to_plot=False,
                                                          ts_length=1000,
                                                          num_anomaly_classes=4,
                                                          num_of_anomalies=50)

    splitter = TSSplitter(time_series=time_series,
                          anomaly_dict=anomaly_intervals,
                          strategy='frequent')

    train_data, test_data = splitter.split(plot=False,
                                           binarize=False)

    industrial = FedotIndustrial(task='ts_classification',
                                 dataset='custom_dataset',
                                 strategy='fedot_preset',
                                 branch_nodes=['fourier_basis'],
                                 use_cache=False,
                                 timeout=1,
                                 n_jobs=2,
                                 logging_level=20,
                                 output_folder='.')

    model = industrial.fit(features=train_data[0],
                           target=train_data[1])

    labels = industrial.predict(features=test_data[0],
                                target=test_data[1])
    probs = industrial.predict_proba(features=test_data[0],
                                     target=test_data[1])

    industrial.solver.get_metrics(target=test_data[1],
                                  metric_names=['f1', 'roc_auc'])
    from sklearn.metrics import classification_report

    print(classification_report(test_data[1], labels))
