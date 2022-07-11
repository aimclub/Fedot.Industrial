import numpy as np
import pandas as pd
from cases.anomaly_detection.SSTdetector import SingularSpectrumTransformation
import matplotlib.pyplot as plt
from tsad.evaluating.evaluating import evaluating
from cases.anomaly_detection.data.data_reader import read_anomaly_data


def plot_data_and_score(raw_data, score, change_point, anomaly, col):
    f, ax = plt.subplots(4, 1, figsize=(20, 10))
    ax[0].plot(raw_data)
    ax[0].set_title("ts_for_{}".format(col))
    ax[1].plot(score, "r")
    ax[1].set_title("score")
    ax[2].plot(change_point, "r")
    ax[2].set_title("change_point")
    ax[3].plot(anomaly, "r")
    ax[3].set_title("anomaly")
    f.show()


if __name__ == '__main__':
    list_of_df, anomaly_free_df, true_cp = read_anomaly_data()
    df = list_of_df[0]
    change_point = df.changepoint.values
    anomaly = df.anomaly.values
    # list_of_df[0].plot(figsize=(12, 6))
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.title('Signals')
    # plt.show()
    all_score_nab = []
    all_score_add = []
    for df in list_of_df[:1]:
        change_point = df.changepoint.values
        anomaly = df.anomaly.values
        nab_scores = {}
        add_scores = {}
        sum_of_cp = []
        for i in range(8):
            x = df.iloc[:, i].values
            scorer = SingularSpectrumTransformation(time_series=x,
                                                    ts_window_length=60,
                                                    lag=10,
                                                    trajectory_window_length=25)
            score = scorer.score_offline(dynamic_mode=False)
            score_diff = np.diff(score)
            col_name = 'score_for_{}'.format(df.columns[i])
            q_95 = np.quantile(score_diff, 0.995)
            filtred = list(filter(lambda x: x > q_95, score_diff))
            idx = list(map(lambda x: np.where(np.isclose(score_diff, x)), filtred))
            idx = [x[0][0] for x in idx]
            predicted_cp = [0 for i in range(anomaly.size - len(score))] + [1 if score.index(x) in idx else 0 for x in
                                                                            score]
            df[col_name] = predicted_cp
            sum_of_cp.append(predicted_cp)
            plot_data_and_score(x, predicted_cp, change_point, anomaly, col=col_name)
            predicted_cp = pd.Series(predicted_cp)
            predicted_cp.index = df.index
            nab = evaluating(df.changepoint, predicted_cp, metric='nab', numenta_time='30 sec')
            add = evaluating(df.changepoint, predicted_cp, metric='average_time', numenta_time='30 sec')
            nab_scores.update({col_name: nab['Standart']})
            add_scores.update({col_name: add})
        df['all_cp'] = df.iloc[:, 11:-1].sum(axis=1)
        plot_data_and_score(x, df['all_cp'].values, change_point, anomaly, col='all_intervals')
        nab = evaluating(df.changepoint, df['all_cp'].values, metric='nab', numenta_time='30 sec')
        all_score_nab.append(nab_scores)
        all_score_add.append(add_scores)
    feature = list(all_score_nab[0].keys())
    mean_metric = {}
    for f in feature:
        metric = [x[f] for x in all_score_nab]
        mean_metric.update({f: np.mean(metric)})
    _ = 1
