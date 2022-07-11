import os

import pandas as pd


def read_anomaly_data():
    all_files = []
    for root, dirs, files in os.walk("./"):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))

    dfs = []
    for path in all_files:
        df = pd.read_csv(path, index_col='datetime', sep=';', parse_dates=True)
        dfs.append(df)
    print('Features:')
    for col in dfs[2].columns:
        print('\t', col)

    # datasets with anomalies loading
    list_of_df = [pd.read_csv(file,
                              sep=';',
                              index_col='datetime',
                              parse_dates=True) for file in all_files if 'anomaly-free' not in file]
    # anomaly-free df loading
    anomaly_free_df = pd.read_csv([file for file in all_files if 'anomaly-free' in file][0],
                                  sep=';',
                                  index_col='datetime',
                                  parse_dates=True)
    true_cp = [df.changepoint for df in list_of_df]

    return list_of_df, anomaly_free_df, true_cp
