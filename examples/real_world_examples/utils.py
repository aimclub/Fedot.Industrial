import pandas as pd


def sota_compare(
        data_path,
        dataset_name,
        best_baseline,
        best_tuned,
        df_automl):
    df = pd.read_csv(data_path + '/ts_regression_sota_results.csv', sep=';')
    df = df[df['ds/type'] == dataset_name].iloc[:, :25]
    df.index = df['algorithm']
    df = df.drop(['algorithm', 'ds/type'], axis=1)
    df = df.replace(',', '.', regex=True).astype(float)
    df['Fedot_Industrial_baseline'] = best_baseline
    df['Fedot_Industrial_tuned'] = best_tuned
    df['Fedot_Industrial_AutoML'] = 0
    df.loc['min', 'Fedot_Industrial_AutoML'] = df_automl['root_mean_squared_error:'].min()
    df.loc['max', 'Fedot_Industrial_AutoML'] = df_automl['root_mean_squared_error:'].max()
    df.loc['average', 'Fedot_Industrial_AutoML'] = df_automl['root_mean_squared_error:'].mean()
    df = df.T
    return df
