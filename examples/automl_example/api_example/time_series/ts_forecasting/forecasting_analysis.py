import os
import pandas as pd

from fedot_ind.api.utils.path_lib import PROJECT_PATH

forecast_result_path = PROJECT_PATH + '/examples/automl_example/api_example/time_series/ts_forecasting/forecasts/'


def read_results(forecast_result_path):
    results = os.listdir(forecast_result_path)
    df_forecast = []
    df_metrics = []
    for file in results:
        df = pd.read_csv(f'{forecast_result_path}/{file}')
        name = file.split('_')[0]
        df['dataset_name'] = name
        if file.__contains__('forecast'):
            df_forecast.append(df)
        else:
            df_metrics.append(df)
    return df_forecast, df_metrics


def create_comprasion_df(df, metric: str = 'rmse'):
    df_full = pd.concat(df)
    df_full = df_full[df_full['Unnamed: 0'] == metric]
    df_full = df_full .drop('Unnamed: 0', axis=1)
    df_full['Difference_industrial'] = (df_full.iloc[:, 1:2].min(axis=1) - df_full['industrial'])
    df_full['industrial_Wins'] = df_full.apply(lambda row: 'Win' if row.loc['Difference_industrial'] > 0 else 'Loose',
                                          axis=1)
    return df_full


if __name__ == "__main__":
    for metric in ['rmse', 'smape']:
        df_forecast, df_metrics = read_results(forecast_result_path)
        df_comprasion = create_comprasion_df(df_metrics, metric)
        print(df_comprasion['industrial_Wins'].value_counts())
