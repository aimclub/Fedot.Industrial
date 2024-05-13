from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.tools.example_utils import read_results, create_comprasion_df

forecast_result_path = PROJECT_PATH + \
    '/examples/automl_example/api_example/time_series/ts_forecasting/forecasts/'

if __name__ == "__main__":
    for metric in ['rmse', 'smape']:
        df_forecast, df_metrics = read_results(forecast_result_path)
        df_comprasion = create_comprasion_df(df_metrics, metric)
        print(df_comprasion['industrial_Wins_All'].value_counts())
        print(df_comprasion['industrial_Wins_AG'].value_counts())
        print(df_comprasion['industrial_Wins_NBEATS'].value_counts())
        _ = 1
