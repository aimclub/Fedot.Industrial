import numpy as np
import pandas as pd

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.path_lib import PROJECT_PATH

if __name__ == "__main__":
    dataset_name = PROJECT_PATH + \
        '/examples/data/forecasting\\monash_benchmark\\MonashBitcoin_30.csv'
    horizon = 60
    metric_names = ('smape', 'rmse', 'median_absolute_error')

    train_data = pd.read_csv(dataset_name)
    variables = train_data['label'].unique().tolist()
    exog_var = ['send_usd', 'market_cap',
                'median_transaction_value', 'google_trends']
    exog_ts = np.vstack(
        [train_data[train_data['label'] == var]['value'].values for var in exog_var])
    exog_ts = exog_ts[0, :]
    ts = train_data[train_data['label'] == 'price']['value'].values
    target = ts[-horizon:].flatten()
    input_data = (ts, target)

    api_config = dict(problem='ts_forecasting',
                      metric='rmse',
                      timeout=15,
                      with_tuning=False,
                      pop_size=10,
                      industrial_strategy_params={'exog_variable': exog_ts},
                      task_params={'forecast_length': horizon},
                      industrial_strategy='forecasting_exogenous',
                      n_jobs=2,
                      logging_level=30)
    industrial = FedotIndustrial(**api_config)
    industrial.fit(input_data)
