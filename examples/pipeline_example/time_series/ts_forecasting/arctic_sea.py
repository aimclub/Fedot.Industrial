from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from fedot.api.main import Fedot
from fedot.core.repository.tasks import TsForecastingParams
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from fedot_ind.api.utils.path_lib import PROJECT_PATH

matplotlib.use('TKagg')

horizon = 365
PATH = Path(PROJECT_PATH, 'examples', 'data', 'ices_areas_ts.csv')

time_series_df = pd.read_csv(PATH).iloc[:, 1:]
target_series = time_series_df['Карское'].values
train_data = {}
feature_cols = list(time_series_df.columns)
for col in time_series_df:
    current_ts = time_series_df[col].values[:-horizon]
    train_data.update({str(col): current_ts})

# Configure AutoML
#del train_data['Карское']

task_parameters = TsForecastingParams(forecast_length=horizon)
model = Fedot(problem='ts_forecasting', task_params=task_parameters)
obtained_pipeline = model.fit(features=train_data,
                              target=target_series[:-horizon])

obtained_pipeline.show()

# Use historical value to make forecast
forecast = model.forecast(train_data)
forecast[forecast<0] = 0
real_target = target_series[-horizon:]

plt.plot(target_series, label='real data')
plt.plot(np.arange(len(target_series) - horizon, len(target_series)), forecast, label='forecast')

plt.grid()
plt.legend()
plt.show()

print(mean_squared_error(real_target, forecast, squared=False))
print(mean_absolute_percentage_error(real_target, forecast))
_ = 1
