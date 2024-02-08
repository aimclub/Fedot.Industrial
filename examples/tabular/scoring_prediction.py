# -*- coding: utf-8 -*-

"""scroing_prediction.ipynb



## Imports

"""

import pandas as pd
from fedot_ind import fedot_api
from sklearn.model_selection import train_test_split

"""## Opening Data"""

data = pd.read_csv('scoring_train.csv', index_col=0)
target = 'target'
X_train, X_test, y_train, y_test = train_test_split(data.drop(target, axis=1), data[target], test_size=0.3)

print('Shape of train', X_train.shape, 'and test', X_test.shape)

"""## Experiments settings"""

TIMEOUT = 15
N_JOBS = 1
EARLY_STOPPING_TIMEOUT = 45
METRIC = 'roc_auc'
TUNING = False

"""## Fedot (master)"""

automl = fedot_api.Fedot(
    problem='classification',
    timeout=TIMEOUT,
    n_jobs=N_JOBS,
    metric=METRIC,
    with_tuning=TUNING,
    early_stopping_timeout=EARLY_STOPPING_TIMEOUT,
    show_progress=True
)

automl.fit(features=X_train, target=y_train)
automl.predict(features=X_test)
metric_after_1 = automl.get_metrics(target=y_test)
print(metric_after_1)
fedot_industrial_report = automl.return_report()
fedot_industrial_report.head(10)

"""## Fedot with use_auto_preprocessing (master)"""

automl = fedot_api.Fedot(
    problem='classification',
    timeout=TIMEOUT,
    n_jobs=N_JOBS,
    metric=METRIC,
    with_tuning=TUNING,
    early_stopping_timeout=EARLY_STOPPING_TIMEOUT,
    show_progress=True

)

automl.fit(features=X_train, target=y_train)
automl.predict(features=X_test)
metric_after_2 = automl.get_metrics(target=y_test)
print(metric_after_2)
fedot_industrial_report = automl.return_report()
fedot_industrial_report.head(10)
print(automl.history.get_leaderboard())