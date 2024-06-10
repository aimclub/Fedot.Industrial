from fedot_ind.api.utils.data import init_input_data
from fedot_ind.api.utils.industrial_strategy import IndustrialStrategy
import pytest
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator

STRATEGY = ['federated_automl', 'lora_strategy', 'kernel_automl', 'forecasting_assumptions', 'forecasting_exogenous']

CONFIGS = {'federated_automl': {'problem': 'classification',
                                'metric': 'f1',
                                'timeout': 0.1,
                                'industrial_strategy': 'federated_automl',
                                'industrial_strategy_params': {}
                                },

           'lora_strategy': {'problem': 'classification',
                             'metric': 'accuracy',
                             'timeout': 0.1,
                             'with_tuning': False,
                             'industrial_strategy': 'lora_strategy',
                             'industrial_strategy_params': {}
                             },

           'kernel_automl': {'problem': 'classification',
                             'metric': 'f1',
                             'timeout': 0.1,
                             'with_tuning': False,
                             'industrial_strategy': 'kernel_automl',
                             'industrial_strategy_params': {}
                             },

           'forecasting_assumptions': {'problem': 'ts_forecasting',
                                       'metric': 'rmse',
                                       'timeout': 0.1,
                                       'with_tuning': False,
                                       'industrial_strategy': 'forecasting_assumptions',
                                       'industrial_strategy_params': {}},

           # 'forecasting_exogenous': {}
           }


@pytest.fixture()
def classification_data():
    train_data, test_data = TimeSeriesDatasetsGenerator(num_samples=80,
                                                        task='classification',
                                                        max_ts_len=50,
                                                        binary=True,
                                                        test_size=0.5,
                                                        multivariate=False).generate_data()
    return train_data, test_data


@pytest.mark.parametrize('strategy', STRATEGY)
def test_industrial_strategy(strategy):
    cnfg = CONFIGS[strategy]
    base = IndustrialStrategy(industrial_strategy_params=None,
                              industrial_strategy=strategy,
                              api_config=cnfg, )

    assert base is not None


@pytest.mark.parametrize('strategy', ('federated_automl', 'kernel_automl'))
def test_clf_strategy_fit(strategy, classification_data):
    data = classification_data[0]
    input_data = init_input_data(X=data[0], y=data[1])
    base = IndustrialStrategy(industrial_strategy_params=None,
                              industrial_strategy=strategy,
                              api_config=CONFIGS[strategy])
    base.fit(input_data)
    assert base is not None
