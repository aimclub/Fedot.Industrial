from fedot.api.main import Fedot

from fedot_ind.core.ensemble.kernel_ensemble import KernelEnsembler, init_kernel_ensemble
import pytest

from fedot_ind.core.ensemble.rank_ensembler import RankEnsemble
from fedot_ind.tools.loader import DataLoader
from fedot_ind.api.utils.path_lib import PROJECT_PATH
import os


@pytest.fixture()
def kernel_dict():
    return {'wavelet': [{'feature_generator_type': 'signal',
                         'feature_hyperparams': {'wavelet': "mexh",
                                                 'n_components': 2}
                         }
                        ],
            'quantile': [{'feature_generator_type': 'quantile',
                          'feature_hyperparams': {'window_mode': True,
                                                  'window_size': 25}
                          }
                         ]
            }


@pytest.fixture()
def data():
    ds_name = 'ItalyPowerDemand'
    folder_path = os.path.join(PROJECT_PATH, 'tests/data/datasets')
    return DataLoader(dataset_name=ds_name).load_data()


def test_kernel_ensembler(kernel_dict, data):
    train_data, test_data = data
    n_best = 2
    feature_dict = {}
    proba_dict = {}
    metric_dict = {}
    dataset_name = 'ItalyPowerDemand'

    fg_names = []
    for key in kernel_dict:
        for model_params in kernel_dict[key]:
            fg_names.append(f'{key}_{model_params}')

    set_of_fg, train_feats, train_target, test_feats, test_target = init_kernel_ensemble(train_data,
                                                                                         test_data,
                                                                                         kernel_list=kernel_dict)
    n_best_generators = set_of_fg.T.nlargest(n_best, 0).index
    for rank in range(n_best):
        fg_rank = n_best_generators[rank]
        train_best = train_feats[fg_rank]
        test_best = test_feats[fg_rank]
        feature_dict.update({fg_names[rank]: (test_best, test_best)})

    for model_name, feature in feature_dict.items():
        industrial = Fedot(metric='roc_auc', timeout=0.1, problem='classification', n_jobs=6)
        model = industrial.fit(feature[0], train_target)
        labels = industrial.predict(feature[1])
        proba_dict.update({model_name: industrial.predict_proba(feature[1])})
        metric_dict.update({model_name: industrial.get_metrics(test_target, metric_names=['roc_auc', 'f1', 'accuracy'])})
    rank_ensembler = RankEnsemble(dataset_name=dataset_name,
                                  proba_dict={dataset_name: proba_dict},
                                  metric_dict={dataset_name: metric_dict})

    ensemble_result = rank_ensembler.ensemble()
    assert ensemble_result is not None


