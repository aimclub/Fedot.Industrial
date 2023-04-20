import os

import pytest

from fedot_ind.core.architecture.postprocessing.results_picker import ResultsPicker
from fedot_ind.core.architecture.utils.utils import PROJECT_PATH
from fedot_ind.core.ensemble.static.RankEnsembler import RankEnsemble


@pytest.fixture()
def get_proba_metric_dict():
    results_path = os.path.join(PROJECT_PATH, '../tests/data/classification_results')
    picker = ResultsPicker(path=results_path)
    proba_dict, metric_dict = picker.run()
    return proba_dict, metric_dict


def test_rank_ensemble_umd(get_proba_metric_dict):
    proba_dict, metric_dict = get_proba_metric_dict

    ensembler_umd = RankEnsemble(dataset_name='UMD',
                                 proba_dict=proba_dict,
                                 metric_dict=metric_dict)
    result = ensembler_umd.ensemble()

    assert result['Base_metric'] == 0.993
    assert result['Base_model'] == 'fedot_preset'


def test_rank_ensemble_chinatown(get_proba_metric_dict):
    proba_dict, metric_dict = get_proba_metric_dict
    ensembler_chinatown = RankEnsemble(dataset_name='Chinatown',
                                       proba_dict=proba_dict,
                                       metric_dict=metric_dict)
    result = ensembler_chinatown.ensemble()

    assert result['Base_metric'] == 0.954
    assert result['Base_model'] == 'fedot_preset'


def test_rank_ensemble_italy(get_proba_metric_dict):
    proba_dict, metric_dict = get_proba_metric_dict
    ensembler_italy = RankEnsemble(dataset_name='ItalyPowerDemand',
                                   proba_dict=proba_dict,
                                   metric_dict=metric_dict)
    result = ensembler_italy.ensemble()

    assert result['Base_metric'] == 0.926
    assert result['Base_model'] == 'fedot_preset'
