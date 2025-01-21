from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.loader import DataLoader
import numpy as np
from sklearn.utils import shuffle
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, \
    DEFAULT_AUTOML_LEARNING_CONFIG

def load_data(dataset_dir='./fedot_ind/data/Lightning7'):
    data_train = np.genfromtxt(dataset_dir + f'/{dataset_name}_TRAIN.txt')
    data_test = np.genfromtxt(dataset_dir + f'/{dataset_name}_TEST.txt')
    train_features, train_target = data_train[:, 1:], data_train[:, 0]
    test_features, test_target = data_test[:, 1:], data_test[:, 0]
    train_features, train_target = shuffle(train_features, train_target)
    input_train = (train_features, train_target)
    input_test = (test_features, test_target)

    dataset = dict(test_data=input_test, train_data=input_train)
    return dataset


def create_feature_generator_strategy():
    stat_params = {'window_size': 0, 'stride': 1, 'add_global_features': True,
                   'channel_independent': False, 'use_sliding_window': False}
    fourier_params = {'low_rank': 5, 'output_format': 'signal', 'compute_heuristic_representation': True,
                      'approximation': 'smooth', 'threshold': 0.9, 'sampling_rate': 64e3}
    wavelet_params = {'n_components': 3, 'wavelet': 'bior3.7', 'compute_heuristic_representation': True}
    rocket_params = {"num_features": 200}
    sampling_dict = dict(samples=dict(start_idx=0,
                                      end_idx=None),
                         channels=dict(start_idx=0,
                                       end_idx=None),
                         elements=dict(start_idx=0,
                                       end_idx=None))
    feature_generator = {
        # 'minirocket': [('minirocket_extractor', rocket_params)],
        'stat_generator': [('quantile_extractor', stat_params)],
        'fourier': [('fourier_basis', fourier_params)],
        'wavelet': [('wavelet_basis', wavelet_params)],
    }
    return feature_generator, sampling_dict


feature_generator, sampling_dict = create_feature_generator_strategy()

INDUSTRIAL_PARAMS = {'feature_generator': feature_generator,
                     'data_type': 'tensor',
                     'learning_strategy': 'all_classes',
                     'head_model': 'rf',
                     'sampling_strategy': sampling_dict
                     }

# DEFINE ALL CONFIG FOR API
AUTOML_LEARNING_STRATEGY = DEFAULT_AUTOML_LEARNING_CONFIG
COMPUTE_CONFIG = DEFAULT_COMPUTE_CONFIG
AUTOML_CONFIG = {'task': 'classification',
                 'use_automl': True,
                 'optimisation_strategy': {'optimisation_strategy': {'mutation_agent': 'bandit',
                                                                     'mutation_strategy': 'growth_mutation_strategy'},
                                           'optimisation_agent': 'Industrial'}}
LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                   'optimisation_loss': {'quality_loss': 'f1'}}
INDUSTRIAL_CONFIG = {'problem': 'classification',
                     'strategy': 'kernel_automl',
                     'strategy_params': INDUSTRIAL_PARAMS
                     }
API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
              'automl_config': AUTOML_CONFIG,
              'learning_config': LEARNING_CONFIG,
              'compute_config': COMPUTE_CONFIG}

dataset_name = 'Lightning7'
dataset = load_data()
metric_names = ('f1', 'accuracy')
api_config = dict(
    problem='classification',
    metric='f1',
    timeout=5,
    n_jobs=2,
    with_tuning=False,
    industrial_strategy='kernel_automl',
    industrial_strategy_params={
        'industrial_task': 'classification',
        'data_type': 'tensor',
        'learning_strategy': 'all_classes',
        'head_model': 'rf'
    },
    logging_level=20)

if __name__ == "__main__":
    industrial = ApiTemplate(api_config=API_CONFIG,
                              metric_list=metric_names).eval(dataset=dataset)
    industrial.fit(dataset.get("train_data"))
    predict = industrial.predict(dataset.get("test_data"), 'ensemble')
    predict_proba = industrial.predict_proba(dataset.get("test_data"), 'ensemble')
    metric = industrial.get_metrics(target=dataset.get("test_data")[1],
                                    metric_names=metric_names)
