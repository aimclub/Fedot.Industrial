import gc

import matplotlib
import numpy as np
from sklearn.utils import shuffle

from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.operation.transformation.data.park_transformation import park_transform

matplotlib.use('TkAgg')
gc.collect()
metric_names = ('f1', 'accuracy', 'precision', 'roc_auc')
stat_params = {'window_size': 0, 'stride': 1, 'add_global_features': True, 'use_sliding_window': False}
fourier_params = {'low_rank': 5, 'output_format': 'signal', 'compute_heuristic_representation': True,
                  'approximation': 'smooth', 'threshold': 0.9, 'sampling_rate': 64e3}
wavelet_params = {'n_components': 3, 'wavelet': 'bior3.7', 'compute_heuristic_representation': True}
park_params = {}
rocket_params = {"num_features": 200}
sampling_dict = dict(samples=dict(start_idx=0,
                                  end_idx=None),
                     channels=dict(start_idx=0,
                                   end_idx=None),
                     elements=dict(start_idx=0,
                                   end_idx=None))

feature_generator = {
    # 'minirocket': [('minirocket_extractor', rocket_params)],
    # 'stat_generator': [('quantile_extractor', stat_params)],
    'fourier': [('fourier_basis', fourier_params)],
    'wavelet': [('wavelet_basis', wavelet_params)],
}


def load_data(use_park_transform: bool = False):
    train_features, train_target = np.load('./dataset/X_train.npy').swapaxes(1, 2), np.load('./dataset/y_train.npy')
    test_features, test_target = np.load('./dataset/X_test.npy').swapaxes(1, 2), np.load('./dataset/y_test.npy')
    train_features, train_target = shuffle(train_features, train_target)
    if use_park_transform:
        train_features, test_features = park_transform(train_features), park_transform(test_features)
    input_train = (train_features, train_target)
    input_test = (test_features, test_target)

    dataset = dict(test_data=input_test, train_data=input_train)
    return dataset


if __name__ == "__main__":
    finetune = False
    dataset = load_data(True)
    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=40,
                      pop_size=10,
                      early_stopping_iterations=10,
                      early_stopping_timeout=30,
                      optimizer_params={'mutation_agent': 'random',
                                        'mutation_strategy': 'params_mutation_strategy'},
                      with_tunig=False,
                      preset='classification_tabular',
                      industrial_strategy_params={'feature_generator': feature_generator,
                                                  'data_type': 'tensor',
                                                  'learning_strategy': 'ts2tabular',
                                                  'sampling_strategy': sampling_dict
                                                  },
                      n_jobs=-1,
                      logging_level=20)

    result_dict = ApiTemplate(api_config=api_config,
                              metric_list=metric_names).eval(dataset=dataset,
                                                             finetune=finetune)
    metrics = result_dict['metrics']
    metrics.to_csv('./metrics.csv')
    hist = result_dict['industrial_model'].save_optimization_history(return_history=True)
    result_dict['industrial_model'].vis_optimisation_history(hist)
    result_dict['industrial_model'].save_best_model()
    _ = 1
