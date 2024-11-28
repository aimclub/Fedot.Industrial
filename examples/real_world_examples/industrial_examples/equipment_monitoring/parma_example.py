import gc

import numpy as np

from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.operation.transformation.data.park_transformation import park_transform

gc.collect()
metric_names = ('f1', 'accuracy', 'precision', 'roc_auc')
take_only_inst_phase = False
sampling_window = 4000
if take_only_inst_phase:
    take_only_inst_phase = 4
else:
    take_only_inst_phase = 1
stat_params = {'window_size': 10, 'stride': 1, 'add_global_features': True, 'use_sliding_window': False}
fourier_params = {'low_rank': 5, 'output_format': 'signal', 'approximation': 'smooth', 'threshold': 0.9}
wavelet_params = {'n_components': 3, 'wavelet': 'bior3.7'}
park_params = {}
feature_generator = {
    # 'fourier+stat': [('fourier_basis', fourier_params), ('quantile_extractor', stat_params)],
    'wavelet+stat': [('wavelet_basis', wavelet_params), ('quantile_extractor', stat_params)],
    'stat_generator': [('quantile_extractor', stat_params)]}

if __name__ == "__main__":
    finetune = False

    train_features, train_target = park_transform(np.load('./X_train.npy').swapaxes(1, 2))[:, take_only_inst_phase:,
                                   :sampling_window] \
        , np.load('./y_train.npy')
    test_features, test_target = park_transform(np.load('./X_test.npy').swapaxes(1, 2))[:, take_only_inst_phase:,
                                 :sampling_window] \
        , np.load('./y_test.npy')
    input_train = (train_features, train_target)
    input_test = (test_features, test_target)

    dataset = dict(test_data=input_test, train_data=input_train)

    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=120,
                      pop_size=5,
                      early_stopping_iterations=20,
                      early_stopping_timeout=100,
                      with_tunig=False,
                      preset='classification_tabular',
                      industrial_strategy_params={'feature_generator': feature_generator,
                                                  'data_type': 'tensor',
                                                  'learning_strategy': 'ts2tabular',
                                                  },
                      n_jobs=2,
                      logging_level=20)

    result_dict = ApiTemplate(api_config=api_config,
                              metric_list=metric_names).eval(dataset=dataset,
                                                             finetune=finetune)
    metrics = result_dict['metrics']
    _ = 1
