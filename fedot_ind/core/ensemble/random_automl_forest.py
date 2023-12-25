import numpy as np
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from sklearn.model_selection import KFold
from fedot.core.data.multi_modal import MultiModalData


class RAFensembler:
    def __init__(self, composing_params, ensemble_type: str = 'random_automl_forest'):
        problem_dict = {'regression': 'fedot_regr',
                        'classification': 'fedot_cls'}
        ensemble_dict = {'random_automl_forest': self._raf_ensemble
                         }
        task_dict = {'classification': Task(TaskTypesEnum.classification),
                     'regression': Task(TaskTypesEnum.regression)}
        self.task = task_dict[composing_params['problem']]
        self.atomized_automl = problem_dict[composing_params['problem']]
        self.ensemble_method = ensemble_dict[ensemble_type]
        self.atomized_automl_params = composing_params

    def fit(self, train_data, n_splits=5):
        new_features = np.split(train_data.features, n_splits)
        new_target = np.split(train_data.target, n_splits)
        self.fitted_ensemble = self.ensemble_method(new_features, new_target, n_splits=n_splits)

    def predict(self, test_data):
        pass

    def _raf_ensemble(self, features, target, n_splits):
        raf_ensemble = PipelineBuilder()
        for i, data_fold_features, data_fold_target in zip(range(n_splits), features, target):
            train_fold = InputData(idx=np.arange(0, len(data_fold_features)),
                                   features=data_fold_features,
                                   target=data_fold_target,
                                   task=self.task,
                                   data_type=DataTypesEnum.image)

            raf_ensemble.add_node('data_source_img', branch_idx=i, params={'data_source_img': train_fold}).add_node(
                self.atomized_automl,
                params=self.atomized_automl_params,
                branch_idx=i)
        raf_ensemble = raf_ensemble.join_branches('logit').build()
        return raf_ensemble.fit()
