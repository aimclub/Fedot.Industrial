from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.data.multi_modal import MultiModalData


class RAFensembler:
    def __init__(self, composing_params,
                 ensemble_type: str = 'random_automl_forest',
                 n_splits: int = None,
                 batch_size: int = 1000):
        problem_dict = {'regression': 'fedot_regr',
                        'classification': 'fedot_cls'}
        ensemble_dict = {'random_automl_forest': self._raf_ensemble
                         }
        task_dict = {'classification': Task(TaskTypesEnum.classification),
                     'regression': Task(TaskTypesEnum.regression)}
        head_dict = {'classification': 'logit',
                     'regression': 'ridge'}

        self.task = task_dict[composing_params['problem']]
        self.atomized_automl = problem_dict[composing_params['problem']]
        self.ensemble_method = ensemble_dict[ensemble_type]
        self.atomized_automl_params = composing_params
        self.head = head_dict[composing_params['problem']]
        self.batch_size = batch_size
        if n_splits is None:
            self.n_splits = n_splits
        else:
            self.n_splits = n_splits
        del self.atomized_automl_params['available_operations']

    def fit(self, train_data):
        if self.n_splits is None:
            self.n_splits = round(train_data.features.shape[0]/self.batch_size)
        new_features = np.array_split(train_data.features, self.n_splits)
        new_target = np.array_split(train_data.target, self.n_splits)
        self.current_pipeline = self.ensemble_method(new_features, new_target, n_splits=self.n_splits)

    def predict(self, test_data):
        data_dict = {}
        for i in range(self.n_splits):
            data_dict.update({f'data_source_img/{i}': test_data})
        test_multimodal = MultiModalData(data_dict)
        return self.current_pipeline.predict(test_multimodal).predict

    def _raf_ensemble(self, features, target, n_splits):
        raf_ensemble = PipelineBuilder()
        data_dict = {}
        for i, data_fold_features, data_fold_target in zip(range(n_splits), features, target):
            train_fold = InputData(idx=np.arange(0, len(data_fold_features)),
                                   features=data_fold_features,
                                   target=data_fold_target,
                                   task=self.task,
                                   data_type=DataTypesEnum.image)

            raf_ensemble.add_node(f'data_source_img/{i}', branch_idx=i).add_node(
                self.atomized_automl,
                params=self.atomized_automl_params,
                branch_idx=i)
            data_dict.update({f'data_source_img/{i}': train_fold})
        train_multimodal = MultiModalData(data_dict)
        raf_ensemble = raf_ensemble.join_branches(self.head).build()
        raf_ensemble.fit(input_data=train_multimodal)
        return raf_ensemble
