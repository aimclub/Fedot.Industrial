from fedot.core.pipelines.pipeline_builder import PipelineBuilder


class RAFEnsembler:
    def __init__(self, composing_params, ensemble_type: str = 'random_automl_forest'):
        problem_dict = {'regression': 'fedot_regr',
                        'classification': 'fedot_cls'}
        ensemble_dict = {'random_automl_forest': self.__raf_ensemle,
                         'two_stage_kernel': self.__two_stage_kernel
                         }
        self.atomized_automl = problem_dict[composing_params['problem']]
        self.ensemble_method = ensemble_dict[ensemble_type]
        self.atomized_automl_params = composing_params

    def fit(self, train_data):
        pass

    def predict(self, test_data):
        pass

    def __raf_ensemble(self, chunks):
        raf_ensemble = PipelineBuilder()
        for i in range(chunks):
            raf_ensemble.add_node(self.atomized_automl, params=self.atomized_automl_params, branch_idx=i)
        raf_ensemble.join_branches('logit')
        return raf_ensemble
