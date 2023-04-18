import numpy as np
import pandas as pd
from MKLpy.algorithms import MEMO, CKA
from MKLpy.scheduler import ReduceOnWorsening
from MKLpy.callbacks import EarlyStopping
from fedot_ind.core.architecture.pipelines.classification import ClassificationPipelines
from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
from scipy.spatial.distance import pdist, squareform
from fedot_ind.core.architecture.settings.pipeline_factory import KernelFeatureGenerator


class KernelEnsembler(ClassificationPipelines):
    def __init__(self, train_data, test_data):
        super().__init__(train_data, test_data)
        self.n_classes = np.unique(self.train_target)
        if self.n_classes.shape[0] > 2:
            self.multiclass_strategy = 'ova'
            self.multiclass = True
        else:
            self.multiclass_strategy = 'ovr'
            self.multiclass = False
        self.kernel_params_list = {i.name: i.value for i in KernelFeatureGenerator}

    def __call__(self, pipeline_type: str = 'one_stage_kernel'):
        pipeline_dict = {'one_stage_kernel': self.__one_stage_kernel,
                         'two_stage_kernel': self.__two_stage_kernel
                         }
        return pipeline_dict[pipeline_type]

    def __convert_weights(self, kernel_model):
        kernels_weights_by_class = []
        if not self.multiclass:
            kernels_weights_by_class.append(kernel_model.solution.weights.cpu().detach().numpy())
        else:
            for n_class in self.n_classes:
                kernels_weights_by_class.append(kernel_model.solution[n_class].weights.cpu().detach().numpy())
        kernel_weight_matrix = pd.DataFrame(kernels_weights_by_class).apply(lambda x: abs(x))
        return kernel_weight_matrix

    def transform(self, kernel_params_dict: dict = None, feature_generator: str = None):
        feature_matrix = []
        if kernel_params_dict is None:
            kernel_params_dict = self.kernel_params_list
        if feature_generator is not None:
            kernel_params_dict = {feature_generator: kernel_params_dict[feature_generator]}

        for feature_generator, kernel_params in kernel_params_dict.items():
            for specified_params in kernel_params:
                feature_extractor, classificator, lambda_func_dict = self._init_pipeline_nodes(**specified_params)

                feature_matrix.append(self.get_feature_generator(input_data=self.train_features,
                                                                 steps=lambda_func_dict,
                                                                 pipeline_type='SpecifiedFeatureGeneratorTSC').value[0])
        return feature_matrix

    def __one_stage_kernel(self, kernel_params_dict: dict = None, feature_generator: str = None):
        feature_matrix = self.transform(kernel_params_dict, feature_generator)
        KLtr = [squareform(pdist(X=feature, metric='cosine')) for feature in feature_matrix]
        mkl = CKA(multiclass_strategy=self.multiclass_strategy).fit(KLtr, self.train_target)
        #mkl = FHeuristic(multiclass_strategy=self.multiclass_strategy).fit(KLtr, self.train_target)
        kernel_weight_matrix = self.__convert_weights(mkl)
        return kernel_weight_matrix

    def __two_stage_kernel(self, kernel_params_dict: dict = None, feature_generator: str = None):
        feature_matrix = self.transform(kernel_params_dict, feature_generator)
        KLtr = [squareform(pdist(X=feature, metric='cosine')) for feature in feature_matrix]
        earlystop = EarlyStopping(
            KLtr,
            self.train_target,  # validation data, KL is a validation kernels list
            patience=5,  # max number of acceptable negative steps
            cooldown=1,  # how ofter we run a measurement, 1 means every optimization step
            metric='roc_auc',  # the metric we monitor
        )

        mkl = MEMO(multiclass_strategy='ova',
                   max_iter=1000,
                   learning_rate=.1,
                   callbacks=[earlystop],
                   scheduler=ReduceOnWorsening()).fit(KLtr, self.train_target)
        kernel_weight_matrix = self.__convert_weights(mkl)
        return kernel_weight_matrix


if __name__ == '__main__':
    dataset_name = 'Earthquakes'
    train, test = DataLoader(dataset_name).load_data()
    kernels = KernelEnsembler(train_data=train, test_data=test)
    kernel_list = {'wavelet': [
        {'feature_generator_type': 'wavelet',
         'feature_hyperparams': {
             'wavelet': "mexh",
             'n_components': 2
         }},
        {'feature_generator_type': 'wavelet',
         'feature_hyperparams': {
             'wavelet': "morl",
             'n_components': 2
         }}]}
    set_of_fg = kernels('two_stage_kernel')(feature_generator='wavelet')

    # model_hyperparams = {
    #     'problem': 'classification',
    #     'seed': 42,
    #     'timeout': 4,
    #     'max_depth': 6,
    #     'max_arity': 3,
    #     'cv_folds': 3,
    #     'logging_level': 20,
    #     'n_jobs': 4
    # }
    # feature_hyperparams = {
    #     'window_mode': True,
    #     'window_size': 10
    # }
    # model_stats, prediction_stats = pipelines('SpecifiedFeatureGeneratorTSC')(feature_generator_type='statistical',
    #                                                                           model_hyperparams=model_hyperparams,
    #                                                                           feature_hyperparams=feature_hyperparams)
    # feature_hyperparams = {
    #     'wavelet': "morl"
    # }
    # model_wavelet, prediction_wavelet = pipelines('SpecifiedFeatureGeneratorTSC')(feature_generator_type='wavelet',
    #                                                                               model_hyperparams=model_hyperparams,
    #                                                                               feature_hyperparams=feature_hyperparams)

    # metrics = PerformanceAnalyzer().calculate_metrics(target=pipelines.test_target,
    #                                                   predicted_labels=prediction_stats['predicted_labels'],
    #                                                   predicted_probs=prediction_stats['predicted_probs'])
