from fedot_ind.core.architecture.settings.computational import backend_methods as np
import pandas as pd
from MKLpy.algorithms import FHeuristic, RMKL
from MKLpy.callbacks import EarlyStopping
from MKLpy.scheduler import ReduceOnWorsening
from scipy.spatial.distance import pdist, squareform

from fedot_ind.core.architecture.pipelines.classification import ClassificationPipelines
from fedot_ind.core.architecture.settings.pipeline_factory import KernelFeatureGenerator


class KernelEnsembler(ClassificationPipelines):
    def __init__(self, train_data, test_data):
        super().__init__(train_data, test_data)
        self.n_classes = np.unique(self.train_target)
        self.feature_matrix_train = []
        self.feature_matrix_test = []
        if self.n_classes.shape[0] > 2:
            self.multiclass_strategy = 'ova'
            self.multiclass = True
        else:
            self.multiclass_strategy = 'ovr'
            self.multiclass = False
        self.kernel_params_list = {
            i.name: i.value for i in KernelFeatureGenerator}

    def __call__(self, pipeline_type: str = 'one_stage_kernel'):
        pipeline_dict = {'one_stage_kernel': self.__one_stage_kernel,
                         'two_stage_kernel': self.__two_stage_kernel
                         }
        return pipeline_dict[pipeline_type]

    def __convert_weights(self, kernel_model):
        kernels_weights_by_class = []
        if not self.multiclass:
            kernels_weights_by_class.append(
                kernel_model.solution.weights.cpu().detach().numpy())
        else:
            for n_class in self.n_classes:
                kernels_weights_by_class.append(
                    kernel_model.solution[n_class].weights.cpu().detach().numpy())
        return pd.DataFrame(kernels_weights_by_class)

    def transform(self, kernel_params_dict: dict = None, feature_generator: str = None):

        if kernel_params_dict is None:
            kernel_params_dict = self.kernel_params_list
        if feature_generator is not None:
            kernel_params_dict = {
                feature_generator: kernel_params_dict[feature_generator]}

        for feature_generator, kernel_params in kernel_params_dict.items():
            for specified_params in kernel_params:
                feature_extractor, classificator, lambda_func_dict = self._init_pipeline_nodes(
                    **specified_params)

                self.feature_matrix_train.append(feature_extractor.extract_features(
                    self.train_features, self.train_target))
                self.feature_matrix_test.append(
                    feature_extractor.extract_features(self.test_features, self.test_target))
        return

    def __one_stage_kernel(self, kernel_params_dict: dict = None, feature_generator: str = None):
        self.transform(kernel_params_dict, feature_generator)
        KLtr = [squareform(pdist(X=feature, metric='cosine'))
                for feature in self.feature_matrix_train]
        # mkl = CKA(multiclass_strategy=self.multiclass_strategy).fit(KLtr, self.train_target)
        mkl = FHeuristic(multiclass_strategy=self.multiclass_strategy).fit(
            KLtr, self.train_target)
        # mkl = EasyMKL(multiclass_strategy=self.multiclass_strategy).fit(KLtr, self.train_target)
        kernel_weight_matrix = self.__convert_weights(mkl)
        return kernel_weight_matrix

    def __two_stage_kernel(self, kernel_params_dict: dict = None, feature_generator: str = None):
        self.transform(kernel_params_dict, feature_generator)
        KLtr = [squareform(pdist(X=feature, metric='cosine'))
                for feature in self.feature_matrix_train]
        earlystop = EarlyStopping(
            KLtr,
            self.train_target,  # validation data, KL is a validation kernels list
            patience=5,  # max number of acceptable negative steps
            cooldown=1,  # how ofter we run a measurement, 1 means every optimization step
            metric='roc_auc',  # the metric we monitor
        )

        mkl = RMKL(multiclass_strategy='ova',
                   max_iter=1000,
                   learning_rate=.1,
                   callbacks=[earlystop],
                   scheduler=ReduceOnWorsening()).fit(KLtr, self.train_target)
        kernel_weight_matrix = self.__convert_weights(mkl)
        return kernel_weight_matrix


def init_kernel_ensemble(train_data,
                         test_data,
                         strategy: str = 'quantile',
                         kernel_list: dict = None):
    kernels = KernelEnsembler(train_data=train_data, test_data=test_data)
    set_of_fg = kernels('one_stage_kernel')(kernel_params_dict=kernel_list)

    train_feats = kernels.feature_matrix_train
    train_target = kernels.train_target

    test_feats = kernels.feature_matrix_test
    test_target = kernels.test_target

    return set_of_fg, train_feats, train_target, test_feats, test_target
