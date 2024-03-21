import numpy as np
import pandas as pd
from pyriemann.estimation import XdawnCovariances, Covariances, Shrinkage
from pyriemann.tangentspace import TangentSpace
from typing import Optional
from sklearn.utils.extmath import softmax
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from pyriemann.utils import mean_covariance
from pyriemann.utils.distance import distance

from fedot_ind.core.models.base_extractor import BaseExtractor


class RiemannExtractor(BaseExtractor):
    """Class responsible for riemann tangent space features generator.

    Attributes:
        window_size (int): size of window
        stride (int): stride for window
        var_threshold (float): threshold for variance

    Example:
        To use this class you need to import it and call needed methods::

            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot_ind.tools.loader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('quantile_extractor',
                                                       params={'window_size': 20, 'window_mode': True})
                                            .add_node('rf')
                                            .build()
                input_data = init_input_data(train_data[0], train_data[1])
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        extraction_dict = {'mdm': self.extract_centroid_distance,
                           'tangent': self.extract_riemann_features,
                           'ensemble': self._ensemble_features}

        self.n_filter = params.get('nfilter', 2)
        self.estimator = params.get('estimator', 'scm')
        self.covariance_metric = params.get('SPD_metric', 'riemann')
        self.distance_metric = params.get('tangent_metric', 'riemann')
        self.extraction_strategy = params.get('extraction_strategy ', 'ensemble')

        self.covarince_transformer = params.get('SPD_space', None)
        self.tangent_projector = params.get('tangent_space', None)
        if np.any([self.covarince_transformer, self.tangent_projector]) is None:
            self._init_spaces()
            self.fit_stage = True
        self.extraction_func = extraction_dict[self.extraction_strategy]

        self.logging_params.update({
            'estimator': self.estimator,
            'tangent_space_metric': self.distance_metric,
            'SPD_space_metric': self.covariance_metric})

    def _init_spaces(self):
        self.covarince_transformer = Covariances(estimator='scm')
        self.tangent_projector = TangentSpace(metric=self.distance_metric)
        self.shinkage = Shrinkage()

    def extract_riemann_features(self, input_data: InputData) -> InputData:
        if not self.fit_stage:
            SPD = self.covarince_transformer.transform(input_data.features)
            SPD = self.shinkage.transform(SPD)
            ref_point = self.tangent_projector.transform(SPD)
        else:
            SPD = self.covarince_transformer.fit_transform(input_data.features, input_data.target)
            SPD = self.shinkage.fit_transform(SPD)
            ref_point = self.tangent_projector.fit_transform(SPD)
            self.fit_stage = False
        return ref_point

    def extract_centroid_distance(self, input_data: InputData):
        self.classes_ = np.unique(input_data.target)
        if not self.fit_stage:
            SPD = self.covarince_transformer.transform(input_data.features)
            SPD = self.shinkage.transform(SPD)
        else:
            SPD = self.covarince_transformer.fit_transform(input_data.features, input_data.target)
            SPD = self.shinkage.fit_transform(SPD)
        self.covmeans_ = [mean_covariance(SPD[input_data.target.flatten() == ll], metric=self.covariance_metric)
                          for ll in self.classes_]

        n_centroids = len(self.covmeans_)
        dist = [distance(SPD, self.covmeans_[m], self.distance_metric) for m in range(n_centroids)]
        dist = np.concatenate(dist, axis=1)
        feature_matrix = softmax(-dist ** 2)
        return feature_matrix

    def _ensemble_features(self, input_data: InputData):
        tangent_features = self.extract_riemann_features(input_data)
        dist_features = self.extract_centroid_distance(input_data)
        feature_matrix = np.concatenate([tangent_features, dist_features], axis=1)
        return feature_matrix

    def _transform(self, input_data: InputData) -> np.array:
        """
        Method for feature generation for all series
        """

        feature_matrix = self.extraction_func(input_data)
        self.predict = self._clean_predict(feature_matrix)
        return self.predict


if __name__ == "__main__":
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from sklearn.metrics import accuracy_score
    from fedot_ind.api.utils.data import init_input_data
    from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
    from sklearn.model_selection import train_test_split

    path_x = '/Users/technocreep/Desktop/Working-Folder/fedot-industrial/Fedot.Industrial/0_lavence/valence_data/pt1/X_eeg_4_24_old_resave.npy'
    path_y = '/Users/technocreep/Desktop/Working-Folder/fedot-industrial/Fedot.Industrial/0_lavence/valence_data/pt1/y_old.npy'
    X = np.load(path_x)
    y = np.load(path_y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    x_train = pd.DataFrame([[pd.Series(i) for i in k] for k in X_train])
    x_test = pd.DataFrame([[pd.Series(i) for i in k] for k in X_test])

    init_train = init_input_data(x_train, y_train)
    init_test = init_input_data(x_test, y_test)

    with IndustrialModels():
        # riemann_extractor
        # pipeline = PipelineBuilder().add_node('eigen_basis').add_node('quantile_extractor').add_node('rf').build()
        pipeline = PipelineBuilder().add_node('riemann_extractor', params={'n_filter': 3})\
                                    .add_node('quantile_extractor')\
                                    .add_node('mlp')\
                                    .build()
        pipeline.fit(init_train)
        pred = pipeline.predict(init_test)
        acc = accuracy_score(y_test, np.round(pred.predict))
        print(acc)

