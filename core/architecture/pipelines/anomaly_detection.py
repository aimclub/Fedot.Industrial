from functools import partial

import numpy as np
import pandas as pd
from pymonad.list import ListMonad
from pymonad.either import Right
from core.architecture.pipelines.abstract_pipeline import AbstractPipelines
from core.architecture.preprocessing.DatasetLoader import DataLoader
from core.models.detection.subspaces.func_pca import FunctionalPCA
from core.operation.transformation.basis.data_driven import DataDrivenBasis
import os


class AnomalyDetectionPipelines(AbstractPipelines):

    def __call__(self, pipeline_type: str = 'SpecifiedFeatureGeneratorTSC'):
        pipeline_dict = {'SST': self.__singular_transformation_pipeline,
                         'FunctionalPCA': self.__functional_pca_pipeline,
                         'Kalman': self.__kalman_filter_pipeline,
                         'VectorAngle': self.__vector_based_pipeline,
                         }
        return pipeline_dict[pipeline_type]

    def get_feature_generator(self, **kwargs):
        lambda_func_dict = kwargs['steps']
        lambda_func_dict['concatenate_features'] = lambda x: list_of_features.append(pd.DataFrame(x))
        input_data = kwargs['input_data']
        list_of_features = []
        if kwargs['pipeline_type'] == 'SST':
            pipeline = Right(input_data)
        elif kwargs['pipeline_type'] == 'FunctionalPCA':
            pipeline = Right(input_data).then(lambda_func_dict['create_list_of_ts']).map(
                lambda_func_dict['transform_to_basis']).map(lambda_func_dict['reduce_basis']).then(
                lambda_func_dict['concatenate_features']).insert(
                self._get_feature_matrix(list_of_features=list_of_features, mode='1D'))
        elif kwargs['pipeline_type'] == 'Kalman':
            pipeline = Right(input_data)
        elif kwargs['pipeline_type'] == 'VectorAngle':
            pipeline = Right(input_data)
        return pipeline

    def __singular_transformation_pipeline(self):
        pass

    def __multi_functional_pca(self):
        pass

    def __functional_pca_pipeline(self, **kwargs):
        feature_extractor, detector, lambda_func_dict = self._init_pipeline_nodes(model_type='functional_pca',
                                                                                  **kwargs)
        data_basis = DataDrivenBasis()
        lambda_func_dict['transform_to_basis'] = lambda x: self.basis if self.basis \
                                                                         is not None else data_basis.fit(x,
                                                                                                         window_length=None)
        lambda_func_dict['reduce_basis'] = lambda x: x[:, 0] if kwargs['component'] is None \
            else x[:, kwargs['component']]

        train_feature_generator_module = self.get_feature_generator(input_data=self.train_features,
                                                                    steps=lambda_func_dict,
                                                                    pipeline_type='FunctionalPCA')
        detector.fit(train_feature_generator_module.value[0])
        detector.feature_generator = partial(self.get_feature_generator, steps=lambda_func_dict,
                                             pipeline_type='FunctionalPCA')
        return detector

    def __kalman_filter_pipeline(self):
        pass

    def __vector_based_pipeline(self):
        pass


if __name__ == "__main__":
    # benchmark files checking
    all_files = []
    model_hyperparams = {
        'n_components': 2,
        'regularization': None,
        'basis_function': None
    }
    feature_hyperparams = {
        'window_mode': True,
        'window_size': 10
    }
    dict_result = {}

    for root, dirs, files in os.walk(r"D:\РАБОТЫ РЕПОЗИТОРИИ\Репозитории\Industiral\IndustrialTS\data\SKAB"):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))

    # datasets with anomalies loading
    list_of_df = [pd.read_csv(file,
                              sep=';',
                              index_col='datetime',
                              parse_dates=True) for file in all_files if 'anomaly-free' not in file]
    # anomaly-free df loading
    anomaly_free_df = pd.read_csv([file for file in all_files if 'anomaly-free' in file][0],
                                  sep=';',
                                  index_col='datetime',
                                  parse_dates=True)
    window_length = 30
    train = anomaly_free_df
    sample = list_of_df[0]
    start = 30
    sample = sample[['Volume Flow RateRMS', 'Temperature', 'Pressure']]
    train = (sample.iloc[:start, :].T, sample.iloc[:30, -1].values)
    test = (None, None)
    # dataset_name = 'Chinatown'
    # train, test = DataLoader(dataset_name).load_data()
    # train_target = train[1]
    # test_target = test[1]
    pipeline_detect = AnomalyDetectionPipelines(train_data=train, test_data=test)

    model_1 = pipeline_detect('FunctionalPCA')(component=None,
                                               model_hyperparams=model_hyperparams,
                                               feature_hyperparams=feature_hyperparams)
    idx_dict = {}
    import matplotlib.pyplot as plt

    pd.DataFrame(model_1.components_.T).plot()
    plt.show()
    for slice in range(start, sample.shape[0] - window_length, window_length):
        next = window_length + slice
        basis_features = model_1.feature_generator(input_data=sample.iloc[slice:next, :].T).value[0]
        projection, out_idx = model_1.predict(basis_features, threshold=0.80)
        pd.DataFrame(projection).plot()
        plt.show()
        idx_dict.update({f'{slice}:{window_length + slice} sec': out_idx})
        print(slice, next)
    _ = 1
