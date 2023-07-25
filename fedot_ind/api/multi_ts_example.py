import sys

import numpy as np
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
from fedot_ind.core.models.statistical.StatsExtractor import StatsExtractor
from fedot_ind.core.operation.transformation.basis.data_driven import DataDrivenBasisImplementation
from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementation
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score, f1_score, max_error, mean_absolute_error, mean_gamma_deviance, \
    mean_poisson_deviance, \
    mean_squared_error, \
    mean_squared_log_error, \
    mean_tweedie_deviance, median_absolute_error, r2_score, \
    roc_auc_score


def init_input(X, y):
    input_data = InputData(idx=np.arange(len(X)),
                           features=np.array(X.values.tolist()),
                           target=y.reshape(-1, 1),
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.image)
    return input_data


if __name__ == "__main__":
    # dataset_name = 'BenzeneConcentration'
    # ddb_features_train = DataDrivenBasisImplementation({'window_size': 30,
    #                                                     'sv_selector': 'median'}).transform(train_input_data)
    # fourier_features_train = FourierBasisImplementation({"spectrum_type": "smoothed",
    #                                                      "threshold": 20000}).transform(train_input_data)
    # fourier_features_test = FourierBasisImplementation({"spectrum_type": "smoothed",
    #                                                     "threshold": 20000}).transform(test_input_data)

    dataset_name = 'LiveFuelMoistureContent'
    pca_n_components = 300

    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()
    train_target = np.array([float(i) for i in train_data[1]])
    test_target = np.array([float(i) for i in test_data[1]])
    train_input_data = init_input(train_data[0], train_target)
    test_input_data = init_input(test_data[0], test_target)

    extractor = StatsExtractor({'window_mode': False,
                                'window_size': None,
                                'var_threshold': 0})
    pca = PCA(n_components=pca_n_components,
              svd_solver='full')

    extracted_features_train = extractor.transform(train_input_data)
    train_size = extracted_features_train.features.shape
    train_features = extracted_features_train.features.reshape(train_size[0], train_size[1] * train_size[2])
    train_features = pca.fit_transform(train_features)

    extracted_features_test = extractor.transform(test_input_data)
    test_size = extracted_features_test.features.shape
    test_features = extracted_features_test.features.reshape(test_size[0], test_size[1] * test_size[2])
    test_features = pca.transform(test_features)

    predictor = Fedot(problem='regression', timeout=10, logging_level=20, metric='rmse')
    model = predictor.fit(features=train_features, target=train_target)
    labels = predictor.predict(features=test_features)

    print('r2_score:', r2_score(test_target, labels))
    print('mean_squared_error:', mean_squared_error(test_target, labels))
    print('root_mean_squared_error:', np.sqrt(mean_squared_error(test_target, labels)))
    print('mean_absolute_error', mean_absolute_error(test_target, labels))
    print('median_absolute_error', median_absolute_error(test_target, labels))
    print('mean_squared_log_error', mean_squared_log_error(test_target, labels))
    print('explained_variance_score', explained_variance_score(test_target, labels))
    print('max_error', max_error(test_target, labels))
    print('mean_poisson_deviance', mean_poisson_deviance(test_target, labels))
    print('mean_gamma_deviance', mean_gamma_deviance(test_target, labels))
    print('mean_tweedie_deviance', mean_tweedie_deviance(test_target, labels))
    print('mean_tweedie_deviance_power_1.5:', mean_tweedie_deviance(test_target, labels, power=1.5))
    print('mean_tweedie_deviance_power_2:', mean_tweedie_deviance(test_target, labels, power=2))
    print('mean_tweedie_deviance_power_3;', mean_tweedie_deviance(test_target, labels, power=3))

    _ = 1
