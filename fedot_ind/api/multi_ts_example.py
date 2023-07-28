import numpy as np
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
from fedot_ind.core.models.quantile.quantile_extractor import QuantileExtractor


def init_input(X, y):
    input_data = InputData(idx=np.arange(len(X)),
                           features=np.array(X.values.tolist()),
                           target=y.reshape(-1, 1),
                           task=Task(TaskTypesEnum.regression),
                           data_type=DataTypesEnum.image)
    return input_data


if __name__ == "__main__":

    components = [0.90, 0.92, 0.94, 0.96, 0.98]
    metrics = []
    for comps in components:
        dataset_name = 'AppliancesEnergy'
        pca_n_components = comps

        train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()
        train_target = np.array([float(i) for i in train_data[1]])
        test_target = np.array([float(i) for i in test_data[1]])
        train_input_data = init_input(train_data[0], train_target)
        test_input_data = init_input(test_data[0], test_target)

        extractor = QuantileExtractor({'window_mode': False,
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

        root_mean_squared_error = np.sqrt(mean_squared_error(test_target, labels))
        metrics.append(root_mean_squared_error)

    import matplotlib.pyplot as plt

    plt.plot(components, metrics)
    plt.xlabel('explained variance')
    plt.ylabel('rmse')
    plt.show()

    _ = 1
