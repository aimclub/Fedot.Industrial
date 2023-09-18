from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader

if __name__ == "__main__":
    dataset_name = 'ApplianceEnergy'
    industrial = FedotIndustrial(task='ts_regression',
                                 dataset=dataset_name,
                                 strategy='quantile',
                                 explained_variance=0.9,
                                 metric='rmse',
                                 use_cache=True,
                                 timeout=1,
                                 n_jobs=2,
                                 logging_level=20)

    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()

    model = industrial.fit(features=train_data[0], target=train_data[1])

    labels = industrial.predict(features=test_data[0], target=test_data[1])

    metric = industrial.get_metrics(target=test_data[1], metric_names=['rmse'])


