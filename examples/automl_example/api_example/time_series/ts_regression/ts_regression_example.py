from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

if __name__ == "__main__":
    dataset_name = 'AppliancesEnergy'
    industrial = FedotIndustrial(problem='regression',
                                 metric='rmse',
                                 timeout=5,
                                 n_jobs=2,
                                 logging_level=20)

    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()

    model = industrial.fit(train_data)

    y_predicted = industrial.predict(test_data)

    print('Metrics:')
    print(f'RMSE: {round(mean_squared_error(test_data[1], y_predicted, squared=False), 3)}')
    print(f'MAPE: {round(mean_absolute_percentage_error(test_data[1], y_predicted), 3)}')
