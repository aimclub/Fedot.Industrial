from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.loader import DataLoader

if __name__ == "__main__":
    dataset_name = 'AppliancesEnergy'
    industrial = FedotIndustrial(problem='regression',
                                 metric='rmse',
                                 timeout=1,
                                 n_jobs=2,
                                 logging_level=20)

    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()

    model = industrial.fit(train_data)

    labels = industrial.predict(test_data)

    industrial.finetune(train_data)
    industrial.predict(test_data)
