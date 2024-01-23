from sklearn.metrics import classification_report

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.loader import DataLoader

if __name__ == "__main__":
    dataset_name = 'Libras'
    industrial = FedotIndustrial(problem='classification',
                                 metric='f1',
                                 timeout=1,
                                 n_jobs=2,
                                 logging_level=20)

    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()

    model = industrial.fit(train_data)

    labels = industrial.predict(test_data)

    #industrial.finetune(train_data)
    print(classification_report(test_data[1], labels, digits=4))
    industrial.predict(test_data)

