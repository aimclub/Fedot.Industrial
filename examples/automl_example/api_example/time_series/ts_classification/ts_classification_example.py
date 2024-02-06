from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.loader import DataLoader

if __name__ == "__main__":
    dataset_name = 'Epilepsy'
    industrial = FedotIndustrial(problem='classification',
                                 metric='f1',
                                 timeout=5,
                                 n_jobs=2,
                                 logging_level=20)

    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()

    model = industrial.fit(train_data)

    labels = industrial.predict(test_data)
    probs = industrial.predict_proba(test_data)
    metrics = industrial.get_metrics(target=test_data[1],
                                     rounding_order=3,
                                     metric_names=['f1', 'accuracy', 'precision', 'roc_auc'])
    # industrial.finetune(train_data)
    print(metrics)
