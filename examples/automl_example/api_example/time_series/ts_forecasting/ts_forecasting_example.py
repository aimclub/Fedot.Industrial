from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.loader import DataLoader

if __name__ == "__main__":
    dataset_name = 'D1317'
    benchmark = 'M4'
    horizon = 14
    finetune = False
    initial_assumption = PipelineBuilder().add_node('eigen_basis',
                                                    params={'low_rank_approximation': False,
                                                            'rank_regularization': 'explained_dispersion'}).add_node(
        'ar')

    industrial = FedotIndustrial(problem='ts_forecasting',
                                 metric='rmse',
                                 task_params={'forecast_length': horizon},
                                 timeout=5,
                                 with_tuning=False,
                                 initial_assumption=initial_assumption,
                                 n_jobs=2,
                                 logging_level=10)

    train_data, test_data = DataLoader(dataset_name=dataset_name).load_forecast_data(folder=benchmark)

    if finetune:
        model = industrial.finetune(train_data)
    else:
        model = industrial.fit(train_data)

    labels = industrial.predict(test_data)
    probs = industrial.predict_proba(test_data)
    metrics = industrial.get_metrics(target=test_data[1],
                                     rounding_order=3,
                                     metric_names=['f1', 'accuracy', 'precision', 'roc_auc'])
    print(metrics)
    _ = 1
