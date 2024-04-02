from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.loader import DataLoader

if __name__ == "__main__":
    dataset_name = 'IEEEPPG' #BeijingPM10Quality
    finetune = True
    initial_assumption = PipelineBuilder().add_node('channel_filtration').add_node('quantile_extractor').add_node('treg')

    industrial = FedotIndustrial(problem='regression',
                                 metric='rmse',
                                 timeout=5,
                                 initial_assumption=initial_assumption,
                                 n_jobs=2,
                                 logging_level=20)

    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()
    if finetune:
        model = industrial.finetune(train_data)
    else:
        model = industrial.fit(train_data)

    labels = industrial.predict(test_data)
    probs = industrial.predict_proba(test_data)
    metrics = industrial.get_metrics(target=test_data[1],
                                     rounding_order=3,
                                     metric_names=('r2', 'rmse', 'mae'))
    print(metrics)
    _ = 1
