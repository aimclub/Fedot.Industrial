import numpy as np
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.api.utils.data import init_input_data
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.tools.loader import DataLoader

if __name__ == "__main__":
    dataset_name = 'Earthquakes'
    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()
    input_train_data = init_input_data(train_data[0], train_data[1])
    input_test_data = init_input_data(test_data[0], test_data[1])

    metric_dict = {'accuracy': accuracy_score, 'f1': f1_score, 'roc_auc': roc_auc_score}
    with IndustrialModels():
        pipeline = PipelineBuilder().add_node('recurrence_extractor', params={'window_size': 30,
                                                                              'stride': 5,
                                                                              'image_mode': True}) \
            .add_node('resnet_model', params={'epochs': 50,
                                              'model_name': 'ResNet50one'}) \
            .build()
        pipeline.fit(input_train_data)
        output = pipeline.predict(input_test_data)

        predict = np.array(output.predict.flatten()) + 1

        metrics = {'accuracy': accuracy_score(test_data[1], predict)}
        if len(set(test_data[1])) > 2:
            metrics['f1'] = f1_score(test_data[1], predict)
        else:
            metrics['roc_auc'] = roc_auc_score(test_data[1], predict)

        print(metrics)
        _ = 1
