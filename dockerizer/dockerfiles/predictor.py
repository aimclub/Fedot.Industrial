import os

import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from core.operation.utils.load_data import DataLoader
from core.operation.utils.utils import PROJECT_PATH

docker_file = os.path.join(PROJECT_PATH, 'dockerizer', 'dockerfiles', 'Dockerfile')
fedot_model_path = os.path.join(PROJECT_PATH, 'dockerizer', 'dockerfiles', 'fedot_model.joblib')
feature_gen_path = os.path.join(PROJECT_PATH, 'dockerizer', 'dockerfiles', 'feature_generator.joblib')


def get_score(y_test, predict):
    metrics_dict = {'roc_auc': roc_auc_score,
                    'f1': f1_score,
                    'accuracy': accuracy_score,
                    'precision': precision_score,
                    'recall': recall_score}

    scores = {}
    for metric_name, metric_func in metrics_dict.items():
        try:
            scores[metric_name] = metric_func(y_test, predict)
        except ValueError:
            print(f'ValueError for {metric_name}')
            scores[metric_name] = None

    return scores


def proba_to_vector(matrix, dummy_flag=True):
    """
    Converts matrix of probabilities to vector of labels.

    :param matrix: np.ndarray with probabilities
    :param dummy_flag: ...
    :return: np.ndarray with labels
    """
    if not dummy_flag:
        dummy_val = -1
    else:
        dummy_val = 0
    if len(matrix.shape) > 1:
        vector = np.array([x.argmax() + x[x.argmax()] + dummy_val for x in matrix])
        return vector
    return matrix


def get_predictions():
    fedot_pipeline = load(fedot_model_path)
    feature_generator = load(feature_gen_path)
    train_data, test_data = DataLoader('Earthquakes').load_data()
    (X_train, y_train), (X_test, y_test) = train_data, test_data

    input_data_train = InputData(idx=np.arange(0, len(X_train)), features=X_train, target=y_train,
                                 task=Task(TaskTypesEnum.classification),
                                 data_type=DataTypesEnum.table)
    fedot_pipeline.fit(input_data_train)

    input_data_test = InputData(idx=np.arange(0, len(X_test)), features=X_test, target=y_test,
                                task=Task(TaskTypesEnum.classification),
                                data_type=DataTypesEnum.table)
    predict = fedot_pipeline.predict(input_data_test)

    predict = proba_to_vector(predict.predict)
    score = get_score(y_test, predict)
    print(predict)
    print(score)


if __name__ == "__main__":
    get_predictions()

# data_path = os.path.join(PROJECT_PATH, 'data', 'Beef')

# def booster_pipeline(booster_models, input_data, ensemble_model):
#     booster_predicts = list()
#
#     for model in booster_models:
#         predict = model.predict(input_data)
#         booster_predicts.append(predict)
#
#     if ensemble_model:
#         boosting_predict_list = pd.DataFrame(i.reshape(-1) for i in booster_predicts).T
#     else:
#         boosting_predict_list = [np.array(_) for _ in booster_predicts]


# def _pipeline(predictor, ecm_models, data):
#     fedot_predict = predictor.predict(data)
#     # booster_models = ecm_models[:-1]
#     # ensemble_model = ecm_models[-1]
#     # for model in booster_models:
