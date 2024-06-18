import pickle

import numpy as np
import pandas as pd

from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate


def create_big_dataset():
    train_X, test_X = np.load(
        './examples/big_dataset/train_airlinescodrnaadult_fold0.npy'), np.load(
        './examples/big_dataset/test_airlinescodrnaadult_fold0.npy')
    train_y, test_y = np.load(
        './examples/big_dataset/trainy_airlinescodrnaadult_fold0.npy'), np.load(
        './examples/big_dataset/testy_airlinescodrnaadult_fold0.npy')
    dataset_dict = dict(train_data=(train_X, train_y),
                        test_data=(test_X, test_y))
    return dataset_dict


model_list = dict(logit=['logit'], rf=['rf'], xgboost=['xgboost'])
finetune = False
task = 'classification'
sampling_range = [0.01, 0.15, 0.3, 0.6]
sampling_algorithm = [
    'Random',
    'CUR']
if __name__ == "__main__":
    results_of_experiments_dict = {}
    dataset_dict = create_big_dataset()
    df = pd.read_pickle('./sampling_experiment.pkl')
    for algo in sampling_algorithm:
        api_config = dict(
            problem=task,
            metric='f1',
            timeout=0.1,
            with_tuning=False,
            industrial_strategy='sampling_strategy',
            industrial_strategy_params={
                'industrial_task': task,
                'sampling_algorithm': algo,
                'sampling_range': sampling_range,
                'data_type': 'table'},
            logging_level=50)
        algo_result = {}
        for model_name, model in model_list.items():
            result_dict = ApiTemplate(api_config=api_config,
                                      metric_list=('f1', 'accuracy')).eval(dataset=dataset_dict,
                                                                           finetune=finetune,
                                                                           initial_assumption=model)
            algo_result.update({f'{algo}_{model_name}': result_dict['metrics']})
        results_of_experiments_dict.update({algo: algo_result})
    with open(f'sampling_experiment.pkl', 'wb') as f:
        pickle.dump(results_of_experiments_dict, f)
