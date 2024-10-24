import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hyperopt import hp

from fedot_ind.core.architecture.pipelines.abstract_pipeline import AbstractPipeline, ApiTemplate


def plot_mean_sample(X, y, labels: list = [], n_channel: int = 1):
    mean_sample = []
    if len(labels) == 0:
        labels = list(np.unique(y))
    for label in labels:
        mean_sample.append(np.mean(X[y == label], axis=0))  # Данные класса 1
    # ax = plt.gca()
    [f'Channel {x}' for x in range(n_channel)]
    df = pd.DataFrame(mean_sample).T
    df.columns = labels
    df.plot(kind='line', subplots=True, layout=(1, len(labels)), figsize=(20, 10))
    plt.legend(fontsize='small')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


# %%
def plot_mean_sample_multi(X, y, labels: list = [], n_channel: int = None):
    mean_sample = {}
    if len(labels) == 0:
        labels = list(np.unique(y))
    if n_channel is None:
        n_channel = X.shape[1]
    [f'Channel {x}' for x in range(n_channel)]
    for label in labels:
        mask = y == label
        for chn in range(n_channel):
            mean_sample.update(
                {f'Label_{label}_channel_{chn}': np.mean(X[mask.flatten(), chn, :], axis=0)})  # Данные класса 1
    # ax = plt.gca()
    df = pd.DataFrame(mean_sample)
    df.plot(kind='line')
    plt.suptitle('Усреднённые семплы по классам')
    plt.legend(fontsize='small')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


# %% md
# Topo Hyperparams
# %%
topological_params = {'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(5, 50, 5)]]},
                      'stride': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(1, 10, 1)]]}},
# %%
stat_params = {'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(5, 50, 5)]]},
               'stride': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(1, 10, 1)]]},
               'add_global_features': {'hyperopt-dist': hp.choice, 'sampling-scope': [[True, False]]}}
# %%
recurrence_params = {'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(5, 50, 5)]]},
                     'stride': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(1, 10, 1)]]},
                     'rec_metric': (hp.choice, [['cosine', 'euclidean']]),
                     'image_mode': {'hyperopt-dist': hp.choice, 'sampling-scope': [[True, False]]}},
# %%
rec_metric = 'cosine'
image_mode = True
window_size = 10
stride = 1
# %%
topological_node_dict = {'topological_extractor': {'window_size': window_size,
                                                   'stride': stride}}
# %%
recurrence_node_dict = {'recurrence_extractor': {'window_size': window_size,
                                                 'stride': stride,
                                                 'rec_metric': rec_metric,
                                                 'image_mode': image_mode}}

finetune = False
metric_names = ('f1', 'accuracy', 'precision', 'roc_auc')
api_config = dict(problem='classification',
                  metric='accuracy',
                  timeout=1,
                  pop_size=20,
                  with_tuning=True,
                  with_tunig=False,
                  n_jobs=-1,
                  logging_level=20)
pipeline_creator = AbstractPipeline(task='classification')
ECG = 'ECG200'
topological_model = ['topological_extractor', 'rf']
recurrence_model = ['recurrence_extractor', 'quantile_extractor', 'rf']
# %%
ecg_dataset = pipeline_creator.create_input_data(ECG)

if __name__ == "__main__":
    topo_list_model = {
        'topological_extractor': {'window_size': 10},
        'logit': {}}
    result_dict_topo = ApiTemplate(api_config=api_config,
                                   metric_list=metric_names).eval(dataset=ECG,
                                                                  finetune=finetune,
                                                                  initial_assumption=topo_list_model)
    _ = 1
