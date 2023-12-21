from fedot import Fedot

from fedot_ind.core.ensemble.kernel_ensemble import init_kernel_ensemble
from fedot_ind.core.ensemble.rank_ensembler import RankEnsemble
from fedot_ind.tools.loader import DataLoader

n_best = 3
feature_dict = {}
metric_list = []
proba_dict = {}
metric_dict = {}
dataset_name = 'Lightning2'
kernel_list = {'wavelet': [
    {'feature_generator_type': 'signal',
     'feature_hyperparams': {
         'wavelet': "mexh",
         'n_components': 2
     }},
    {'feature_generator_type': 'signal',
     'feature_hyperparams': {
         'wavelet': "morl",
         'n_components': 2
     }}],
    'quantile': [
        {'feature_generator_type': 'quantile',
         'feature_hyperparams': {
             'window_mode': True,
             'window_size': 25
         }
         },
        {'feature_generator_type': 'quantile',
         'feature_hyperparams': {
             'window_mode': False,
             'window_size': 40
         }
         }]
}
fg_names = []
for key in kernel_list:
    for model_params in kernel_list[key]:
        fg_names.append(f'{key}_{model_params}')

train_data, test_data = DataLoader(dataset_name).load_data()
set_of_fg, train_feats, train_target, test_feats, test_target = init_kernel_ensemble(train_data,
                                                                                     test_data,
                                                                                     kernel_list=kernel_list)

n_best_generators = set_of_fg.T.nlargest(n_best, 0).index
for rank in range(n_best):
    fg_rank = n_best_generators[rank]
    train_best = train_feats[fg_rank]
    test_best = test_feats[fg_rank]
    feature_dict.update({fg_names[rank]: (test_best, test_best)})

for model_name, feature in feature_dict.items():
    industrial = Fedot(metric='roc_auc', timeout=5, problem='classification', n_jobs=6)

    model = industrial.fit(feature[0], train_target)
    labels = industrial.predict(feature[1])
    proba_dict.update({model_name: industrial.predict_proba(feature[1])})
    metric_dict.update({model_name: industrial.get_metrics(test_target, metric_names=['roc_auc', 'f1', 'accuracy'])})
rank_ensembler = RankEnsemble(dataset_name=dataset_name,
                              proba_dict={dataset_name: proba_dict},
                              metric_dict={dataset_name: metric_dict})

ensemble_result = rank_ensembler.ensemble()
_ = 1
