import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fedot_ind.core.architecture.postprocessing.Analyzer import PerformanceAnalyzer
from fedot_ind.core.models.spectral.SSAExtractor import SSAExtractor
from fedot_ind.core.models.statistical.StatsExtractor import StatsExtractor
from fedot_ind.core.operation.optimization.FeatureSpace import VarianceSelector
from fedot_ind.core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier
from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
from fedot_ind.core.operation.transformation.basis.legendre import LegenderBasis


def get_results(model, features, target):
    labels = model.predict(test_features=features)
    probs = model.predict_proba(test_features=features)
    metrics = PerformanceAnalyzer().calculate_metrics(target=target,
                                                      predicted_labels=labels['label'],
                                                      predicted_probs=probs['class_probability'])

    return metrics


def get_new_basis(train_data: tuple):
    data = []
    for i in range(0, train_data[0].shape[0], 1):
        x = train_data[0].iloc[i:i + 1, :].values.flatten()
        basis = LegenderBasis(data_range=len(x), n_components=10)
        basis.fit(x)
        basis.evaluate_derivative(order=1)
        # polynom_number = basis.analytical_form()
        # basis.show(visualisation_type='basis representation', basis_function=polynom_number)
        # basis.show(visualisation_type='derivative', basis_function=polynom_number)
        data.append(basis.basis[5])
        # fbasis = FourierDecomposition(data_range=len(x), n_components=10)
        # fbasis.decompose(x)

    update_train = pd.DataFrame(data)
    return update_train

dataset_name = 'Lightning7'
dataset = np.load(r'D:\РАБОТЫ РЕПОЗИТОРИИ\Репозитории\Industiral\IndustrialTS\data\UCI_HAR.npz')

test_data = dataset['data']

test_labels = dataset['labels']
train, test = DataLoader(dataset_name).load_data()
train_target = train[1]
test_target = test[1]
if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\user\Downloads\Telegram Desktop\12.11.22.csv')
    df.rename({'Base_metric': 'Metric_values_before_ensemble', 'Best_ensemble_metric': 'Metric_values_after_ensemble'},
              inplace=True,
              axis=1)
    df[df['Ensemble_models'] != '0'].boxplot(column=['Metric_values_before_ensemble', 'Metric_values_after_ensemble'])
    plt.show()
    update_train = get_new_basis(train_data=train)
    update_test = get_new_basis(train_data=test)

    spectral_model_basic = SSAExtractor(window_mode=True,
                                        window_sizes={dataset_name: [30]},
                                        spectral_hyperparams={'combine_eigenvectors': False,
                                                           'correlation_level': 0.8})
    train_feats_spectral_basic = spectral_model_basic.get_features(train[0], dataset_name)
    test_feats_spectral_basic = spectral_model_basic.get_features(test[0], dataset_name)
    train_eigenvectors_spectral_basic = spectral_model_basic.eigenvectors_list_train

    quantile_model_basic = StatsExtractor(window_mode=True)
    train_feats_baseline = quantile_model_basic.get_features(update_train, dataset_name)
    test_feats_baseline = quantile_model_basic.get_features(update_test, dataset_name)

    model_features = {
        'spectral_basic': train_feats_spectral_basic,
        'quantile': train_feats_baseline
    }

    strategy = VarianceSelector(models=model_features)

    best_model = strategy.get_best_model()
    model_data = model_features[best_model]
    projected_data = strategy.transform(model_data=model_data, principal_components=best_model)
    set_of_features = strategy.select_discriminative_features(model_data=model_data, projected_data=projected_data)
    for key, val in set_of_features.items():
        model_data[val].plot()
        plt.show()
    clf = TimeSeriesClassifier(model_hyperparams={
        'problem': 'classification',
        'seed': 42,
        'timeout': 2,
        'max_depth': 4,
        'max_arity': 2,
        'cv_folds': 2,
        'logging_level': 20,
        'n_jobs': 2
    })
    quantile_model_basic = StatsExtractor(window_mode=True)
    train_feats_baseline = quantile_model_basic.get_features(update_train, dataset_name)
    test_feats_baseline = quantile_model_basic.get_features(update_test, dataset_name)
    IndustrialModelBaseline = clf.fit(train_features=train_feats_baseline, train_target=train_target)[0]
    baseline_results = get_results(clf, test_feats_baseline, test_target)
    predict_baseline = clf.predict(test_feats_baseline)['label']
