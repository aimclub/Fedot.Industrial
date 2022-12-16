from core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier
from core.architecture.postprocessing.Analyzer import PerformanceAnalyzer
from core.architecture.utils.Testing import ModelTestingModule
from core.models.spectral.SSARunner import SSARunner


def test_advanced_spectral_model():
    TestModule = ModelTestingModule(model=SSARunner(window_mode=False,
                                                    window_sizes={'Lightning7': [10, 20, 30]},
                                                    spectral_hyperparams={'combine_eigenvectors': True,
                                                                          'correlation_level': 0.8}))
    train_feats_lightning7, test_feats_lightning7 = TestModule.extract_from_multi_class(dataset_name='Lightning7')
    train_eigenvectors = TestModule.model.eigenvectors_list_train
    TestModule.visualise(train_eigenvectors[:5])
    train_target = TestModule.train_target
    test_target = TestModule.test_target
    basic_ts_clf_class = TimeSeriesClassifier(model_hyperparams={
        'problem': 'classification',
        'seed': 42,
        'timeout': 1,
        'max_depth': 4,
        'max_arity': 2,
        'cv_folds': 2,
        'logging_level': 20,
        'n_jobs': 2
    })
    IndustrialModel = basic_ts_clf_class._fit_model(features=train_feats_lightning7, target=train_target)

    labels = basic_ts_clf_class.predict(test_features=test_feats_lightning7)
    probs = basic_ts_clf_class.predict_proba(test_features=test_feats_lightning7)

    metrics = PerformanceAnalyzer().calculate_metrics(target=test_target,
                                                      predicted_labels=labels['label'],
                                                      predicted_probs=probs['class_probability'])
    assert train_feats_lightning7 is not None
    assert test_feats_lightning7 is not None
    assert metrics is not None
    assert IndustrialModel is not None
