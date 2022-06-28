from fedot.api.main import Fedot

from cases.analyzer import PerfomanceAnalyzer
from core.operation.utils.Composer import FeatureGeneratorComposer
from core.operation.utils.FeatureBuilder import FeatureBuilderSelector


class TimeSeriesClf:
    def __init__(self,
                 feature_generator_dict: dict,
                 model_hyperparams: dict):
        self.feature_generator_dict = feature_generator_dict
        self.model_hyperparams = model_hyperparams
        self.analyzer = PerfomanceAnalyzer()
        self.metrics_name = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision']
        self._init_composer()
        self._init_builder()

    def _init_composer(self):
        self.composer = FeatureGeneratorComposer()
        if self.feature_generator_dict is not None:
            for operation_name, operation_functionality in self.feature_generator_dict.items():
                self.composer.add_operation(operation_name, operation_functionality)

        self.list_of_generators = list(self.composer.dict.values())

    def _init_builder(self):
        for operation_name, operation_functionality in self.feature_generator_dict.items():
            self.feature_generator_dict[operation_name] = \
                FeatureBuilderSelector(operation_name, operation_functionality).add_transformation()

    def _fit_fedot_model(self, feature, target):
        fedot_model = Fedot(**self.model_hyperparams)
        fedot_model.fit(feature, target)
        return fedot_model

    def fit(self, train_tuple):
        feature_list = list(map(lambda x: x.extract_features(train_tuple[0]), self.list_of_generators))
        predictor_list = list(map(lambda x: self._fit_fedot_model(x, train_tuple[1]), feature_list))
        return dict(predictors=predictor_list, train_features=feature_list)

    def predict(self, predictor_list, test_tuple):
        feature_list = list(map(lambda x: x.extract_features(test_tuple[0]), self.list_of_generators))
        predictions_list = list(map(lambda x, y: x.predict(y), predictor_list, feature_list))
        predictions_proba_list = list(map(lambda x, y: x.predict_proba(y), predictor_list, feature_list))
        metrics_dict = list(map(lambda x, y: self.analyzer.calculate_metrics(self.metrics_name,
                                                                             target=test_tuple[1],
                                                                             predicted_labels=x,
                                                                             predicted_probs=y
                                                                             ), predictions_list,
                                predictions_proba_list))

        return dict(prediction=predictions_list,
                    prediction_proba=predictions_proba_list,
                    metrics=metrics_dict,
                    test_features=feature_list)
