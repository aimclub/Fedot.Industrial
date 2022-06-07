from fedot.api.main import Fedot

from core.operation.utils.Composer import FeatureGeneratorBuilder, FeatureGeneratorComposer


class TimeSeriesClf:
    def __init__(self,
                 feature_generator_dict: dict,
                 model_hyperparams: dict):
        self.feature_generator_dict = feature_generator_dict
        self.model_hyperparams = model_hyperparams
        self._init_composer()
        self._init_builder()

    def _init_composer(self):
        self.composer = FeatureGeneratorComposer()
        if self.feature_generator_dict is not None:
            for operation_name, operation_functionality in self.feature_generator_dict.items():
                self.composer.add_operation(operation_name, operation_functionality)

    def _init_builder(self):
        self.builder = FeatureGeneratorBuilder
        for operation_name, operation_functionality in self.feature_generator_dict.items():
            if operation_name.startswith('window'):
                self.feature_generator_dict[operation_name] = self.builder(
                    feature_generator=operation_functionality).add_window_transformation
            elif operation_name.startswith('random'):
                self.feature_generator_dict[operation_name] = self.builder(
                    feature_generator=operation_functionality).add_random_interval_transformation
            else:
                self.feature_generator_dict[operation_name] = self.builder(
                    feature_generator=operation_functionality).add_steady_transformation

    def fit(self, feature, target):
        feature_list = map(lambda x: x(feature), self.composer.dict.values())
        predictor_list = map(lambda x: Fedot(**self.model_hyperparams).fit(feature, target), feature_list)
        return predictor_list

    def predict(self, predictor_list, target):
        predictions_list = map(lambda x: x.predict(target), predictor_list)
        predictions_proba_list = map(lambda x: x.predict_proba(target), predictor_list)
        return predictions_list, predictions_proba_list

    def get_metrics(self, predictor_list, target):
        predictions_list = map(lambda x: x.predict(target), predictor_list)
        predictions_proba_list = map(lambda x: x.predict_proba(target), predictor_list)
        return predictions_list, predictions_proba_list
