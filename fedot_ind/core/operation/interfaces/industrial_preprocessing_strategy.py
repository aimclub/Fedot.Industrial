import warnings

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.common_preprocessing import FedotPreprocessingStrategy
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy, \
    convert_to_multivariate_model, is_multi_output_task
from fedot.core.operations.evaluation.operation_implementations.data_operations.categorical_encoders import \
    OneHotEncodingImplementation, LabelEncodingImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.decompose import \
    DecomposerClassImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_filters import \
    IsolationForestClassImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_imbalanced_class import \
    ResampleImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_selectors import \
    NonLinearClassFSImplementation, LinearClassFSImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import \
    *
from fedot.core.operations.evaluation.operation_implementations.data_operations.topological.topological_extractor import \
    TopologicalFeaturesImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.utilities.random import ImplementationRandomStateHandler
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestClassifier, GradientBoostingRegressor, ExtraTreesRegressor, \
    RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import (
    Lasso as SklearnLassoReg,
    LinearRegression as SklearnLinReg,
    LogisticRegression as SklearnLogReg,
    Ridge as SklearnRidgeReg,
    SGDRegressor as SklearnSGD
)
from fedot_ind.core.models.nn.network_impl.mini_rocket import MiniRocketExtractor
from fedot_ind.core.models.recurrence.reccurence_extractor import RecurrenceExtractor
from fedot_ind.core.models.signal.signal_extractor import SignalExtractor
from fedot_ind.core.models.quantile.quantile_extractor import QuantileExtractor
from fedot_ind.core.models.topological.topological_extractor import TopologicalExtractor
from fedot_ind.core.operation.dummy.dummy_operation import DummyOperation
from fedot_ind.core.operation.filtration.feature_filtration import FeatureFilter

from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementation
from fedot_ind.core.operation.transformation.basis.wavelet import WaveletBasisImplementation
from fedot_ind.core.operation.transformation.basis.eigen_basis import EigenBasisImplementation

from fedot_ind.core.repository.IndustrialOperationParameters import IndustrialOperationParameters


def convert_input_data(train_data, mode: str = 'feature_extraction'):
    if mode == 'model_fitting':
        new_shape = train_data.features.shape[0], train_data.features.shape[1] * train_data.features.shape[2]
        features = train_data.features.reshape(new_shape)
        input_data = InputData(idx=train_data.idx,
                               features=features,
                               target=train_data.target,
                               task=train_data.task,
                               data_type=train_data.data_type,
                               supplementary_data=train_data.supplementary_data)
    else:
        input_data = [InputData(idx=train_data.idx,
                                features=features,
                                target=train_data.target,
                                task=train_data.task,
                                data_type=train_data.data_type,
                                supplementary_data=train_data.supplementary_data) for features in
                      train_data.features.swapaxes(1, 0)]
    return input_data


class IndustrialPreprocessingStrategy(FedotPreprocessingStrategy):
    """
    Args:
        operation_type: ``str`` of the operation defined in operation or data operation repositories

            .. details:: possible operations:

                - ``data_driven_basic``-> EigenBasisImplementation,
                - ``topological_features``-> TopologicalExtractor,


        params: hyperparameters to fit the operation with

    """

    __operations_by_types = {
        'eigen_basis': EigenBasisImplementation,
        'wavelet_basis': WaveletBasisImplementation,
        'fourier_basis': FourierBasisImplementation,
        'topological_extractor': TopologicalExtractor,
        'quantile_extractor': QuantileExtractor,
        'signal_extractor': SignalExtractor,
        'recurrence_extractor': RecurrenceExtractor,
        'minirocket_extractor': MiniRocketExtractor,
        'cat_features': DummyOperation,
        'dimension_reduction': FeatureFilter
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        params = IndustrialOperationParameters().from_params(operation_type, params) if params \
            else IndustrialOperationParameters().from_operation_type(operation_type)
        super().__init__(operation_type, params)

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain custom preprocessing strategy for {operation_type}')


class IndustrialCustomPreprocessingStrategy(FedotPreprocessingStrategy):
    """
    Args:
        operation_type: ``str`` of the operation defined in operation or data operation repositories

            .. details:: possible operations:

                - ``scaling``-> ScalingImplementation,
                - ``normalization``-> NormalizationImplementation,
                - ``simple_imputation``-> ImputationImplementation,
                - ``pca``-> PCAImplementation,
                - ``kernel_pca``-> KernelPCAImplementation,
                - ``poly_features``-> PolyFeaturesImplementation,
                - ``one_hot_encoding``-> OneHotEncodingImplementation,
                - ``label_encoding``-> LabelEncodingImplementation,
                - ``fast_ica``-> FastICAImplementation

        params: hyperparameters to fit the operation with

    """

    _operations_by_types = {
        'scaling': ScalingImplementation,
        'normalization': NormalizationImplementation,
        'simple_imputation': ImputationImplementation,
        'pca': PCAImplementation,
        'kernel_pca': KernelPCAImplementation,
        'poly_features': PolyFeaturesImplementation,
        'one_hot_encoding': OneHotEncodingImplementation,
        'label_encoding': LabelEncodingImplementation,
        'fast_ica': FastICAImplementation
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """This method is used for operation training with the data provided

        Args:
            train_data: data used for operation training

        Returns:
            trained Sklearn operation
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        operation_implementation = self.operation_impl(self.params_for_fit)
        fit_method = operation_implementation.fit
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            if len(train_data.features.shape) > 2:
                input_data = convert_input_data(train_data)
                fitted_operation = list(map(fit_method, input_data))
                operation_implementation = fitted_operation
            else:
                operation_implementation = fit_method(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """Transform method for preprocessing task

        Args:
            trained_operation: model object
            predict_data: data used for prediction

        Returns:
            prediction
        """
        if type(trained_operation) is list:
            prediction = self._predict_for_ndim(predict_data, trained_operation)
        else:
            prediction = trained_operation.transform(predict_data)
        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(prediction, predict_data, output_data_type=predict_data.data_type)
        return converted

    def _predict_for_ndim(self, predict_data, trained_operation):
        test_data = [InputData(idx=predict_data.idx,
                               features=features,
                               target=predict_data.target,
                               task=predict_data.task,
                               data_type=predict_data.data_type,
                               supplementary_data=predict_data.supplementary_data) for features in
                     predict_data.features.swapaxes(1, 0)]
        prediction = list(operation.transform(data.features) for operation, data in zip(trained_operation, test_data))
        prediction = np.stack(prediction).swapaxes(0, 1)
        return prediction

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Transform method for preprocessing task for fit stage

        Args:
            trained_operation: model object
            predict_data: data used for prediction
        Returns:
            OutputData:
        """
        if type(trained_operation) is list:
            prediction = self._predict_for_ndim(predict_data, trained_operation)
        else:
            prediction = trained_operation.transform(predict_data)
        converted = self._convert_to_output(prediction, predict_data, output_data_type=predict_data.data_type)
        return converted


class IndustrialSkLearnEvaluationStrategy(EvaluationStrategy):
    """This class defines the certain operation implementation for the sklearn operations
    defined in operation repository

    Args:
        operation_type: ``str`` of the operation defined in operation or
            data operation repositories

            .. details:: possible operations:

                - ``xgbreg``-> XGBRegressor
                - ``adareg``-> AdaBoostRegressor
                - ``gbr``-> GradientBoostingRegressor
                - ``dtreg``-> DecisionTreeRegressor
                - ``treg``-> ExtraTreesRegressor
                - ``rfr``-> RandomForestRegressor
                - ``linear``-> SklearnLinReg
                - ``ridge``-> SklearnRidgeReg
                - ``lasso``-> SklearnLassoReg
                - ``svr``-> SklearnSVR
                - ``sgdr``-> SklearnSGD
                - ``lgbmreg``-> LGBMRegressor
                - ``xgboost``-> XGBClassifier
                - ``logit``-> SklearnLogReg
                - ``bernb``-> SklearnBernoulliNB
                - ``multinb``-> SklearnMultinomialNB
                - ``dt``-> DecisionTreeClassifier
                - ``rf``-> RandomForestClassifier
                - ``mlp``-> MLPClassifier
                - ``lgbm``-> LGBMClassifier
                - ``kmeans``-> SklearnKmeans

        params: hyperparameters to fit the operation with
    """

    _operations_by_types = {
        'xgbreg': XGBRegressor,
        'adareg': AdaBoostRegressor,
        'gbr': GradientBoostingRegressor,
        'dtreg': DecisionTreeRegressor,
        'treg': ExtraTreesRegressor,
        'rfr': RandomForestRegressor,
        'linear': SklearnLinReg,
        'ridge': SklearnRidgeReg,
        'lasso': SklearnLassoReg,
        'lgbmreg': LGBMRegressor,
        'xgboost': XGBClassifier,
        'logit': SklearnLogReg,
        'rf': RandomForestClassifier,
        'mlp': MLPClassifier,
        'lgbm': LGBMClassifier,

    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        if len(train_data.features.shape) > 2:
            train_data = convert_input_data(train_data, mode='model_fitting')
        return self.fit_one_sample(train_data)

    def fit_one_sample(self, train_data: InputData):
        """This method is used for operation training with the data provided

        Args:
            train_data: data used for operation training

        Returns:
            trained Sklearn operation
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        operation_implementation = self.operation_impl(**self.params_for_fit.to_dict())

        # If model doesn't support multi-output and current task is ts_forecasting
        current_task = train_data.task.task_type
        models_repo = OperationTypesRepository()
        non_multi_models = models_repo.suitable_operation(task_type=current_task,
                                                          tags=['non_multi'])
        is_model_not_support_multi = self.operation_type in non_multi_models

        # Multi-output task or not
        is_multi_target = is_multi_output_task(train_data)
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            if is_model_not_support_multi and is_multi_target:
                # Manually wrap the regressor into multi-output model
                operation_implementation = convert_to_multivariate_model(operation_implementation,
                                                                         train_data)
            else:
                operation_implementation.fit(train_data.features, train_data.target)
        return operation_implementation

    def _find_operation_by_impl(self, impl):
        for operation, operation_impl in self._operations_by_types.items():
            if operation_impl == impl:
                return operation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """Method to predict the target data for predict stage.

        Args:
            trained_operation: trained operation object
            predict_data: data to predict

        Returns:
            passed data with new predicted target
        """
        if len(predict_data.features.shape) > 2:
            predict_data = convert_input_data(predict_data)
        return self.predict(trained_operation, predict_data)

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        """Method to predict the target data for fit stage.
        Allows to implement predict method different from main predict method
        if another behaviour for fit graph stage is needed.

        Args:
            trained_operation: trained operation object
            predict_data: data to predict
        Returns:
            passed data with new predicted target
        """
        return self.predict(trained_operation, predict_data)

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))

    def _sklearn_compatible_prediction(self, trained_operation, features):
        is_multi_output_target = isinstance(trained_operation.classes_, list)
        # Check if target is multilabel (has 2 or more columns)
        if is_multi_output_target:
            n_classes = len(trained_operation.classes_[0])
        else:
            n_classes = len(trained_operation.classes_)
        if self.output_mode == 'labels':
            prediction = trained_operation.predict(features)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_operation.predict_proba(features)
            if n_classes < 2:
                raise ValueError('Data set contain only 1 target class. Please reformat your data.')
            elif n_classes == 2 and self.output_mode != 'full_probs':
                if is_multi_output_target:
                    prediction = np.stack([pred[:, 1] for pred in prediction]).T
                else:
                    prediction = prediction[:, 1]
        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

        return prediction


class IndustrialSkLearnClassificationStrategy(IndustrialSkLearnEvaluationStrategy):
    """ Strategy for applying classification algorithms from Sklearn library """

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Predict method for classification task for predict stage

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :return: prediction target
        """
        if len(predict_data.features.shape) > 2:
            predict_data = convert_input_data(predict_data, mode='model_fitting')
        prediction = self._sklearn_compatible_prediction(trained_operation=trained_operation,
                                                         features=predict_data.features)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def fit(self, train_data: InputData):
        if len(train_data.features.shape) > 2:
            train_data = convert_input_data(train_data, mode='model_fitting')
        return self.fit_one_sample(train_data)

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Transform method for preprocessing task for fit stage

        Args:
            trained_operation: model object
            predict_data: data used for prediction
        Returns:
            OutputData:
        """
        if len(predict_data.features.shape) > 2:
            predict_data = convert_input_data(predict_data, mode='model_fitting')
        prediction = self._sklearn_compatible_prediction(trained_operation=trained_operation,
                                                         features=predict_data.features)
        converted = self._convert_to_output(prediction, predict_data)
        return converted


class IndustrialClassificationPreprocessingStrategy(IndustrialCustomPreprocessingStrategy):
    """ Strategy for applying custom algorithms from FEDOT to preprocess data
    for classification task
    """

    _operations_by_types = {
        'rfe_lin_class': LinearClassFSImplementation,
        'rfe_non_lin_class': NonLinearClassFSImplementation,
        'class_decompose': DecomposerClassImplementation,
        'resample': ResampleImplementation,
        'isolation_forest_class': IsolationForestClassImplementation,
        'topological_features': TopologicalFeaturesImplementation
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)

    def fit(self, train_data: InputData):
        """This method is used for operation training with the data provided

        Args:
            train_data: data used for operation training

        Returns:
            trained Sklearn operation
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        operation_implementation = self.operation_impl(self.params_for_fit)
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            if len(train_data.features.shape) > 2:
                input_data = convert_input_data(train_data)
                fitted_operation = [operation_implementation for i in range(len(input_data))]
                operation_implementation = fitted_operation
        return operation_implementation

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Transform method for preprocessing task for fit stage

        Args:
            trained_operation: model object
            predict_data: data used for prediction
        Returns:
            OutputData:
        """
        if type(trained_operation) is list:
            prediction = self._predict_for_ndim(predict_data, trained_operation)
        else:
            prediction = trained_operation.transform(predict_data)
        converted = self._convert_to_output(prediction, predict_data, output_data_type=predict_data.data_type)
        return converted

    def _predict_for_ndim(self, predict_data, trained_operation):
        test_data = [InputData(idx=predict_data.idx,
                               features=features,
                               target=predict_data.target,
                               task=predict_data.task,
                               data_type=predict_data.data_type,
                               supplementary_data=predict_data.supplementary_data) for features in
                     predict_data.features.swapaxes(1, 0)]
        prediction = list(operation.transform(data).predict for operation, data in zip(trained_operation, test_data))
        prediction = np.stack(prediction).swapaxes(0, 1)
        return prediction
