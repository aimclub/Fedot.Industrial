from typing import List, Iterable, Union, Optional
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.data.array_utilities import atleast_4d
from fedot.core.data.data import InputData

from fedot_ind.core.architecture.preprocessing.data_convertor import NumpyConverter


def preprocess_predicts(*args) -> List[np.array]:
    predicts = args[1]
    if len(predicts[0].shape) <= 3:
        return predicts
    else:
        reshaped_predicts = list(map(atleast_4d, predicts))

        # And check image sizes
        img_wh = [predict.shape[1:3] for predict in reshaped_predicts]
        invalid_sizes = len(set(img_wh)) > 1  # Can merge only images of the same size
        if invalid_sizes:
            raise ValueError("Can't merge images of different sizes: " + str(img_wh))
        return reshaped_predicts


def merge_predicts(*args) -> np.array:
    predicts = args[1]

    predicts = [NumpyConverter(data=prediction).convert_to_torch_format() for prediction in predicts]
    sample_shape, channel_shape, elem_shape = [(x.shape[0], x.shape[1], x.shape[2]) for x in predicts][0]

    sample_wise_concat = [x.shape[0] == sample_shape for x in predicts]
    chanel_concat = [x.shape[1] == channel_shape for x in predicts]
    element_wise_concat = [x.shape[2] == elem_shape for x in predicts]

    channel_match = all(chanel_concat)
    element_match = all(element_wise_concat)
    sample_match = all(sample_wise_concat)

    if sample_match and element_match:
        return np.concatenate(predicts, axis=1)
    elif sample_match and channel_match:
        return np.concatenate(predicts, axis=2)
    else:
        prediction_2d = np.concatenate([x.reshape(x.shape[0], x.shape[1] * x.shape[2]) for x in predicts], axis=1)
        return prediction_2d.reshape(prediction_2d.shape[0], 1, prediction_2d.shape[1])


def predict_operation(self, fitted_operation, data: InputData, params: Optional[OperationParameters] = None,
                      output_mode: str = 'default', is_fit_stage: bool = False):
    is_main_target = data.supplementary_data.is_main_target
    data_flow_length = data.supplementary_data.data_flow_length
    self._init(data.task, output_mode=output_mode, params=params, n_samples_data=data.features.shape[0])

    if is_fit_stage:
        prediction = self._eval_strategy.predict_for_fit(
            trained_operation=fitted_operation,
            predict_data=data,
            output_mode=output_mode)
    else:
        prediction = self._eval_strategy.predict(
            trained_operation=fitted_operation,
            predict_data=data,
            output_mode=output_mode)
    prediction = self.assign_tabular_column_types(prediction, output_mode)

    # any inplace operations here are dangerous!
    if is_main_target is False:
        prediction.supplementary_data.is_main_target = is_main_target

    prediction.supplementary_data.data_flow_length = data_flow_length
    return prediction


def predict(self, fitted_operation, data: InputData, params: Optional[Union[OperationParameters, dict]] = None,
            output_mode: str = 'labels'):
    """This method is used for defining and running of the evaluation strategy
    to predict with the data provided

    Args:
        fitted_operation: trained operation object
        data: data used for prediction
        params: hyperparameters for operation
        output_mode: string with information about output of operation,
        for example, is the operation predict probabilities or class labels
    """
    return self._predict(fitted_operation, data, params, output_mode, is_fit_stage=False)


def predict_for_fit(self, fitted_operation, data: InputData, params: Optional[OperationParameters] = None,
                    output_mode: str = 'default'):
    """This method is used for defining and running of the evaluation strategy
    to predict with the data provided during fit stage

    Args:
        fitted_operation: trained operation object
        data: data used for prediction
        params: hyperparameters for operation
        output_mode: string with information about output of operation,
            for example, is the operation predict probabilities or class labels
    """
    return self._predict(fitted_operation, data, params, output_mode, is_fit_stage=True)
