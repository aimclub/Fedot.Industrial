from copy import copy
from functools import partial

import pandas as pd
from fedot.core.constants import default_data_split_ratio_by_task
from fedot.core.data.array_utilities import atleast_4d
from fedot.core.data.cv_folds import cv_generator
from fedot.core.data.data_split import _split_input_data_by_indexes
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    transform_features_and_target_into_lagged
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.optimisers.objective import DataSource
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.preprocessing.data_types import TYPE_TO_ID
from sklearn.model_selection import train_test_split

from fedot_ind.core.architecture.preprocessing.data_convertor import NumpyConverter
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.repository.constanst_repository import FEDOT_HEAD_ENSEMBLE
from typing import Optional, Tuple, Union, Sequence, List, Dict
from fedot.core.data.data import InputData, OutputData

def split_time_series(data: InputData,
                       validation_blocks: Optional[int] = None,
                       **kwargs):
    """ Split time series data into train and test parts

    :param data: InputData object to split
    :param validation_blocks: validation blocks are used for test
    """

    forecast_length = data.task.task_params.forecast_length
    if validation_blocks is not None:
        forecast_length *= validation_blocks

    target_length = len(data.target)
    train_data = _split_input_data_by_indexes(data, index=np.arange(0, target_length - forecast_length),)
    test_data = _split_input_data_by_indexes(data, index=np.arange(target_length - forecast_length, target_length),
                                             retain_first_target=True)

    if validation_blocks is None:
        # for in-sample
        test_data.features = train_data.features
    else:
        # for out-of-sample
        test_data.features = data.features

    return train_data, test_data
def split_any(data: InputData,
              split_ratio: float,
              shuffle: bool,
              stratify: bool,
              random_seed: int,
              **kwargs):
    """ Split any data except timeseries into train and test parts

    :param data: InputData object to split
    :param split_ratio: share of train data between 0 and 1
    :param shuffle: is data needed to be shuffled or not
    :param stratify: make stratified sample or not
    :param random_seed: random_seed for shuffle
    """

    stratify_labels = data.target if stratify else None

    def __split_loop(data, ratio, shuffle, stratify_labels):
        train_ids, test_ids = train_test_split(np.arange(0, len(data.target)),
                                               test_size=1 - ratio,
                                               shuffle=shuffle,
                                               random_state=random_seed,
                                               stratify=stratify_labels)

        train_data = _split_input_data_by_indexes(data, index=train_ids)
        test_data = _split_input_data_by_indexes(data, index=test_ids)
        correct_split = np.unique(test_data.target).shape[0] == np.unique(
            train_data.target).shape[0]
        return train_data, test_data, correct_split

    for ratio in [split_ratio, 0.6, 0.5, 0.4, 0.3, 0.1]:
        train_data, test_data, correct_split = __split_loop(
            data, ratio, shuffle, stratify_labels)
        if correct_split:
            break
    return train_data, test_data


def _are_stratification_allowed(data: Union[InputData, MultiModalData], split_ratio: float) -> bool:
    """ Check that stratification may be done
        :param data: data for split
        :param split_ratio: relation between train data length and all data length
        :return bool: stratification is allowed"""

    # check task_type
    if data.task.task_type is not TaskTypesEnum.classification:
        return False
    else:
        return True


def _are_cv_folds_allowed(data: Union[InputData, MultiModalData], split_ratio: float, cv_folds: int) -> bool:
    try:
        # fast way
        classes = np.unique(data.target, return_counts=True)
    except Exception:
        # slow way
        from collections import Counter
        classes = Counter(data.target)
        classes = [list(classes), list(classes.values())]

    # check that there are enough labels for two samples
    if not all(x > 1 for x in classes[1]):
        if __debug__:
            # tests often use very small datasets that are not suitable for data splitting
            # stratification is disabled for tests
            return None
        else:
            raise ValueError(("There is the only value for some classes:"
                              f" {', '.join(str(val) for val, count in zip(*classes) if count == 1)}."
                              f" Data split can not be done for {data.task.task_type.name} task."))

    # check that split ratio allows to set all classes to both samples
    test_size = round(len(data.target) * (1. - split_ratio))
    labels_count = len(classes[0])
    if test_size < labels_count:
        return None
    else:
        return cv_folds


def _build(self, data: Union[InputData, MultiModalData]) -> DataSource:
    # define split_ratio
    self.split_ratio = self.split_ratio or default_data_split_ratio_by_task[
        data.task.task_type]

    # Check cv_folds
    if self.cv_folds is not None:
        try:
            self.cv_folds = int(self.cv_folds)
        except ValueError:
            raise ValueError(f"cv_folds is not integer: {self.cv_folds}")
        if self.cv_folds < 2:
            self.cv_folds = None
        if self.cv_folds > data.target.shape[0] - 1:
            raise ValueError((f"cv_folds ({self.cv_folds}) is greater than"
                              f" the maximum allowed count {data.target.shape[0] - 1}"))

    # Calculate the number of validation blocks for timeseries forecasting
    if data.task.task_type is TaskTypesEnum.ts_forecasting and self.validation_blocks is None:
        self._propose_cv_folds_and_validation_blocks(data)

    # Check split_ratio
    if self.cv_folds is None and not (0 < self.split_ratio < 1):
        raise ValueError(
            f'split_ratio is {self.split_ratio} but should be between 0 and 1')

    if data.task.task_type is not TaskTypesEnum.ts_forecasting and self.stratify:
        # check that stratification can be done
        # for cross validation split ratio is defined as validation_size / all_data_size
        split_ratio = self.split_ratio if self.cv_folds is None else (
            1 - 1 / (self.cv_folds + 1))
        self.stratify = _are_stratification_allowed(data, split_ratio)
        self.cv_folds = _are_cv_folds_allowed(data, split_ratio, self.cv_folds)
        if not self.stratify:
            self.log.info("Stratificated splitting of data is disabled.")

    # Stratification can not be done without shuffle
    self.shuffle |= self.stratify

    # Random seed depends on shuffle
    self.random_seed = (self.random_seed or 42) if self.shuffle else None

    # Split data
    if self.cv_folds is not None:
        self.log.info("K-folds cross validation is applied.")
        data_producer = partial(cv_generator,
                                data=data,
                                shuffle=self.shuffle,
                                cv_folds=self.cv_folds,
                                random_seed=self.random_seed,
                                stratify=self.stratify,
                                validation_blocks=self.validation_blocks)
    else:
        self.log.info("Hold out validation is applied.")
        data_producer = self._build_holdout_producer(data)

    return data_producer


def build_tuner(self, model_to_tune, tuning_params, train_data, mode):
    pipeline_tuner = TunerBuilder(train_data.task) \
        .with_tuner(tuning_params['tuner']) \
        .with_metric(tuning_params['metric']) \
        .with_timeout(tuning_params.get('tuning_timeout', 20)) \
        .with_early_stopping_rounds(tuning_params.get('tuning_early_stop', 50)) \
        .with_iterations(tuning_params.get('tuning_iterations', 200)) \
        .build(train_data)
    if mode == 'full':
        batch_pipelines = [automl_branch for automl_branch in self.solver.current_pipeline.nodes if
                           automl_branch.name in FEDOT_HEAD_ENSEMBLE]
        for b_pipeline in batch_pipelines:
            b_pipeline.fitted_operation.current_pipeline = pipeline_tuner.tune(
                b_pipeline.fitted_operation.current_pipeline)
            b_pipeline.fitted_operation.current_pipeline.fit(train_data)
    model_to_tune = pipeline_tuner.tune(model_to_tune)
    model_to_tune.fit(train_data)
    return pipeline_tuner, model_to_tune


def postprocess_predicts(self, merged_predicts: np.array) -> np.array:
    """ Post-process merged predictions (e.g. reshape). """
    return merged_predicts


def transform_lagged(self, input_data: InputData):
    train_data = copy(input_data)
    forecast_length = train_data.task.task_params.forecast_length

    # Correct window size parameter
    self._check_and_correct_window_size(train_data.features, forecast_length)
    window_size = self.window_size
    new_idx, transformed_cols, new_target = transform_features_and_target_into_lagged(train_data,
                                                                                      forecast_length,
                                                                                      window_size)

    # Update target for Input Data
    train_data.target = new_target
    train_data.idx = new_idx
    output_data = self._convert_to_output(train_data,
                                          transformed_cols,
                                          data_type=DataTypesEnum.image)
    return output_data


def transform_smoothing(self, input_data: InputData) -> OutputData:
    """Method for smoothing time series

    Args:
        input_data: data with features, target and ids to process

    Returns:
        output data with smoothed time series
    """

    source_ts = input_data.features
    if input_data.data_type == DataTypesEnum.multi_ts:
        full_smoothed_ts = []
        for ts_n in range(source_ts.shape[1]):
            ts = pd.Series(source_ts[:, ts_n])
            smoothed_ts = self._apply_smoothing_to_series(ts)
            full_smoothed_ts.append(smoothed_ts)
        output_data = self._convert_to_output(input_data,
                                              np.array(full_smoothed_ts).T,
                                              data_type=input_data.data_type)
    else:
        source_ts = pd.Series(input_data.features.flatten())
        smoothed_ts = np.ravel(self._apply_smoothing_to_series(source_ts))
        output_data = self._convert_to_output(input_data,
                                              smoothed_ts,
                                              data_type=input_data.data_type)

    return output_data


def _check_and_correct_window_size(self, time_series: np.ndarray, forecast_length: int):
    """ Method check if the length of the time series is not enough for
        lagged transformation

        Args:
            time_series: time series for transformation
            forecast_length: forecast length

        Returns:

        """
    max_allowed_window_size = max(
        1, round((len(time_series) - forecast_length - 1) * 0.25))
    window_list = list(range(3 * forecast_length,
                             max_allowed_window_size, round(1.5 * forecast_length)))

    if self.window_size == 0 or self.window_size > max_allowed_window_size:
        try:
            window_size = np.random.choice(window_list)
        except Exception:
            window_size = 3 * forecast_length
        self.log.message((f"Window size of lagged transformation was changed "
                          f"by WindowSizeSelector from {self.params.get('window_size')} to {window_size}"))
        self.params.update(window_size=window_size)

    # Minimum threshold
    if self.window_size < self.window_size_minimum:
        self.log.info((f"Warning: window size of lagged transformation was changed "
                       f"from {self.params.get('window_size')} to {self.window_size_minimum}"))
        self.params.update(window_size=self.window_size_minimum)


def transform_lagged_for_fit(self, input_data: InputData) -> OutputData:
    """Method for transformation of time series to lagged form for fit stage

    Args:
        input_data: data with features, target and ids to process

    Returns:
        output data with transformed features table
    """
    input_data.features = input_data.features.squeeze()
    new_input_data = copy(input_data)
    forecast_length = new_input_data.task.task_params.forecast_length
    # Correct window size parameter
    self._check_and_correct_window_size(
        new_input_data.features, forecast_length)
    window_size = self.window_size
    new_idx, transformed_cols, new_target = transform_features_and_target_into_lagged(
        input_data,
        forecast_length,
        window_size)

    # Update target for Input Data
    new_input_data.target = new_target
    new_input_data.idx = new_idx
    output_data = self._convert_to_output(new_input_data,
                                          transformed_cols,
                                          data_type=DataTypesEnum.image)
    return output_data


def update_column_types(self, output_data: OutputData):
    """Update column types after lagged transformation. All features becomes ``float``
    """

    _, features_n_cols, _ = output_data.predict.shape
    feature_type_ids = np.array([TYPE_TO_ID[float]] * features_n_cols)
    col_type_ids = {'features': feature_type_ids}

    if output_data.target is not None and len(output_data.target.shape) > 1:
        _, target_n_cols = output_data.target.shape
        target_type_ids = np.array([TYPE_TO_ID[float]] * target_n_cols)
        col_type_ids['target'] = target_type_ids
    output_data.supplementary_data.col_type_ids = col_type_ids


def preprocess_predicts(*args) -> List[np.array]:
    predicts = args[1]
    if len(predicts[0].shape) <= 3:
        return predicts
    else:
        reshaped_predicts = list(map(atleast_4d, predicts))

        # And check image sizes
        img_wh = [predict.shape[1:3] for predict in reshaped_predicts]
        # Can merge only images of the same size
        invalid_sizes = len(set(img_wh)) > 1
        if invalid_sizes:
            raise ValueError(
                "Can't merge images of different sizes: " + str(img_wh))
        return reshaped_predicts


def merge_targets(self) -> np.array:
    filtered_main_target = self.main_output.target
    # if target has the same form as index
    #  then it makes sense to extract target with common indices
    if filtered_main_target is not None and len(self.main_output.idx) == len(filtered_main_target):
        filtered_main_target = self.select_common(
            self.main_output.idx, filtered_main_target)
    return filtered_main_target


def merge_predicts(*args) -> np.array:
    predicts = args[1]

    predicts = [NumpyConverter(
        data=prediction).convert_to_torch_format() for prediction in predicts]
    sample_shape, channel_shape, elem_shape = [
        (x.shape[0], x.shape[1], x.shape[2]) for x in predicts][0]

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
        prediction_2d = np.concatenate(
            [x.reshape(x.shape[0], x.shape[1] * x.shape[2]) for x in predicts], axis=1)
        return prediction_2d.reshape(prediction_2d.shape[0], 1, prediction_2d.shape[1])


def predict_operation(self, fitted_operation, data: InputData, params: Optional[OperationParameters] = None,
                      output_mode: str = 'default', is_fit_stage: bool = False):
    is_main_target = data.supplementary_data.is_main_target
    data_flow_length = data.supplementary_data.data_flow_length
    self._init(data.task, output_mode=output_mode, params=params,
               n_samples_data=data.features.shape[0])

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
