import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from fedot_ind.core.models.ts_forecasting.lagged_strategy.lagged_forecaster import (
    LaggedAR,
    resolve_lagged_window_size,
)
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.repository.industrial_implementations.data_transformation import prepare_lagged_table_data
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.core.repository.model_repository import FORECASTING_PREPROC


def _build_ts_input(horizon: int = 14):
    series = np.arange(160, dtype=float)
    return InputData(
        idx=np.arange(len(series)),
        features=series.reshape(-1, 1),
        target=series,
        task=Task(TaskTypesEnum.ts_forecasting,
                  TsForecastingParams(forecast_length=horizon)),
        data_type=DataTypesEnum.ts,
    )


def test_resolve_lagged_window_size_clips_small_values():
    assert resolve_lagged_window_size(50, 1) == 2


def test_lagged_forecaster_builds_forecasting_pipeline_with_lagged_node():
    model = LaggedAR({'window_size': 10, 'stride': 2})
    model.resolved_window_size_ = 16
    model.resolved_hankel_stride_ = 2
    pipeline = model._define_forecasting_pipeline_model()

    assert pipeline.root_node.name == model.channel_model
    assert len(pipeline.root_node.nodes_from) == 1
    assert pipeline.root_node.nodes_from[0].name == 'hankelisation'
    assert pipeline.root_node.nodes_from[0].parameters['window_size'] == 16
    assert pipeline.root_node.nodes_from[0].parameters['stride'] == 2


def test_hankelisation_operation_is_available_in_forecasting_preprocessing_repository():
    assert 'hankelisation' in FORECASTING_PREPROC


def test_lagged_forecaster_compatibility_pcd_uses_lagged_transform():
    input_data = _build_ts_input()
    model = LaggedAR({'window_size': 10, 'stride': 2})
    model.resolved_window_size_ = model._resolve_window_size(input_data)
    model.resolved_hankel_stride_ = model._resolve_hankel_stride()
    transformed = model._create_pcd(input_data, True)

    assert transformed.data_type is DataTypesEnum.table
    assert len(transformed.features.shape) == 2
    assert len(transformed.target.shape) == 2
    assert transformed.target.shape[1] == input_data.task.task_params.forecast_length


def test_prepare_lagged_table_data_matches_reference_hankel_contract():
    input_data = _build_ts_input()
    window_size = resolve_lagged_window_size(input_data.features.shape[0], 10)
    transformed = prepare_lagged_table_data(
        input_data, window_size=window_size, stride=2, is_fit_stage=True)

    reference_features = HankelMatrix(
        time_series=input_data.features,
        window_size=window_size,
        strides=2,
    ).trajectory_matrix.T
    reference_target = HankelMatrix(
        time_series=np.ravel(input_data.features)[window_size:],
        window_size=input_data.task.task_params.forecast_length + 1,
        strides=2,
    ).trajectory_matrix.T
    reference_features = reference_features[:reference_target.shape[0], :]

    assert np.array_equal(transformed.features, reference_features)
    assert np.array_equal(transformed.target, reference_target)
    assert transformed.data_type is DataTypesEnum.table


def test_lagged_forecaster_fit_uses_hankel_pipeline(monkeypatch):
    input_data = _build_ts_input()
    model = LaggedAR({'window_size': 10, 'stride': 2})
    calls = []

    monkeypatch.setattr(
        model,
        '_fit_hankel_pipeline',
        lambda data: calls.append(
            ('hankelisation_pipeline', data.data_type, model.resolved_hankel_stride_)) or model,
    )

    model.fit(input_data)

    assert calls == [('hankelisation_pipeline', DataTypesEnum.ts, 2)]


def test_lagged_forecaster_hankel_pipeline_activates_industrial_repository(monkeypatch):
    input_data = _build_ts_input()
    model = LaggedAR({'window_size': 10, 'stride': 2})
    model.resolved_window_size_ = model._resolve_window_size(input_data)
    model.resolved_hankel_stride_ = model._resolve_hankel_stride()
    calls = []

    monkeypatch.setattr(
        model, '_is_industrial_repository_active', lambda: False)
    monkeypatch.setattr(IndustrialModels, 'setup_repository',
                        lambda self, backend='default': calls.append('setup'))
    monkeypatch.setattr(IndustrialModels, 'setup_default_repository',
                        lambda self, backend='default': calls.append('restore'))
    monkeypatch.setattr(
        model, '_define_forecasting_pipeline_model', lambda: 'pipeline')
    monkeypatch.setattr(
        model,
        '_build_forecasting_tuner',
        lambda model_to_tune, tuning_params, train_data: calls.append(
            ('tune', model_to_tune, train_data.data_type)
        ) or 'tuned-model',
    )

    model._fit_hankel_pipeline(input_data)

    assert calls == [
        'setup', ('tune', 'pipeline', DataTypesEnum.ts), 'restore']
    assert model.tuned_model == 'tuned-model'


def test_lagged_forecaster_pipeline_tuning_uses_forecasting_task(monkeypatch):
    input_data = _build_ts_input()
    model = LaggedAR({'window_size': 10, 'stride': 2})
    model.resolved_window_size_ = model._resolve_window_size(input_data)
    model.resolved_hankel_stride_ = model._resolve_hankel_stride()
    events = []

    class _FakeBuiltTuner:
        def tune(self, pipeline):
            events.append(('tune', pipeline))
            return pipeline

    class _FakeTunerBuilder:
        def __init__(self, task):
            events.append(('task', task.task_type))

        def with_search_space(self, value):
            events.append(('search_space', value.__class__.__name__))
            return self

        def with_tuner(self, value):
            events.append(('tuner', value.__name__))
            return self

        def with_cv_folds(self, value):
            events.append(('cv_folds', value))
            return self

        def with_n_jobs(self, value):
            events.append(('n_jobs', value))
            return self

        def with_metric(self, value):
            events.append(('metric', value))
            return self

        def with_iterations(self, value):
            events.append(('iterations', value))
            return self

        def build(self, tuning_data):
            events.append(('build_data_type', tuning_data.data_type))
            events.append(('build_task', tuning_data.task.task_type))
            return _FakeBuiltTuner()

    class _FakeModel:
        def fit(self, data):
            events.append(('fit', data.data_type, data.task.task_type))

    monkeypatch.setattr(
        'fedot_ind.core.models.ts_forecasting.lagged_strategy.lagged_forecaster.TunerBuilder',
        _FakeTunerBuilder,
    )

    tuned = model._build_forecasting_tuner(
        model_to_tune=_FakeModel(),
        tuning_params=model.tuning_params,
        train_data=input_data,
    )

    assert tuned is not None
    assert ('task', TaskTypesEnum.ts_forecasting) in events
    assert ('build_task', TaskTypesEnum.ts_forecasting) in events
    assert ('build_data_type', DataTypesEnum.ts) in events
    assert ('fit', DataTypesEnum.ts, TaskTypesEnum.ts_forecasting) in events


def test_lagged_forecaster_tuning_data_is_regression_table():
    horizon = 14
    input_data = _build_ts_input(horizon=horizon)

    model = LaggedAR({'window_size': 10, 'stride': 2})
    model.resolved_window_size_ = model._resolve_window_size(input_data)
    model.resolved_hankel_stride_ = model._resolve_hankel_stride()
    model.ts_patch_len = model.resolved_window_size_
    lagged_data = model._create_pcd(input_data, True)
    tuning_data = model._define_tuning_data(lagged_data)

    assert tuning_data.data_type is DataTypesEnum.table
    assert tuning_data.task.task_type is TaskTypesEnum.regression
    assert len(tuning_data.target.shape) == 2
    assert tuning_data.target.shape[1] == horizon
