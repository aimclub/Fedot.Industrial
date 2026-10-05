import golem.core.log
import numpy as np
import pytest

from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from golem.core.tuning.sequential import SequentialTuner

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.checkers_collections import ApiConfigCheck
from fedot_ind.core.operation.dummy.dummy_operation import init_input_data
from fedot_ind.core.repository.config_repository import DEFAULT_CLF_API_CONFIG
from fedot_ind.integration.fedot.compatibility import ensure_fedot_tensor_data
from fedot_ind.integration.fedot.extensions import industrial_extension_scope
from fedot_ind.tools.loader import DataLoader


def initialize_uni_data():
    train_data, test_data = DataLoader('Lightning7').load_data()
    train_input_data = init_input_data(train_data[0], train_data[1])
    test_input_data = init_input_data(test_data[0], test_data[1])
    return train_input_data, test_input_data


def initialize_multi_data():
    train_data, test_data = DataLoader('Epilepsy').load_data()
    train_input_data = init_input_data(train_data[0], train_data[1])
    test_input_data = init_input_data(test_data[0], test_data[1])
    return train_input_data, test_input_data


def mock_message(self, msg: str, **kwargs):
    level = 40
    self.log(level, msg, **kwargs)


def test_industrial_uni_series(monkeypatch):
    monkeypatch.setattr(golem.core.log.LoggerAdapter, 'message', mock_message)
    train_data, test_data = DataLoader('Lightning7').load_data()
    api_config = ApiConfigCheck().update_config_with_kwargs(
        DEFAULT_CLF_API_CONFIG,
        task='classification',
        timeout=1,
        n_jobs=-1,
    )
    model = FedotIndustrial(**api_config)

    with industrial_extension_scope():
        model.fit(train_data)
        labels = model.predict(test_data)
        probs = model.predict_proba(test_data)

    model.get_metrics(
        labels=labels,
        probs=probs,
        target=test_data[1],
        rounding_order=3,
        metric_names=('accuracy', 'f1'),
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FEDOT d1875e7 TunerBuilder records an infinite metric after adapting "
        "a valid TensorData extension pipeline"
    ),
)
def test_tuner_industrial_uni_series(monkeypatch):
    monkeypatch.setattr(golem.core.log.LoggerAdapter, 'message', mock_message)
    train_data, test_data = initialize_uni_data()
    train_tensor = ensure_fedot_tensor_data(train_data, fit_stage=True)
    test_tensor = ensure_fedot_tensor_data(
        test_data,
        fit_stage=False,
        reference_data=train_tensor,
    )

    with industrial_extension_scope():
        pipeline = (
            PipelineBuilder()
            .add_node('eigen_basis')
            .add_node('quantile_extractor')
            .add_node('pdl_clf')
            .build()
        )
        pipeline_tuner = (
            TunerBuilder(train_tensor.task)
            .with_tuner(SequentialTuner)
            .with_timeout(2)
            .with_iterations(2)
            .build(train_tensor)
        )
        pipeline = pipeline_tuner.tune(pipeline)
        assert np.all(np.isfinite(np.asarray(pipeline_tuner.init_metric)))
        pipeline.fit(train_tensor)
        prediction = pipeline.predict(test_tensor)

    assert len(prediction.predict) == len(test_tensor.idx)
