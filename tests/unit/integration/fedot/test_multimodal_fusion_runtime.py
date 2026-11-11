"""Real named-modality fusion through FEDOT and both owned local executors."""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest
import torch
from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluateWithTensorData
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.pipelines.adapters import PipelineAdapter
from golem.core.optimisers.opt_history_objects.individual import Individual

from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.core.multimodal.packing import unpack_modalities
from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend, EvaluationExecutionConfig
from fedot_ind.core.optimizer.evaluation.dispatcher import LocalEvaluationDispatcher
from fedot_ind.integration.fedot import IntegrationContractError, IntegrationErrorCode, TensorMultimodalClassifier
from fedot_ind.integration.fedot.contracts import RuntimeState
from fedot_ind.integration.fedot.extensions import industrial_extension_scope
from fedot_ind.integration.fedot.multimodal import (
    build_multimodal_fusion_pipeline, prepare_multimodal_fusion_data,
)


def _bundle(samples=12, seed=7, reverse=False):
    generator = torch.Generator().manual_seed(seed)
    items = [
        (MultimodalModality.raw, torch.randn(samples, 1, 32, generator=generator)),
        (MultimodalModality.stats, torch.randn(samples, 6, generator=generator)),
    ]
    return MultimodalDataBundle(dict(reversed(items) if reverse else items),
                                target=torch.tensor([3, 7] * (samples // 2)))


def _parameters(method="concat"):
    return {"fusion_method": method, "d_model": 8,
            "training_config": {"epochs": 1, "batch_size": 4, "device": "cpu",
                                "seed": 7, "timing_warmup": False}}


@pytest.mark.parametrize("method", ["concat", "ordinary_bottleneck", "raw_residual_bottleneck"])
def test_real_fusion_graph_preserves_modalities_predictions_and_ownership(method):
    from fedot.extensions import get_registered_extensions
    before = tuple(get_registered_extensions())
    bundle = _bundle()
    original = {name: value.clone() for name, value in bundle.modalities.items()}
    coordinates = np.array([f"sample-{i}" for i in reversed(range(12))])
    with TensorMultimodalClassifier(parameters=_parameters(method)) as runtime:
        runtime.fit(bundle, idx=coordinates)
        graph = runtime._pipeline
        assert len(graph.nodes) == 3
        implementation = graph.root_node.fitted_operation.implementation
        assert implementation.trainer_.history.train_loss
        assert set(implementation.model_.modalities) == set(bundle.modalities)
        labels = runtime.predict(bundle.without_target(), idx=coordinates)
        probs = runtime.predict(_bundle(reverse=True).without_target(), probabilities=True, idx=coordinates)
        repeated = runtime.predict(bundle.without_target(), probabilities=True, idx=coordinates)
        graph.root_node.nodes_from.reverse()
        reordered = runtime.predict(bundle.without_target(), probabilities=True, idx=coordinates)
        assert set(labels.values.reshape(-1)) <= {3, 7}
        assert probs.values.shape == (12, 2)
        np.testing.assert_allclose(probs.values.sum(axis=1), 1., atol=1e-6)
        np.testing.assert_allclose(probs.values, repeated.values, rtol=0, atol=0)
        np.testing.assert_array_equal(probs.values, reordered.values)
        np.testing.assert_array_equal(labels.idx, coordinates)
        np.testing.assert_array_equal(probs.classes, [3, 7])
    assert runtime.state is RuntimeState.CLOSED
    assert all(node.fitted_operation is None for node in graph.nodes)
    assert tuple(get_registered_extensions()) == before
    for name, value in bundle.modalities.items():
        assert torch.equal(value, original[name])


def test_preflight_rejects_changed_modalities_before_prediction(monkeypatch):
    with TensorMultimodalClassifier(parameters=_parameters()) as runtime:
        bundle = _bundle()
        runtime.fit(bundle)

        def forbidden(*args, **kwargs):
            pytest.fail("Model called before rejecting an incompatible schema")

        monkeypatch.setattr(runtime._pipeline, "predict", forbidden)
        for data in (
            MultimodalDataBundle({MultimodalModality.raw: bundle.modalities[MultimodalModality.raw]}),
            bundle.with_modalities({**bundle.modalities, MultimodalModality.stats: torch.ones(12, 7)}).without_target(),
        ):
            with pytest.raises(IntegrationContractError) as error:
                runtime.predict(data)
            assert error.value.code in {IntegrationErrorCode.MODALITY_MISMATCH, IntegrationErrorCode.SCHEMA_MISMATCH}


def test_prepared_transport_restores_shapes_values_and_owns_storage():
    bundle = _bundle(reverse=True)
    prepared = prepare_multimodal_fusion_data(bundle, stage="train")
    restored = unpack_modalities(prepared.data.features, prepared.layout, target=prepared.data.target)
    for item in prepared.layout:
        name = MultimodalModality(item.name)
        assert restored.modalities[name].shape == bundle.modalities[name].shape
        torch.testing.assert_close(restored.modalities[name], bundle.modalities[name])
        assert restored.modalities[name].data_ptr() != bundle.modalities[name].data_ptr()


@pytest.mark.parametrize("with_specs", [False, True])
def test_image_modality_keeps_spatial_shape_through_actual_fusion_graph(with_specs):
    from fedot_ind.core.multimodal.mapping import DEFAULT_MODALITY_SPECS
    bundle = _bundle()
    bundle = bundle.with_modalities({**bundle.modalities,
                                    MultimodalModality.gaf: torch.arange(12 * 64).reshape(12, 1, 8, 8).float() / 100})
    if with_specs:
        bundle = bundle.replace(specs=DEFAULT_MODALITY_SPECS)
    original_specs = dict(bundle.specs)
    prepared = prepare_multimodal_fusion_data(bundle, stage="train")
    restored = unpack_modalities(prepared.data.features, prepared.layout)
    torch.testing.assert_close(restored.modalities[MultimodalModality.gaf], bundle.modalities[MultimodalModality.gaf])
    with TensorMultimodalClassifier(parameters=_parameters()) as runtime:
        runtime.fit(bundle)
        result = runtime.predict(bundle.without_target(), probabilities=True)
        assert result.values.shape == (12, 2)
        assert runtime._pipeline.root_node.fitted_operation.implementation.trainer_._build_shapes[
            MultimodalModality.gaf] == (12, 1, 8, 8)
        changed = bundle.with_modalities({**bundle.modalities,
                                         MultimodalModality.gaf: torch.ones(12, 1, 4, 16)})
        with pytest.raises(IntegrationContractError) as error:
            runtime.predict(changed)
        assert error.value.code is IntegrationErrorCode.SCHEMA_MISMATCH
    assert dict(bundle.specs) == original_specs


def test_dataset_preparer_output_trains_without_discarding_domain_specs():
    from fedot_ind.core.multimodal.preparation import MultimodalDatasetPreparer
    from fedot_ind.core.multimodal.configs import build_preparation_config
    features = np.random.default_rng(7).normal(size=(12, 1, 64)).astype(np.float32)
    preparer = MultimodalDatasetPreparer(build_preparation_config(torch_device="cpu"))
    bundle = preparer.fit_transform(features, np.array([3, 7] * 6))
    assert bundle.specs[MultimodalModality.gaf].allowed_ndim == (4,)
    with TensorMultimodalClassifier(parameters=_parameters()) as runtime:
        runtime.fit(bundle)
        held_out = preparer.transform(features[:4])
        result = runtime.predict(held_out, probabilities=True)
        assert result.values.shape == (4, 2)
        np.testing.assert_allclose(result.values.sum(axis=1), 1., atol=1e-6)


def _source(folds):
    return iter(folds)


def _loss(pipeline, reference_data, **kwargs):
    output = pipeline.predict(replace(reference_data, target=None, state="predict"), output_mode="labels")
    predicted = torch.as_tensor(output.predict).reshape(-1)
    return float((predicted != reference_data.target.reshape(-1)).float().mean())


def _cleanup(pipeline):
    pipeline.unfit()


@pytest.mark.parametrize("backend", [EvaluationBackend.SEQUENTIAL, EvaluationBackend.PROCESS])
def test_multimodal_composite_graph_is_evaluated_and_cleaned_by_local_executor(backend):
    folds = []
    for seed in (7, 11):
        train = prepare_multimodal_fusion_data(_bundle(seed=seed), stage="train")
        valid = prepare_multimodal_fusion_data(_bundle(seed=seed + 1), stage="predict", training=train)
        # Held-out labels belong to the synthetic validation metric, never prediction inputs.
        folds.append((train.data, replace(valid.data, target=_bundle(seed=seed + 1).target)))
    evaluator = PipelineObjectiveEvaluateWithTensorData(
        MetricsObjective(_loss), partial(_source, tuple(folds)),
        expected_folds=2, result_cache_size=0, eval_n_jobs=1)
    adapter = PipelineAdapter()
    with industrial_extension_scope():
        pipeline = build_multimodal_fusion_pipeline(train.layout, parameters=_parameters())
        individual = Individual(adapter.adapt(pipeline))
    dispatcher = LocalEvaluationDispatcher(
        adapter, n_jobs=1, config=EvaluationExecutionConfig(backend), graph_cleanup_fn=_cleanup)

    assert dispatcher.dispatch(evaluator.evaluate)([individual]) == [individual]

    assert individual.fitness.valid
    assert np.isfinite(individual.fitness.value)
    result, = dispatcher.last_results
    returned = adapter.restore(result.value.graph)
    assert all(node.fitted_operation is None for node in returned.nodes)


def test_runtime_state_and_failed_fit_do_not_leave_fitted_graph():
    runtime = TensorMultimodalClassifier(parameters=_parameters())
    with pytest.raises(IntegrationContractError):
        runtime.predict(_bundle())
    invalid = _bundle().replace(target=torch.ones(12, dtype=torch.long))
    with pytest.raises(ValueError):
        runtime.fit(invalid)
    assert runtime.state is RuntimeState.CREATED
    assert runtime._pipeline is None
    runtime.close()
    runtime.close()
    with pytest.raises(IntegrationContractError):
        runtime.fit(_bundle())


def test_prediction_requires_training_schema_and_duplicate_indices_are_rejected():
    with pytest.raises(IntegrationContractError) as error:
        prepare_multimodal_fusion_data(_bundle(), stage="predict")
    assert error.value.code is IntegrationErrorCode.PLAN_MISMATCH
    with pytest.raises(IntegrationContractError) as error:
        prepare_multimodal_fusion_data(_bundle(), stage="train", idx=np.zeros(12, dtype=int))
    assert error.value.code is IntegrationErrorCode.INDEX_MISMATCH


def test_fusion_manifest_requires_explicit_layout_and_selector_bounds():
    from fedot_ind.integration.fedot.extensions.manifest import build_industrial_extension_manifest
    manifest = build_industrial_extension_manifest()
    model = next(item for item in manifest.models if item.name == "future_fusion_clf")
    selector = next(item for item in manifest.transforms if item.name == "industrial_modality_slice")
    assert model.hyperparams_schema.required == ("modality_layout",)
    assert set(selector.hyperparams_schema.required) == {"input_width", "start", "stop"}
