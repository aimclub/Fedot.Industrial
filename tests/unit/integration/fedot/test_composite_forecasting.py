from dataclasses import replace

import numpy as np
import pytest

from fedot import create_data
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.extensions.contracts import ExtensionContractError

from fedot_ind.core.models.composite_templates import ARINO, CWT, HFCM, MLSTM, PAGE, template_spec
from fedot_ind.core.optimizer.graph_validation import IndustrialGraphVerifier, ValidationIssueCode
from fedot_ind.integration.fedot.extensions import industrial_extension_scope
from fedot_ind.integration.fedot.extensions.discovery import default_industrial_available_operations
from industrial_forecasting.dwt import predict_arino
from industrial_forecasting.wavelet_hfcm import fit_wavelet_hfcm


@pytest.mark.parametrize("template,params", [
    (HFCM, {"haar.levels": 2, "hfcm.order": 2, "hfcm.alpha": 0.2}),
    (ARINO, {"dwt.levels": 2, "seasonal.period": 6, "trend.lookback": 18}),
])
def test_template_is_opt_in_single_terminal_operation_and_matches_reference(template, params):
    series = np.asarray(np.arange(96) * .03 + np.sin(np.arange(96) / 2), dtype=np.float32)
    horizon = 4
    spec = template_spec(template.operation, params)
    if template is HFCM:
        expected = fit_wavelet_hfcm(series, spec).predict(horizon)
    else:
        pytest.importorskip("pywt")
        expected = predict_arino(series, horizon, spec)
    assert template.operation not in default_industrial_available_operations("ts_forecasting")
    with industrial_extension_scope():
        builder = PipelineBuilder().add_node(template.operation, params=params)
        candidate = builder.build()
        assert len(candidate.nodes) == 1
        verifier = IndustrialGraphVerifier(adapter=None, task_type="ts_forecasting")
        assert verifier.verify(candidate)
        assert not verifier.verify(builder.add_node("ridge").build())
        data = create_data(series, target=series, task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(horizon)),
                           ts_forecast_horizon=horizon, without_target=True, use_cache=False)
        candidate.fit(data)
        actual = candidate.predict(replace(data, target=None, state="predict"))
        np.testing.assert_allclose(np.asarray(actual.predict).reshape(-1), expected, rtol=2e-5, atol=2e-5)


def test_role_names_and_cross_parameter_validation():
    assert ARINO.describe({"seasonal.period": 6})["edges"] == ARINO.edges
    with pytest.raises(ValueError, match="Unknown"):
        template_spec(HFCM.operation, {"sum.weight": 0.4})
    with pytest.raises(ValueError, match="nonempty"):
        template_spec(ARINO.operation, {"dwt.levels": 2, "dwt.slow_bands": 3})


@pytest.mark.parametrize("malformed", [None, True, 0, 1.0, "1"])
def test_horizon_is_validated_without_an_execution_context(malformed):
    with pytest.raises(ValueError, match="positive integer"):
        template_spec(ARINO.operation, {"forecast_length": malformed})
    with pytest.raises(ValueError, match="one-step"):
        template_spec(PAGE.operation, {"forecast_length": 4})
    with pytest.raises(ValueError, match="Unknown"):
        template_spec(MLSTM.operation, {"forecast_length": 1})


def test_search_rejects_unsupported_horizon_before_evaluation():
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
    from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
    from golem.core.optimisers.objective import Objective

    from benchmark.industrial.experiments.evolution_parallelism.search import build_environment
    from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer

    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(4))
    settings = {"tasks": {"ts_forecasting": {"operations": [PAGE.operation, HFCM.operation]}},
                "max_depth": 2, "max_arity": 1, "budget_seconds": 60}
    with industrial_extension_scope():
        requirements, params = build_environment(settings, task, workers=1)
        page, conflict, valid = [params.adapter.adapt(PipelineBuilder().add_node(name, params=config).build())
                                 for name, config in ((PAGE.operation, {}),
                                                      (HFCM.operation, {"forecast_length": 1}),
                                                      (HFCM.operation, {}))]
        optimizer = IndustrialEvoOptimizer(
            Objective({"loss": lambda _: pytest.fail("Unexpected evaluation")}),
            [page, conflict, valid], requirements, params,
            GPAlgorithmParameters(mutation_types=[MutationTypesEnum.single_change]),
        )
        assert optimizer._admissible_initial_graphs() == [valid]
        for rejected in (page, conflict):
            report = optimizer.graph_verifier.verify_with_report(rejected)
            assert report.error_codes == (ValidationIssueCode.FORECAST_HORIZON_CONTRACT.value,)


def test_deferred_factory_checks_the_same_horizon_as_the_template():
    from fedot_ind.integration.fedot.extensions.manifest import build_industrial_extension_manifest

    page = next(model for model in build_industrial_extension_manifest().models if model.name == PAGE.operation)
    with pytest.raises(ValueError, match="one-step"):
        page.factory({"forecast_length": 2})
    flat = page.factory({"forecast_length": 1, "projection.rank": 3})
    nested = page.factory({"forecast_length": 1, "projection": {"rank": 3}})
    assert flat.implementation.spec == nested.implementation.spec


def test_native_crossover_and_parameter_mutation_preserve_atomic_nodes():
    from golem.core.optimisers.genetic.operators.crossover import subtree_crossover
    from fedot_ind.core.optimizer.search_policy.contracts import ParameterGrid
    from fedot_ind.core.optimizer.search_policy.parameters import BoundedParameterMutation

    with industrial_extension_scope():
        left = PipelineBuilder().add_node(HFCM.operation, params={"hfcm": {"alpha": .2}}).build()
        right = PipelineBuilder().add_node(ARINO.operation, params={"seasonal": {"period": 6}}).build()
        mutation = BoundedParameterMutation((ParameterGrid(HFCM.operation, "hfcm.alpha", (-1., .5)),))
        adapter = PipelineAdapter()
        assert mutation.has_candidates(adapter.adapt(left))
        tuned = mutation(left)
        assert template_spec(HFCM.operation, tuned.root_node.parameters).alpha == .5
        assert template_spec(HFCM.operation, left.root_node.parameters).alpha == .2
        first, second = subtree_crossover(adapter.adapt(tuned), adapter.adapt(right), max_depth=2, inplace=False)
        swapped = [adapter.restore(graph) for graph in (first, second)]
        verifier = IndustrialGraphVerifier(adapter=None, task_type="ts_forecasting")
        assert [item.root_node.operation.operation_type for item in swapped] == [ARINO.operation, HFCM.operation]
        assert all(len(item.nodes) == 1 and verifier.verify(item) for item in swapped)
        assert template_spec(HFCM.operation, swapped[1].root_node.parameters).alpha == .5
        assert template_spec(ARINO.operation, swapped[0].root_node.parameters).seasonal_period == 6


@pytest.mark.parametrize("template,horizon,settings", [
    (HFCM, 3, {"hfcm.alpha": .2}),
    (ARINO, 3, {"dwt.levels": 2, "seasonal.period": 8}),
    (CWT, 3, {"cwt.window": 12, "encoders.width": 4, "training.epochs": 1}),
    (PAGE, 1, {"page.length": 9, "projection.rank": 2, "training.epochs": 1}),
])
def test_round_trip_on_disk_preserves_forecast_and_template(template, horizon, settings, tmp_path):
    series = np.asarray(np.sin(np.arange(80.) / 4), dtype=np.float32)
    forecast = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(horizon))
    with industrial_extension_scope():
        verifier = IndustrialGraphVerifier(adapter=None, task_type="ts_forecasting", forecast_horizon=horizon)
        model = PipelineBuilder().add_node(template.operation, params=settings).build()
        assert verifier.verify(model)
        data = create_data(series, target=series, task=forecast, ts_forecast_horizon=horizon,
                           without_target=True, use_cache=False)
        ready = replace(data, target=None, state="predict")
        model.fit(data)
        before = np.asarray(model.predict(ready).predict)
        path = tmp_path / "graph.json"
        model.save(str(path), create_subdir=False)
        loaded = Pipeline().load(str(path))
        assert len(loaded.nodes) == 1 and verifier.verify(loaded)
        assert template.describe(loaded.root_node.parameters) == template.describe(model.root_node.parameters)
        assert before.size == horizon and np.isfinite(before).all()
        np.testing.assert_array_equal(np.asarray(loaded.predict(ready).predict), before)
        if template in (ARINO, HFCM):
            changed = replace(ready, features=ready.features + 17)
            with pytest.raises(ExtensionContractError) as failure:
                loaded.predict(changed)
            error = failure.value
            while error.__cause__ is not None:
                error = error.__cause__
            assert isinstance(error, ValueError) and "Refit with the changed history" in str(error)


@pytest.mark.parametrize("template", [ARINO, HFCM])
def test_fixed_origin_adapter_rejects_new_or_corrected_observations_until_refit(template):
    from types import SimpleNamespace

    from fedot_ind.core.models.ts_forecasting.composite import ArinoImplementation, WaveletHFCMImplementation

    implementation = ArinoImplementation if template is ARINO else WaveletHFCMImplementation
    reference = (lambda x: predict_arino(x, 4)) if template is ARINO else (lambda x: fit_wavelet_hfcm(x).predict(4))
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(4))
    series = np.arange(80.) / 2 + np.sin(np.arange(80.) / 3)
    original = series.copy()
    model = implementation().fit(SimpleNamespace(features=series, task=task))
    series[-1] += 100
    extended = np.r_[original, [100., 125., 152., 170.]]
    for changed in (series, extended):
        with pytest.raises(ValueError, match="Refit with the changed history"):
            model.predict(SimpleNamespace(features=changed, task=task))
    np.testing.assert_array_equal(model.predict(SimpleNamespace(features=original, task=task)), reference(original))
    updated = SimpleNamespace(features=extended, task=task)
    model.fit(updated)
    np.testing.assert_array_equal(model.predict(updated), reference(extended))


def test_one_step_adapter_rejects_invalid_task_before_training(monkeypatch):
    from types import SimpleNamespace

    from fedot_ind.core.models.ts_forecasting.composite import PageProjectionImplementation
    from industrial_forecasting.neural import page_projection

    monkeypatch.setattr(page_projection, "fit_page_projection", lambda *a, **kw: pytest.fail("Unexpected fit"))
    data = SimpleNamespace(features=np.ones(24), task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(3)))
    with pytest.raises(ValueError, match="one-step"):
        PageProjectionImplementation().fit(data)


@pytest.mark.parametrize("data_type", [DataTypesEnum.image, DataTypesEnum.multi_ts])
def test_mlstm_opt_in_graph_fits_panel_and_ignores_unobserved_suffix(data_type, tmp_path):
    rng = np.random.default_rng(10)
    labels = np.repeat((0, 1), 5)
    x = rng.normal(size=(len(labels), 2, 12)).astype(np.float32)
    assert MLSTM.operation not in default_industrial_available_operations("classification")
    settings = {"prefix.cutoff": 6, "encoders.width": 4, "training.epochs": 1}
    with industrial_extension_scope():
        task = Task(TaskTypesEnum.classification)
        model = PipelineBuilder().add_node(MLSTM.operation, params=settings).build()
        verifier = IndustrialGraphVerifier(adapter=None, task_type="classification", input_data_type=data_type)
        assert verifier.verify(model)
        model.fit(create_data(x, target=labels, task=task, data_type=data_type, use_cache=False))
        clean = create_data(x, task=task, data_type=data_type, use_cache=False)
        changed = x.copy()
        changed[..., 6:] += 1e3
        cut = create_data(changed, task=task, data_type=data_type, use_cache=False)
        expected = np.asarray(model.predict(clean, output_mode="full_probs").predict)
        assert expected.shape == (len(labels), 2)
        np.testing.assert_array_equal(np.asarray(model.predict(cut, output_mode="full_probs").predict), expected)
        path = tmp_path / "mlstm.json"
        model.save(str(path), create_subdir=False)
        loaded = Pipeline().load(str(path))
        np.testing.assert_array_equal(np.asarray(loaded.predict(clean, output_mode="full_probs").predict), expected)
