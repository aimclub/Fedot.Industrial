from dataclasses import replace

import numpy as np
import pytest

from fedot import create_data
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from fedot_ind.core.models.ts_forecasting.composite import ARINO, HFCM, template_spec
from fedot_ind.core.optimizer.graph_validation import IndustrialGraphVerifier
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


def test_native_crossover_and_parameter_mutation_preserve_atomic_nodes():
    from fedot.core.pipelines.adapters import PipelineAdapter
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
