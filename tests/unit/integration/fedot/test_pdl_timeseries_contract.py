"""Declared raw-series support must agree with planning and actual PDL runtime."""

from dataclasses import replace

import numpy as np
import torch
import cv2  # noqa: F401
import pytest
from fedot.core.data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.extensions import extension_scope, clear_extension_registry, get_registered_extensions
from fedot.extensions.registry import registered_extensions_identity

from benchmark.industrial.experiments.mutation_planning.runner import _environment
from fedot_ind.core.optimizer.graph_validation import (
    GraphNodeSpec, GraphSpec, ValidationIssueCode, industrial_capability_index, validate_graph,
)
from fedot_ind.core.optimizer.mutation_planning import (
    AppliedMutation, DropPrimaryTransform, MutationKind, MutationLimits, RawInputSpec,
    apply_mutation_plan, enumerate_mutation_plans,
)
from fedot_ind.core.optimizer.mutation_runtime import (
    advisor_allows_plan, install_extension_aware_advisor, interpret_graph_spec,
    load_runtime_operation_catalog, project_for_mutation,
)
from fedot_ind.integration.fedot.extensions import industrial_extension_scope
from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog
from fedot_ind.integration.fedot.extensions.manifest import build_industrial_extension_manifest


def test_pdl_raw_series_declaration_preserves_existing_types_and_task_boundaries():
    catalog = load_industrial_extension_catalog()
    for name, task in (('pdl_clf', 'classification'), ('pdl_reg', 'regression')):
        operation = catalog.operation(name)
        assert operation.data_types == ('tabular', 'image', 'ts')
        assert operation.tasks == (task,)
        assert operation.requires_target


@pytest.mark.parametrize('task, model, downstream', [
    ('classification', 'pdl_clf', 'dt'), ('regression', 'pdl_reg', 'dtreg'),
])
def test_planned_primary_drop_produces_a_working_targetless_raw_pdl_predictor(task, model, downstream):
    previous = tuple(get_registered_extensions())
    with industrial_extension_scope():
        _, params, graph = _environment(
            {'name': task, 'task': task, 'operations': ['quantile_extractor_torch', model],
             'initial_chain': ['quantile_extractor_torch', model]}, {'max_depth': 4, 'max_arity': 2})
        install_extension_aware_advisor(params)
        before = project_for_mutation(graph, params)
        capabilities = load_runtime_operation_catalog(params).capabilities
        raw = RawInputSpec('ts', True)
        limits = MutationLimits(max_nodes=4, max_depth=4, max_candidates=128, max_arity=2)

        def drops(spec, source):
            candidates = enumerate_mutation_plans(spec, capabilities, limits=limits,
                                                  kinds=(MutationKind.DROP,), raw_input=source)
            return [plan for plan in candidates.plans if isinstance(plan.edits[0], DropPrimaryTransform)]

        plan, = drops(before, raw)
        outcome = apply_mutation_plan(before, plan, limits=limits, raw_input=raw)
        assert isinstance(outcome, AppliedMutation)
        assert len(outcome.graph.nodes) == 1
        assert outcome.graph.nodes[0].operation_name == model
        assert validate_graph(outcome.graph).is_valid
        assert advisor_allows_plan(before, outcome.graph, graph, params.advisor)
        changed = interpret_graph_spec(graph, before, outcome.graph)
        assert params.verifier(changed)
        # Reproduce the old declaration without changing the shared registry or JSON.
        old = replace(before, nodes=tuple(
            replace(node, capabilities=replace(node.capabilities, input_data_types=('tabular', 'image')))
            if node.operation_name == model else node for node in before.nodes))
        assert drops(old, raw) == []
        assert drops(before, RawInputSpec('ts', False)) == []

        pipeline = params.adapter.restore(changed)
        pipeline.root_node.parameters = {'model': downstream, 'max_pairs': 100, 'random_state': 42}
        values = torch.randn(12, 1, 32, generator=torch.Generator().manual_seed(42))
        target = torch.arange(12) % 2 if task == 'classification' else values.mean(dim=(1, 2))
        data = TensorData(task=Task(TaskTypesEnum[task]), data_type=DataTypesEnum.ts,
                          features=values, target=target, idx=np.arange(12))
        test_values = torch.randn(4, 1, 32, generator=torch.Generator().manual_seed(43))
        test = replace(data, features=test_values, target=None, idx=np.array([101, 105, 109, 112]), state='predict')
        snapshot, test_snapshot, target_snapshot = values.clone(), test_values.clone(), target.clone()
        try:
            pipeline.fit(data)
            result = pipeline.predict(test, output_mode='labels' if task == 'classification' else 'default')
            assert result.predict.shape[0] == 4 and torch.isfinite(result.predict).all()
            np.testing.assert_array_equal(result.idx, test.idx)
            if task == 'classification':
                assert set(result.predict.reshape(-1).tolist()).issubset(set(target.tolist()))
                probabilities = pipeline.predict(test, output_mode='full_probs').predict
                assert probabilities.shape == (4, 2)
                assert torch.all(probabilities >= 0)
                torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(4, dtype=probabilities.dtype))
            torch.testing.assert_close(data.features, snapshot)
            torch.testing.assert_close(data.target, target_snapshot)
            torch.testing.assert_close(test.features, test_snapshot)
            assert test.target is None
        finally:
            pipeline.unfit()
    assert tuple(get_registered_extensions()) == previous


@pytest.mark.parametrize('model', ['pdl_clf', 'pdl_reg'])
def test_raw_series_support_does_not_make_pdl_a_forecasting_model(model):
    capabilities = industrial_capability_index()[model]
    spec = GraphSpec(model, (GraphNodeSpec('root', model, capabilities=capabilities),),
                     task_type='ts_forecasting')
    assert ValidationIssueCode.TASK_NOT_SUPPORTED.value in validate_graph(spec).error_codes


def test_pdl_series_capability_changes_cache_identity_without_changing_factories_or_version():
    manifest = build_industrial_extension_manifest()
    old_models = tuple(
        replace(spec, capabilities=replace(spec.capabilities, data_types=tuple(
            kind for kind in spec.capabilities.data_types if kind is not DataTypesEnum.ts)))
        if spec.name in ('pdl_clf', 'pdl_reg') else spec for spec in manifest.models)
    old = replace(manifest, models=old_models)
    previous = tuple(get_registered_extensions())
    with extension_scope():
        clear_extension_registry()
        identities = []
        for candidate in (old, manifest, manifest):
            with extension_scope(candidate):
                identities.append(registered_extensions_identity())
    assert identities[0] != identities[1] == identities[2]
    assert tuple(get_registered_extensions()) == previous
