"""Real two-generator pipelines, not only static graph admissibility."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from benchmark.industrial.experiments.mutation_planning.runner import _environment
from fedot_ind.core.optimizer.mutation_planning import (
    AddPrimaryBranch, MutationKind, MutationLimits, RawInputSpec,
    apply_mutation_plan, enumerate_mutation_plans,
)
from fedot_ind.core.optimizer.mutation_runtime import (
    advisor_allows_plan, install_extension_aware_advisor, interpret_graph_spec,
    load_runtime_operation_catalog, project_for_mutation,
)
from fedot_ind.integration.fedot.extensions import industrial_extension_scope


@pytest.mark.parametrize('task, model, downstream', [
    ('classification', 'pdl_clf', 'dt'), ('regression', 'pdl_reg', 'dtreg'),
])
def test_two_current_generators_train_and_predict_with_preserved_rows(task, model, downstream):
    with industrial_extension_scope():
        settings = {'max_depth': 4, 'max_arity': 2}
        scenario = {'name': task, 'task': task,
                    'operations': ['quantile_extractor', 'quantile_extractor_torch', model],
                    'initial_chain': ['quantile_extractor_torch', model]}
        _, params, graph = _environment(scenario, settings)
        install_extension_aware_advisor(params)
        before = project_for_mutation(graph, params)
        raw = RawInputSpec('ts', True)
        limits = MutationLimits(max_nodes=4, max_depth=4, max_candidates=128, max_arity=2)
        candidates = enumerate_mutation_plans(before, load_runtime_operation_catalog(params).capabilities,
                                              limits=limits, kinds=(MutationKind.ADD,), raw_input=raw)
        plan = next(plan for plan in candidates.plans if isinstance(plan.edits[0], AddPrimaryBranch))
        outcome = apply_mutation_plan(before, plan, limits=limits, raw_input=raw)
        assert advisor_allows_plan(before, outcome.graph, graph, params.advisor)
        changed = interpret_graph_spec(graph, before, outcome.graph)
        assert params.verifier(changed)
        pipeline = params.adapter.restore(changed)
        pipeline.root_node.parameters = {'model': downstream, 'max_pairs': 100, 'random_state': 42}
        values = torch.randn(12, 1, 32, generator=torch.Generator().manual_seed(42))
        target = torch.arange(12) % 2 if task == 'classification' else values.mean(dim=(1, 2))
        indices = np.array([9, 2, 7, 1, 6, 3, 8, 0, 5, 4, 11, 10])
        data = TensorData(task=Task(TaskTypesEnum[task]), data_type=DataTypesEnum.ts,
                          features=values, target=target, idx=indices)
        snapshot = data.features.clone()
        fitted = pipeline.fit(data)
        assert fitted.predict.shape[0] == len(values)
        np.testing.assert_array_equal(fitted.idx, indices)
        torch.testing.assert_close(fitted.target, target)
        test = replace(data, features=values[:4].clone(), target=None, idx=indices[:4], state='predict')
        result = pipeline.predict(test)
        assert result.predict.shape[0] == 4 and torch.isfinite(result.predict).all()
        np.testing.assert_array_equal(result.idx, indices[:4])
        torch.testing.assert_close(data.features, snapshot)
