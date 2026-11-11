"""Public API selects the owned Dask interpreter, not an ambient scheduler."""

import json

import numpy as np

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend
from fedot_ind.core.optimizer.evaluation.dask_config import DaskEvaluationPlan
from fedot_ind.core.optimizer.evaluation.dask_dispatcher import DaskEvaluationDispatcher
from tests.unit.core.optimizer.evaluation.test_public_api import _config, _inputs


def test_fedot_industrial_selects_dask_and_releases_every_batch(tmp_path, monkeypatch):
    plans = []
    original = DaskEvaluationDispatcher._execute_plan

    def observed(self, plan, jobs):
        plans.append(plan)
        return original(self, plan, jobs)

    monkeypatch.setattr(DaskEvaluationDispatcher, "_execute_plan", observed)
    training, prediction, count = _inputs("classification")
    model = FedotIndustrial(**_config("classification", EvaluationBackend.DASK, tmp_path))
    try:
        model.fit(training)
        values = np.asarray(model.predict(prediction)).reshape(-1)
        assert values.size == count
        assert set(values).issubset(set(training[1]))
        assert plans and all(isinstance(plan, DaskEvaluationPlan) for plan in plans)
        assert model.manager.dask_client is None
        assert model.manager.dask_cluster is None
        summary = json.loads((tmp_path / "diagnostics/evolution_summary.json").read_text(encoding="utf-8"))
        assert summary["evaluations_succeeded"] >= 2
    finally:
        model.shutdown()
