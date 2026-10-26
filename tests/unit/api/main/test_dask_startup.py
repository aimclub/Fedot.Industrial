from copy import deepcopy
from types import SimpleNamespace

import pytest

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.services.dask_runtime import DaskRuntimeInitializer
from fedot_ind.core.repository.config_repository import DEFAULT_REG_API_CONFIG


@pytest.mark.parametrize('backend', ['legacy', 'sequential', 'process', 'dask'])
@pytest.mark.parametrize('compute_backend', ['cpu', 'dask'])
@pytest.mark.parametrize('distributed', [None, {'n_workers': 2, 'processes': True}])
def test_public_fit_does_not_start_legacy_dask_runtime(backend, compute_backend, distributed, monkeypatch):
    def unexpected_start(*args, **kwargs):
        pytest.fail('FedotIndustrial must not create an implicit Dask runtime')

    monkeypatch.setattr(DaskRuntimeInitializer, 'start', unexpected_start)
    config = deepcopy(DEFAULT_REG_API_CONFIG)
    config['automl_config']['optimisation_strategy'] = {
        'optimisation_agent': 'Industrial',
        'optimisation_strategy': {'evaluation_execution': {'backend': backend}},
    }
    config['compute_config']['distributed'] = distributed
    config['compute_config']['backend'] = compute_backend
    model = FedotIndustrial(**config)
    marker, solver = object(), object()
    plans, fitted = [], []
    model._process_input_data = lambda data, *, fit_stage: data
    model._FedotIndustrial__init_industrial_backend = lambda data: data
    model.solver_factory = SimpleNamespace(create=lambda plan: plans.append(plan) or solver)
    model.fit_service = SimpleNamespace(fit=lambda manager, data: fitted.append((manager.solver, data)))

    model.fit(marker)
    model.shutdown()
    model.shutdown()

    assert len(plans) == 1
    assert fitted == [(solver, marker)]
    assert model.manager.dask_client is None
    assert model.manager.dask_cluster is None
