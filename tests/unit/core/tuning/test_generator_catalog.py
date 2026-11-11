"""Catalog laws: typed values, conditional identities and shared tuning domains."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

from hypothesis import given, strategies as st
import pytest

from fedot_ind.core.tuning.generator_catalog import (
    generator_catalog, generator_space, generator_hyperopt_space, parse_generator_catalog,
)


def payload():
    import fedot_ind.core.tuning.generator_catalog as module
    return json.loads(Path(module.__file__).with_name('generator_spaces.json').read_text('utf-8'))


@given(st.sampled_from(generator_catalog()), st.data())
def test_normalization_is_idempotent_and_preserves_context(space, data):
    values = {p.name: data.draw(st.sampled_from(p.values), label=p.name) for p in space.parameters}
    values['random_state'] = 42
    normalized = space.normalize(values)
    assert space.normalize(normalized) == normalized
    assert normalized['random_state'] == 42
    assert values['random_state'] == 42
    for p in space.parameters:
        assert type(normalized[p.name]) in {type(v) for v in p.values}
        if not p.active(normalized):
            assert normalized[p.name] == p.default


def test_quantile_alias_and_inactive_values_have_one_identity():
    numpy = generator_space('quantile_extractor')
    torch = generator_space('quantile_extractor_torch')
    assert numpy.parameters == torch.parameters
    for sliding, stride in product((False, True), (1, 3, 9)):
        assert numpy.normalize(dict(window_size=0, use_sliding_window=sliding, stride=stride)) == numpy.defaults
    assert numpy.normalize(dict(window_size=10, use_sliding_window=True, stride=3))['stride'] == 3
    assert numpy.normalize(dict(window_size=10, stride=3))['stride'] == 1


def test_hyperopt_domains_are_the_same_catalog():
    class HP:
        choice = staticmethod(lambda *args: None)
    actual = generator_hyperopt_space(HP)
    for space in generator_catalog():
        for p in space.parameters:
            assert actual[space.operation][p.name]['sampling-scope'] == [list(p.values)]


def test_bd_profile_uses_catalog_and_keeps_head_grids():
    from benchmark.industrial.experiments.cyclic_mutations.profile import parameter_profile, parse_parameter_profile
    from fedot_ind.core.optimizer.search_policy import load_staged_search_defaults
    grids = parameter_profile()
    head = load_staged_search_defaults().parameters
    assert grids[:len(head)] == head
    assert next(g.values for g in grids if (g.operation, g.parameter)
                == ('fourier_basis', 'output_format')) == ('signal',)
    from benchmark.industrial.experiments.cyclic_mutations import profile
    value = json.loads(Path(profile.__file__).with_name('parameters.json').read_text('utf-8'))
    value['generators']['quantile_extractor']['window_size'] = [99]
    with pytest.raises(ValueError, match='exceed'):
        parse_parameter_profile(value)


@pytest.mark.parametrize('bad', [True, 1., -1, 100, None, '10'])
def test_quantile_window_rejects_values_outside_typed_domain(bad):
    with pytest.raises(ValueError):
        generator_space('quantile_extractor').normalize(dict(window_size=bad))


@pytest.mark.parametrize('case', ['shape', 'dependency_shape', 'cycle', 'alias', 'nan', 'duplicate'])
def test_malformed_catalog_raises_domain_error(case):
    value = deepcopy(payload())
    rows = value['operations']['quantile_extractor']
    if case == 'shape':
        rows['stride'] = []
    elif case == 'dependency_shape':
        rows['window_size'] = {'values': []}
    elif case == 'cycle':
        rows['window_size']['active_when'] = {'stride': [1]}
    elif case == 'alias':
        value['aliases']['bad'] = {}
    elif case == 'nan':
        rows['stride']['values'] = [float('nan')]
    else:
        rows['stride']['values'] = [1, 1]
    with pytest.raises(ValueError):
        parse_generator_catalog(value)


def test_preparation_preserves_generator_choices_and_canonicalizes_inactive_values():
    from fedot.core.pipelines.adapters import PipelineAdapter
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    from fedot_ind.core.optimizer.search_policy import ParameterGrid
    from fedot_ind.core.optimizer.search_policy.preparation import ParameterSearchPreparation
    with industrial_extension_scope():
        adapter = PipelineAdapter()
        grids = tuple(ParameterGrid('quantile_extractor_torch', p.name, p.values)
                      for p in generator_space('quantile_extractor_torch').parameters)
        prepare = ParameterSearchPreparation.from_defaults(
            adapter, {'quantile_extractor_torch': {}}, grids, 'regression', normalize_generators=True)
        pipeline = PipelineBuilder().add_node('quantile_extractor_torch', params={
            'window_size': 10, 'use_sliding_window': True, 'stride': 3, 'add_global_features': False}).add_node('ridge').build()
        result = prepare(adapter.adapt(pipeline))
        assert prepare(result.graph).configuration_id == result.configuration_id
        params = next(n['parameters'] for n in json.loads(result.configuration_json)['nodes']
                      if n['operation'] == 'quantile_extractor_torch')
        assert params['stride'] == 3 and params['window_size'] == 10
        assert params['task_type'] == 'regression'
