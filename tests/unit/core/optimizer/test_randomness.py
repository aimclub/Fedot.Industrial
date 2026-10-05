from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
import pickle
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.optimization_parameters import GraphRequirements
from hypothesis import given, strategies as st

from fedot_ind.core.optimizer.randomness import (
    CandidateRandomness,
    SeededMutation,
    SeededOperator,
    install_seeded_operators,
    operator_random_scope,
)


def _draw():
    return random.random(), float(np.random.random())


def _mutate(parent):
    return Individual(_draw(), metadata={"source": "test"})


def _address_draw(address):
    with operator_random_scope(address.seed):
        return _draw()


@given(st.integers(min_value=0, max_value=2**64), st.integers(min_value=0, max_value=10000))
def test_seed_address_is_stable_and_matches_independent_rngs(seed, attempt):
    address = CandidateRandomness(seed, "mutation", attempt)
    expected = (random.Random(address.seed).random(),
                float(np.random.RandomState(address.seed).random_sample()))
    assert _address_draw(address) == expected
    assert address.seed == CandidateRandomness(seed, "mutation", attempt).seed
    assert 0 <= address.seed < 2**32


@pytest.mark.parametrize("seed,stream,attempt", [
    (-1, "mutation", 0), (True, "mutation", 0), (0, "", 0),
    (0, None, 0), (0, "mutation", -1), (0, "mutation", True),
])
def test_seed_address_rejects_invalid_inputs(seed, stream, attempt):
    with pytest.raises(ValueError):
        CandidateRandomness(seed, stream, attempt)


def test_rng_states_are_restored_after_nested_scope_and_failure():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    expected = _draw()
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    with pytest.raises(RuntimeError, match="operator failed"):
        with operator_random_scope(42):
            before_nested = random.getstate(), np.random.get_state()
            expected_outer = _draw()
            random.setstate(before_nested[0])
            np.random.set_state(before_nested[1])
            with operator_random_scope(7):
                _draw()
            assert _draw() == expected_outer
            raise RuntimeError("operator failed")
    assert _draw() == expected
    random.setstate(python_state)
    np.random.set_state(numpy_state)


def test_parallel_scopes_match_sequential_draws_without_worker_ids():
    addresses = [CandidateRandomness(99, "mutation", i) for i in range(40)]
    expected = list(map(_address_draw, addresses))
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(_address_draw, addresses)) == expected
    assert expected != [_address_draw(replace(a, run_seed=100)) for a in addresses]


def test_address_replays_in_fresh_process_with_different_hash_seed():
    address = CandidateRandomness(2**40, "mutation", 17)
    code = """
import json
import random
import numpy as np
from fedot_ind.core.optimizer.randomness import CandidateRandomness, operator_random_scope
random.seed(987)
np.random.seed(654)
a = CandidateRandomness(2**40, 'mutation', 17)
with operator_random_scope(a.seed):
    print(json.dumps([a.seed, random.random(), float(np.random.random())]))
"""
    environment = {**os.environ, "PYTHONHASHSEED": "123"}
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                            text=True, env=environment, timeout=60)
    assert json.loads(result.stdout) == [address.seed, *_address_draw(address)]


def test_operator_pickle_keeps_next_attempt_and_forwards_attributes():
    operator = SeededOperator(_draw, 7, "selection")
    operator()
    restored = pickle.loads(pickle.dumps(operator))
    assert restored() == operator()
    assert restored.__name__ == "_draw"
    assert restored.attempt == 2


def test_failed_operator_consumes_one_address_and_restores_rng():
    def fail():
        _draw()
        raise ValueError("failure")

    operator = SeededOperator(fail, 7, "selection")
    state = random.getstate()
    with pytest.raises(ValueError, match="failure"):
        operator()
    assert operator.attempt == 1
    assert random.getstate() == state


def test_mutation_batching_preserves_candidate_addresses_and_metadata():
    parents = [Individual(i, metadata={"parent": i}) for i in range(4)]
    batch = SeededMutation(_mutate, 42, "mutation")(parents)
    individual_operator = SeededMutation(_mutate, 42, "mutation")
    separate = [individual_operator(parent) for parent in parents]
    assert [i.graph for i in batch] == [i.graph for i in separate]
    assert [i.metadata for i in batch] == [i.metadata for i in separate]
    assert [i.metadata["candidate_randomness"]["attempt"] for i in batch] == list(range(4))
    assert [i.metadata for i in parents] == [{"parent": i} for i in range(4)]
    assert all(i.metadata["source"] == "test" for i in batch)


def test_mutation_skipped_or_exhausted_retains_parent_identity_and_advances_seed():
    parent = Individual("parent", metadata={"candidate_randomness": "old"})
    unchanged = SeededMutation(lambda p: p, 7, "mutation")
    assert unchanged(parent) is parent
    assert parent.metadata == {"candidate_randomness": "old"}
    exhausted = SeededMutation(lambda p: [], 7, "mutation")
    assert exhausted([parent, parent]) == []
    assert exhausted.attempt == 2
    assert exhausted([]) == []
    assert exhausted.attempt == 2
    invalid = SeededMutation(lambda p: "invalid", 7, "mutation")
    with pytest.raises(TypeError, match="Individual"):
        invalid(parent)


@pytest.mark.parametrize("invalid", [0, False, "", "empty", ["wrong"]])
def test_mutation_does_not_accept_untyped_empty_sentinels(invalid):
    operator = SeededMutation(lambda parent: invalid, 7, "mutation")
    with pytest.raises(TypeError, match="Individual"):
        operator(Individual("parent"))


def test_mutation_metadata_does_not_change_individual_identity_or_fitness():
    candidate = Individual("candidate", metadata={"source": "operator"})
    operator = SeededMutation(lambda parent: candidate, 42, "mutation")
    result = operator(Individual("parent"))
    assert result.uid == candidate.uid
    assert result.fitness is candidate.fitness
    assert result.parent_operator is candidate.parent_operator
    assert candidate.metadata == {"source": "operator"}
    assert result.metadata["candidate_randomness"]["run_seed"] == 42


@pytest.mark.parametrize("seed", [None, 42])
def test_installation_updates_all_golem_operator_references(seed):
    selection, crossover, mutation, other = (object() for _ in range(4))
    optimizer = SimpleNamespace(selection=selection, crossover=crossover, mutation=mutation,
                                operators=[other, selection, crossover, mutation],
                                reproducer=SimpleNamespace(), inheritance=SimpleNamespace())
    install_seeded_operators(optimizer, seed)
    if seed is None:
        assert optimizer.operators == [other, selection, crossover, mutation]
        return
    assert optimizer.operators == [other, optimizer.selection, optimizer.crossover, optimizer.mutation]
    assert optimizer.reproducer.selection is optimizer.inheritance.selection is optimizer.selection
    assert optimizer.reproducer.crossover is optimizer.crossover
    assert optimizer.reproducer.mutation is optimizer.mutation
    assert optimizer.mutation.operator is mutation


def _golem_mutation(graph, **kwargs):
    graph.nodes[0].content["name"] = f"operation_{random.randrange(100000)}_{np.random.randint(100000)}"
    return graph


def test_real_golem_mutation_replays_graphs_and_retains_genealogy():
    def run(ambient_seed):
        random.seed(ambient_seed)
        np.random.seed(ambient_seed)
        parameters = GPAlgorithmParameters(mutation_types=[_golem_mutation], mutation_prob=1.0)
        requirements = GraphRequirements(show_progress=False, n_jobs=1)
        graph_params = GraphGenerationParams()
        operator = SeededMutation(Mutation(parameters, requirements, graph_params), 42, "mutation")
        parent = Individual(OptGraph([OptNode("initial")]))
        candidates = [operator(parent) for _ in range(4)]
        assert all(candidate.parent_operator.parent_individuals == (parent,) for candidate in candidates)
        assert parent.graph.nodes[0].name == "initial"
        updated = replace(parameters, mutation_prob=0.0)
        operator.update_requirements(parameters=updated)
        assert operator(parent) is parent
        return [(candidate.graph.descriptive_id, candidate.metadata) for candidate in candidates]

    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        assert run(100) == run(200)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
