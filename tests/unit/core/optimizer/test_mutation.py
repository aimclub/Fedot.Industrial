from types import SimpleNamespace

from fedot_ind.core.optimizer.mutation import mutation_name, without_resample_mutations


def keep_mutation():
    pass


def resample_mutation():
    pass


def test_mutation_name_supports_callable_and_enum_like_values():
    assert mutation_name(keep_mutation) == "keep_mutation"
    assert mutation_name(SimpleNamespace(name="named_mutation")) == "named_mutation"
    assert mutation_name(object()) == ""


def test_resample_filter_is_pure_and_preserves_order():
    named = SimpleNamespace(name="other_mutation")
    mutations = [keep_mutation, resample_mutation, named]

    filtered = without_resample_mutations(mutations)

    assert filtered == [keep_mutation, named]
    assert mutations == [keep_mutation, resample_mutation, named]
