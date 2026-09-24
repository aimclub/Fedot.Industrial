import pytest

try:
    from fedot_ind.integration.fedot.extensions import discovery
except ModuleNotFoundError:
    pytest.skip("The active FEDOT profile has no extension contract.", allow_module_level=True)


def test_tabular_problem_is_a_filtered_view_of_task_catalog():
    classification = set(discovery.catalog_operation_names("classification"))
    tabular = set(discovery.catalog_operation_names("classification_tabular"))

    assert tabular
    assert tabular.issubset(classification)


@pytest.mark.parametrize(
    ("problem", "excluded"),
    [
        ("classification", {"resnet_model", "one_class_svm"}),
        ("regression", {"lora_model", "topological_extractor"}),
        ("classification_tabular", {"resnet_model", "one_class_svm"}),
        ("regression_tabular", {"lora_model", "topological_extractor"}),
    ],
)
def test_catalog_discovery_preserves_operation_exclusions(problem, excluded):
    assert excluded.isdisjoint(discovery.catalog_operation_names(problem))


@pytest.mark.parametrize(
    "problem",
    ["classification", "regression", "ts_forecasting", "anomaly_detection"],
)
def test_automatic_discovery_excludes_operations_that_need_explicit_task_type(problem):
    explicit_task_operations = {
        "inception_model",
        "resnet_model",
        "xcm_model",
        "lora_model",
        "sst",
        "stat_detector",
        "arima_detector",
        "iforest_detector",
        "conv_ae_detector",
        "lstm_ae_detector",
        "channel_filtration",
    }

    assert explicit_task_operations.isdisjoint(discovery.catalog_operation_names(problem))
