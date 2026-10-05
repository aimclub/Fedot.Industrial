"""Pure operation-discovery restrictions shared by FEDOT integration paths."""

from __future__ import annotations


_TEMPORARY_EXCLUSIONS = frozenset({
    "bernb",
    "cat_features",
    "dimension_reduction",
    "dummy",
    "exog_ts",
    "fast_ica",
    "gbr",
    "inception_model",
    "isolation_forest_class",
    "isolation_forest_reg",
    "knn",
    "knnreg",
    "label_encoding",
    "linear",
    "lora_model",
    "multinb",
    "one_hot_encoding",
    "pca",
    "poly_features",
    "rfe_lin_class",
    "rfe_non_lin_class",
    "rfr",
    "riemann_extractor",
    "tst_model",
    "xcm_model",
})

_PROBLEM_EXCLUSIONS = {
    "classification": frozenset({
        "resnet_model",
        "one_class_svm",
        "knnreg",
        "recurrence_extractor",
        "bernb",
        "qda",
    }),
    "classification_tabular": frozenset({
        "resnet_model",
        "knnreg",
        "recurrence_extractor",
        "bernb",
        "qda",
        "one_class_svm",
    }),
    "regression": frozenset({
        "recurrence_extractor",
        "lora_model",
        "topological_extractor",
        "nbeats_model",
        "tcn_model",
        "dummy",
        "deepar_model",
    }),
    "regression_tabular": frozenset({
        "recurrence_extractor",
        "lora_model",
        "topological_extractor",
        "nbeats_model",
        "tcn_model",
        "dummy",
        "deepar_model",
    }),
}


def excluded_operation_names(problem: str) -> frozenset[str]:
    """Return exclusions without importing legacy implementation classes."""
    return _TEMPORARY_EXCLUSIONS.union(_PROBLEM_EXCLUSIONS.get(problem, ()))
