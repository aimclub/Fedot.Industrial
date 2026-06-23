import json

import numpy as np
import pytest

from fedot_ind.core.kernel_learning import (
    KernelApproximation,
    KernelBundle,
    KernelConfigValidationError,
    KernelLearningReport,
    KernelMatrixPolicy,
    KernelNormalization,
    KernelSelectionReport,
    KernelTaskType,
    PSDCorrectionPolicy,
    TEXT2IMAGE_PROMPTS,
    build_kernel_learning_report,
)


def test_kernel_selection_report_serializes_to_json_ready_dict():
    report = KernelSelectionReport(
        generator_names=("a", "b"),
        weights=(0.75, 0.25),
        selected_generators=("a",),
        selected_weights=(0.75,),
        scores={"a": 1.0, "b": 0.1},
        alignments={"a": 1.0, "b": 0.1},
        complexities={"a": 0.2, "b": 0.3},
        redundancies={"a": 0.0, "b": 0.5},
        task_type="classification",
        diagnostics={"objective_history": (0.1, 0.2)},
    )

    payload = report.to_dict()

    assert payload["generator_names"] == ["a", "b"]
    assert payload["selected_generators"] == ["a"]
    assert json.loads(json.dumps(payload))["weights"] == [0.75, 0.25]


def test_kernel_matrix_policy_rejects_unsupported_modes():
    with pytest.raises(KernelConfigValidationError, match="Unsupported kernel normalization"):
        KernelMatrixPolicy(normalize="distance")


def test_kernel_matrix_policy_normalizes_strings_to_enum_contracts():
    policy = KernelMatrixPolicy(
        kernel="RBF",
        normalize="frobenius",
        psd_correction=PSDCorrectionPolicy.NONE,
        approximation="nystrom",
        nystrom_components=4,
    ).normalized_policy

    assert policy.kernel == "rbf"
    assert policy.normalize is KernelNormalization.FROBENIUS
    assert policy.psd_correction is None
    assert policy.approximation is KernelApproximation.NYSTROM
    assert policy.to_dict()["normalize"] == "frobenius"
    assert policy.to_dict()["psd_correction"] is None


def test_kernel_selection_report_normalizes_task_type_to_enum():
    report = KernelSelectionReport(
        generator_names=("identity",),
        weights=(1.0,),
        selected_generators=("identity",),
        selected_weights=(1.0,),
        scores={"identity": 1.0},
        alignments={"identity": 1.0},
        complexities={"identity": 0.0},
        redundancies={"identity": 0.0},
        task_type="forecasting",
    )

    assert report.task_type is KernelTaskType.FORECASTING
    assert report.to_dict()["task_type"] == "forecasting"


def test_kernel_learning_report_includes_bundles_and_text2image_prompts():
    selection = KernelSelectionReport(
        generator_names=("identity",),
        weights=(1.0,),
        selected_generators=("identity",),
        selected_weights=(1.0,),
        scores={"identity": 1.0},
        alignments={"identity": 1.0},
        complexities={"identity": 0.0},
        redundancies={"identity": 0.0},
        task_type="regression",
    )
    bundle = KernelBundle(name="identity", train_kernel=np.eye(2), diagnostics={"min_eigenvalue": 1.0})

    report = build_kernel_learning_report(selection, kernel_bundles=(bundle,))

    assert isinstance(report, KernelLearningReport)
    assert report.kernel_bundles[0]["train_shape"] == (2, 2)
    assert report.prompts["sparse_mkl"] == TEXT2IMAGE_PROMPTS["sparse_mkl"]
