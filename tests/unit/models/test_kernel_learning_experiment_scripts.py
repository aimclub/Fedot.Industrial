from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchmark.experiments.kernel_learning import configs as kl_configs
from benchmark.experiments.kernel_learning.datasets import KernelLearningDatasetValidationError
from benchmark.industrial.experiments.kernel_learning import resolve_existing_stage1_run_dir
from benchmark.experiments.kernel_learning.configs import (
    KernelLearningM4ExperimentConfig,
    KernelLearningTSERExperimentConfig,
    KernelLearningTwoStageUCRExperimentConfig,
    KernelLearningUCRExperimentConfig,
    load_kernel_learning_defaults,
)
from benchmark.industrial import TaskType

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read_experiment_script(*parts: str) -> str:
    return (PROJECT_ROOT / "benchmark" / "experiments" / "kernel_learning" / Path(*parts)).read_text(
        encoding="utf-8"
    )


def test_ucr_experiment_script_is_thin_typed_config_shell():
    source = _read_experiment_script("classification", "run_ucr.py")

    assert source.index("sys.path.insert") < source.index("from benchmark.experiments.kernel_learning.configs import")
    assert "KernelLearningUCRExperimentConfig.from_env()" in source
    assert "run_kernel_learning_suite(" in source
    assert "print_benchmark_run_bundle(" in source
    assert "BenchmarkSuiteConfig(" not in source
    assert "run_tsc_benchmark_suite" not in source


def test_kernel_learning_constants_live_in_defaults_json():
    defaults_path = PROJECT_ROOT / "benchmark" / "experiments" / "kernel_learning" / "defaults.json"
    source = (PROJECT_ROOT / "benchmark" / "experiments" / "kernel_learning" / "configs.py").read_text(
        encoding="utf-8"
    )
    defaults = load_kernel_learning_defaults()

    assert defaults_path.exists()
    assert defaults["version"] == "kernel_learning_benchmark_defaults@1"
    assert "KernelEnsembleClassifier_adaptive_all_non_topological" in {
        item["display_name"] for item in defaults["models"]["ucr"]
    }
    assert defaults["experiments"]["ucr"]["custom_dataset_policy"] == "ucr_only"
    assert defaults["experiments"]["two_stage_ucr"]["stage1_run_id"] is None
    assert defaults["experiments"]["two_stage_ucr"]["stage1_run_policy"] == "latest"
    assert defaults["experiments"]["two_stage_ucr"]["custom_dataset_policy"] == "ucr_only"
    assert "kernel_learning_ucr_stage1_ba419d49e4" not in source
    assert "KernelEnsembleClassifier_adaptive_all_non_topological" not in source
    assert "KernelEnsembleRegressor_embedding_nystrom" not in source
    assert "KernelEnsembleForecaster_embedding_nystrom_okhs" not in source
    assert "BenchmarkSuiteConfig(" in source


def test_ucr_config_rejects_non_ucr_names_by_default(tmp_path):
    config = KernelLearningUCRExperimentConfig(
        data_root=tmp_path / "data",
        datasets=("CustomLocal",),
        allowed_dataset_names=("Coffee",),
    )

    with pytest.raises(KernelLearningDatasetValidationError, match="neither a known UCR dataset"):
        config.build_suite_config()


def test_ucr_config_allows_local_custom_split_when_policy_enabled(tmp_path):
    dataset_dir = tmp_path / "data" / "CustomLocal"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "CustomLocal_TRAIN.tsv").write_text("a\t0.0\nb\t1.0\n", encoding="utf-8")
    (dataset_dir / "CustomLocal_TEST.tsv").write_text("a\t0.2\nb\t1.2\n", encoding="utf-8")

    config = KernelLearningUCRExperimentConfig(
        data_root=tmp_path / "data",
        datasets=("CustomLocal",),
        allowed_dataset_names=("Coffee",),
        custom_dataset_policy="allow_local",
        output_dir=tmp_path / "out",
    ).build_suite_config()

    assert config.datasets[0].dataset_name == "CustomLocal"
    assert config.datasets[0].adapter_options["dataset_origin"] == "local_custom"
    assert config.datasets[0].adapter_options["download_if_missing"] is False


def test_ucr_config_builds_classification_suite_and_models(tmp_path):
    for dataset_name in ("Coffee", "Lightning7"):
        dataset_dir = tmp_path / "data" / dataset_name
        dataset_dir.mkdir(parents=True)
        (dataset_dir / f"{dataset_name}_TRAIN.tsv").write_text("a\t0.0\nb\t1.0\n", encoding="utf-8")
        (dataset_dir / f"{dataset_name}_TEST.tsv").write_text("a\t0.2\nb\t1.2\n", encoding="utf-8")

    config = KernelLearningUCRExperimentConfig(
        data_root=tmp_path / "data",
        dataset_limit=1,
        allowed_dataset_names=("Coffee", "Lightning7"),
        output_dir=tmp_path / "out",
    ).build_suite_config()

    assert config.task_type is TaskType.TS_CLASSIFICATION
    assert tuple(spec.dataset_name for spec in config.datasets) == ("Coffee",)
    assert config.datasets[0].adapter_options["local_data_root"] == str(tmp_path / "data")
    assert config.datasets[0].adapter_options["download_if_missing"] is True
    assert config.artifact_spec.output_dir == str(tmp_path / "out")
    assert config.run_spec.resume_enabled is True
    assert config.metrics == ("accuracy", "balanced_accuracy", "f1_macro")

    model_names = tuple(spec.display_name for spec in config.models)
    assert "KernelEnsembleClassifier_score_baseline_summary" in model_names
    assert "KernelEnsembleClassifier_adaptive_all_non_topological" in model_names
    assert "KernelEnsembleClassifier_shapelet_motif_rbf" in model_names
    assert "KernelEnsembleClassifier_embedding_nystrom" in model_names
    adaptive_model = next(spec for spec in config.models if spec.display_name.endswith("all_non_topological"))
    assert "topological_extractor" not in adaptive_model.params["generator_names"]
    assert "recurrence_extractor" in adaptive_model.params["generator_names"]
    assert "tabular_extractor" in adaptive_model.params["generator_names"]


def test_tser_experiment_script_is_thin_typed_config_shell():
    source = _read_experiment_script("regression", "run_tser.py")

    assert source.index("sys.path.insert") < source.index("from benchmark.experiments.kernel_learning.configs import")
    assert "KernelLearningTSERExperimentConfig.from_env()" in source
    assert "run_kernel_learning_suite(" in source
    assert "BenchmarkSuiteConfig(" not in source
    assert "run_tser_benchmark_suite" not in source


def test_tser_config_builds_regression_suite_and_models(tmp_path):
    config = KernelLearningTSERExperimentConfig(
        data_root=tmp_path / "fedot_ind" / "data",
        datasets=("AppliancesEnergy", "ElectricityPredictor"),
        dataset_limit=1,
        output_dir=tmp_path / "out",
    ).build_suite_config()

    assert config.task_type is TaskType.TS_REGRESSION
    assert tuple(spec.dataset_name for spec in config.datasets) == ("AppliancesEnergy",)
    assert config.datasets[0].adapter_options["local_data_root"] == str(tmp_path / "fedot_ind" / "data")
    assert config.datasets[0].adapter_options["download_if_missing"] is False
    assert config.metrics == ("rmse", "mae", "r2")

    model_names = tuple(spec.display_name for spec in config.models)
    assert "KernelEnsembleRegressor_score_linear_summary" in model_names
    assert "KernelEnsembleRegressor_adaptive_rbf_summary" in model_names
    assert "KernelEnsembleRegressor_shapelet_rbf" in model_names
    assert "KernelEnsembleRegressor_embedding_nystrom" in model_names
    assert any(spec.params.get("kernel_approximation") == "nystrom" for spec in config.models)


def test_forecasting_experiment_script_is_thin_typed_config_shell():
    source = _read_experiment_script("forecasting", "run_m4.py")

    assert source.index("sys.path.insert") < source.index("from benchmark.experiments.kernel_learning.configs import")
    assert "KernelLearningM4ExperimentConfig.from_env()" in source
    assert "run_kernel_learning_suite(" in source
    assert "BenchmarkSuiteConfig(" not in source
    assert "run_forecasting_benchmark_suite" not in source


def test_m4_config_builds_forecasting_suite_and_models(tmp_path):
    config = KernelLearningM4ExperimentConfig(
        subsets=("daily", "monthly"),
        sample_size=3,
        output_dir=tmp_path / "out",
    ).build_suite_config()

    assert config.task_type is TaskType.FORECASTING
    assert tuple(spec.dataset_name for spec in config.datasets) == (
        "m4_daily_kernel_learning",
        "m4_monthly_kernel_learning",
    )
    assert tuple(spec.sample_size for spec in config.datasets) == (3, 3)
    assert all(spec.adapter_options["use_local_files"] for spec in config.datasets)
    assert config.metrics == ("mase", "smape", "owa", "rmse", "mae")

    model_names = tuple(spec.display_name for spec in config.models)
    assert "NaiveLastValue" in model_names
    assert "LaggedRidgeForecaster" in model_names
    assert "KernelEnsembleForecaster_identity_shapelet" in model_names
    assert "KernelEnsembleForecaster_embedding_nystrom_okhs" in model_names
    assert any(spec.params.get("kernel_approximation") == "nystrom" for spec in config.models)


def test_two_stage_ucr_experiment_script_normalizes_cli_into_typed_config():
    source = _read_experiment_script("classification", "run_ucr_two_stage.py")

    assert "def parse_args()" in source
    assert "def config_from_args(args: argparse.Namespace)" in source
    assert "KernelLearningTwoStageUCRExperimentConfig(" in source
    assert "--stage1-run-policy" in source
    assert "config.load_or_run_stage1()" in source
    assert "config.run_stage2(stage1_result)" in source

    assert "class KernelLearningStage2Runner" not in source
    assert "def iter_over_dataset" not in source
    assert "def _read_csv_records" not in source
    assert "def load_stage1_result_from_artifacts" not in source
    assert "def importance_report_from_selection" not in source
    assert "KernelInitialPopulationBuilder" not in source
    assert "FedotIndustrial" not in source
    assert "IndustrialEvoOptimizer" not in source


def test_two_stage_config_uses_latest_stage1_run_when_id_is_omitted(tmp_path):
    older = tmp_path / "stage1" / "kernel_learning_ucr_stage1_older"
    newer = tmp_path / "stage1" / "kernel_learning_ucr_stage1_newer"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    older_marker = older / "aggregate"
    newer_marker = newer / "aggregate"
    older_marker.mkdir()
    newer_marker.mkdir()
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    config = KernelLearningTwoStageUCRExperimentConfig(stage1_output_dir=tmp_path / "stage1")

    assert resolve_existing_stage1_run_dir(
        stage1_output_dir=config.stage1_output_dir,
        run_id=config.resolve_stage1_run_id(),
    ) == newer


def test_two_stage_config_keeps_stage_defaults(tmp_path):
    config = KernelLearningTwoStageUCRExperimentConfig(
        data_root=tmp_path / "data",
        datasets=("Coffee",),
        stage1_output_dir=tmp_path / "stage1",
        stage2_output_dir=tmp_path / "stage2",
        timeout_minutes=7,
        pop_size=9,
    )

    assert config.data_root == tmp_path / "data"
    assert config.datasets == ("Coffee",)
    assert config.stage1_output_dir == tmp_path / "stage1"
    assert config.stage2_output_dir == tmp_path / "stage2"
    assert config.stage1_run_id is None
    assert config.stage1_run_policy == "latest"
    assert config.resolve_stage1_run_id() is None
    assert config.generator_names == kl_configs.STAGE1_NON_TOPOLOGICAL_GENERATORS
    assert config.metrics == ("accuracy", "balanced_accuracy", "f1_macro")
    assert config.timeout_minutes == 7
    assert config.pop_size == 9


def test_kernel_learning_experiment_scripts_are_grouped_by_task():
    experiment_root = PROJECT_ROOT / "benchmark" / "experiments" / "kernel_learning"

    assert (experiment_root / "classification" / "run_ucr.py").exists()
    assert (experiment_root / "classification" / "run_ucr_two_stage.py").exists()
    assert (experiment_root / "regression" / "run_tser.py").exists()
    assert (experiment_root / "forecasting" / "run_m4.py").exists()
    assert (experiment_root / "analysis" / "analyze_stage1.py").exists()
    assert (experiment_root / "configs.py").exists()
    assert (experiment_root / "controls.py").exists()

    old_script_names = (
        "run_kernel" + "_learning_ucr.py",
        "run_kernel" + "_learning_tser.py",
        "run_kernel" + "_learning_forecasting.py",
    )
    for script_name in old_script_names:
        assert not (PROJECT_ROOT / "benchmark" / script_name).exists()
