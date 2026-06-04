from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_ucr_experiment_script_is_declarative_and_local_first():
    source = (
            PROJECT_ROOT
            / "benchmark"
            / "experiments"
            / "kernel_learning"
            / "classification"
            / "run_ucr.py"
    ).read_text(encoding="utf-8")

    assert source.index("sys.path.insert") < source.index("from benchmark.industrial import")
    assert "BenchmarkSuiteConfig(" in source
    assert "run_tsc_benchmark_suite(config)" in source
    assert 'UCR_DATA_ROOT = PROJECT_ROOT / "data"' in source
    assert '"local_data_root": str(UCR_DATA_ROOT)' in source
    assert '"download_if_missing": True' in source
    assert "discover_local_ucr_datasets" in source
    assert "UCR_DATASETS = ()" in source
    assert "KERNEL_LEARNING_UCR_DATASETS" in source
    assert "KERNEL_LEARNING_UCR_LIMIT" in source
    assert "kernel_ensemble_classifier" in source
    assert "NON_TOPOLOGICAL_GENERATORS" in source
    assert "KernelEnsembleClassifier_score_baseline_summary" in source
    assert "KernelEnsembleClassifier_adaptive_all_non_topological" in source
    assert "KernelEnsembleClassifier_shapelet_motif_rbf" in source
    assert "KernelEnsembleClassifier_embedding_nystrom" in source
    assert '"selector_optimizer": "score"' in source
    assert '"selector_optimizer": "projected_gradient"' in source
    assert '"shapelet_extractor"' in source
    assert '"embedding_extractor"' in source
    assert '"kernel_approximation": "nystrom"' in source
    assert '"recurrence_extractor"' in source
    assert '"tabular_extractor"' in source
    non_topological_section = \
        source.split("NON_TOPOLOGICAL_GENERATORS = ", maxsplit=1)[1].split("DATASETS =", maxsplit=1)[0]
    assert "topological_extractor" not in non_topological_section


def test_two_stage_ucr_experiment_script_declares_stage_artifacts_and_warm_start():
    source = (
            PROJECT_ROOT
            / "benchmark"
            / "experiments"
            / "kernel_learning"
            / "classification"
            / "run_ucr_two_stage.py"
    ).read_text(encoding="utf-8")

    assert "def load_or_run_stage1()" in source
    assert "def run_stage2(stage1)" in source
    assert "KernelLearningStage1Runner" in source
    assert "KernelLearningStage2Runner" in source
    assert "load_stage1_result_from_artifacts" in source
    assert "resolve_existing_stage1_run_dir" in source
    assert "STAGE1_RUN_ID" in source
    assert "RUN_STAGE_1" in source
    assert "--run-stage-1" in source
    assert "--stage1-run-id" in source
    assert "--datasets" in source
    assert "--stage2-output-dir" in source

    assert "class KernelLearningStage2Runner" not in source
    assert "def iter_over_dataset" not in source
    assert "def _read_csv_records" not in source
    assert "def load_stage1_result_from_artifacts" not in source
    assert "def importance_report_from_selection" not in source
    assert "KernelInitialPopulationBuilder" not in source
    assert "FedotIndustrial" not in source
    assert "IndustrialEvoOptimizer" not in source


def test_tser_experiment_script_is_declarative_and_uses_local_data_root():
    source = (
            PROJECT_ROOT
            / "benchmark"
            / "experiments"
            / "kernel_learning"
            / "regression"
            / "run_tser.py"
    ).read_text(encoding="utf-8")

    assert source.index("sys.path.insert") < source.index("from benchmark.industrial import")
    assert "BenchmarkSuiteConfig(" in source
    assert "run_tser_benchmark_suite(config)" in source
    assert 'TSER_DATA_ROOT = PROJECT_ROOT / "fedot_ind" / "data"' in source
    assert '"local_data_root": str(TSER_DATA_ROOT)' in source
    assert '"download_if_missing": False' in source
    assert "KERNEL_LEARNING_TSER_DATASETS" in source
    assert "KERNEL_LEARNING_TSER_LIMIT" in source
    assert "kernel_ensemble_regressor" in source
    assert "KernelEnsembleRegressor_score_linear_summary" in source
    assert "KernelEnsembleRegressor_adaptive_rbf_summary" in source
    assert "KernelEnsembleRegressor_shapelet_rbf" in source
    assert "KernelEnsembleRegressor_embedding_nystrom" in source
    assert '"selector_optimizer": "score"' in source
    assert '"selector_optimizer": "projected_gradient"' in source
    assert '"kernel_approximation": "nystrom"' in source


def test_forecasting_experiment_script_is_declarative_and_uses_kernel_adapter():
    source = (
            PROJECT_ROOT
            / "benchmark"
            / "experiments"
            / "kernel_learning"
            / "forecasting"
            / "run_m4.py"
    ).read_text(encoding="utf-8")

    assert source.index("sys.path.insert") < source.index("from benchmark.industrial import")
    assert "BenchmarkSuiteConfig(" in source
    assert "run_forecasting_benchmark_suite(config)" in source
    assert "TaskType.FORECASTING" in source
    assert "KERNEL_LEARNING_M4_SUBSETS" in source
    assert "KERNEL_LEARNING_M4_SAMPLE_SIZE" in source
    assert "kernel_ensemble_forecaster" in source
    assert "KernelEnsembleForecaster_identity_shapelet" in source
    assert "KernelEnsembleForecaster_embedding_nystrom_okhs" in source
    assert '"use_local_files": True' in source
    assert '"shapelet_extractor"' in source
    assert '"embedding_extractor"' in source
    assert '"kernel_approximation": "nystrom"' in source


def test_kernel_learning_experiment_scripts_are_grouped_by_task():
    experiment_root = PROJECT_ROOT / "benchmark" / "experiments" / "kernel_learning"

    assert (experiment_root / "classification" / "run_ucr.py").exists()
    assert (experiment_root / "classification" / "run_ucr_two_stage.py").exists()
    assert (experiment_root / "regression" / "run_tser.py").exists()
    assert (experiment_root / "forecasting" / "run_m4.py").exists()
    assert (experiment_root / "analysis" / "analyze_stage1.py").exists()
    assert (experiment_root / "controls.py").exists()

    old_script_names = (
        "run_kernel" + "_learning_ucr.py",
        "run_kernel" + "_learning_tser.py",
        "run_kernel" + "_learning_forecasting.py",
    )
    for script_name in old_script_names:
        assert not (PROJECT_ROOT / "benchmark" / script_name).exists()
