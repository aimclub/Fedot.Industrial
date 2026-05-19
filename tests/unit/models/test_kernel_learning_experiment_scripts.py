from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_ucr_experiment_script_is_declarative_and_local_first():
    source = (PROJECT_ROOT / "benchmark" / "run_kernel_learning_ucr.py").read_text(encoding="utf-8")

    assert "BenchmarkSuiteConfig(" in source
    assert "run_tsc_benchmark_suite(config)" in source
    assert 'UCR_DATA_ROOT = PROJECT_ROOT / "data"' in source
    assert '"local_data_root": str(UCR_DATA_ROOT)' in source
    assert '"download_if_missing": True' in source
    assert "discover_local_ucr_datasets" in source
    assert "UCR_DATASETS = ()" in source
    assert "kernel_ensemble_classifier" in source
    assert "NON_TOPOLOGICAL_GENERATORS" in source
    assert "KernelEnsembleClassifier_all_non_topological" in source
    assert '"recurrence_extractor"' in source
    assert '"tabular_extractor"' in source
    non_topological_section = \
        source.split("NON_TOPOLOGICAL_GENERATORS = ", maxsplit=1)[1].split("DATASETS =", maxsplit=1)[0]
    assert "topological_extractor" not in non_topological_section


def test_two_stage_ucr_experiment_script_declares_stage_artifacts_and_warm_start():
    source = (PROJECT_ROOT / "benchmark" / "run_kernel_learning_ucr_two_stage.py").read_text(encoding="utf-8")

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
    source = (PROJECT_ROOT / "benchmark" / "run_kernel_learning_tser.py").read_text(encoding="utf-8")

    assert "BenchmarkSuiteConfig(" in source
    assert "run_tser_benchmark_suite(config)" in source
    assert 'TSER_DATA_ROOT = PROJECT_ROOT / "data"' in source
    assert '"local_data_root": str(TSER_DATA_ROOT)' in source
    assert '"download_if_missing": False' in source
    assert "kernel_ensemble_regressor" in source
