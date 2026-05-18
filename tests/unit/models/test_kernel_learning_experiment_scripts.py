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
    assert "kernel_ensemble_classifier" in source
    assert "NON_TOPOLOGICAL_GENERATORS" in source
    assert "KernelEnsembleClassifier_all_non_topological" in source
    assert '"recurrence_extractor"' in source
    assert '"tabular_extractor"' in source
    non_topological_section = \
    source.split("NON_TOPOLOGICAL_GENERATORS = ", maxsplit=1)[1].split("DATASETS =", maxsplit=1)[0]
    assert "topological_extractor" not in non_topological_section


def test_tser_experiment_script_is_declarative_and_uses_local_data_root():
    source = (PROJECT_ROOT / "benchmark" / "run_kernel_learning_tser.py").read_text(encoding="utf-8")

    assert "BenchmarkSuiteConfig(" in source
    assert "run_tser_benchmark_suite(config)" in source
    assert 'TSER_DATA_ROOT = PROJECT_ROOT / "data"' in source
    assert '"local_data_root": str(TSER_DATA_ROOT)' in source
    assert '"download_if_missing": False' in source
    assert "kernel_ensemble_regressor" in source
