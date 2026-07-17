import subprocess
from pathlib import Path

import benchmark.industrial as industrial_benchmark
from benchmark.industrial import datasets, evaluation, experiments, models, visualization

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_industrial_package_is_canonical_benchmark_entrypoint():
    assert industrial_benchmark.ModelSpec is models.ModelSpec
    assert industrial_benchmark.run_local_benchmark_preset is experiments.run_local_benchmark_preset
    assert datasets.load_local_supervised_split
    assert evaluation.render_tsc_publication_pack
    assert visualization.visualize_forecasting_progress_items


def test_legacy_benchmark_modules_are_removed():
    assert not (PROJECT_ROOT / "benchmark" / ("v" + "2")).exists()
    assert not (PROJECT_ROOT / "benchmark" /
                ("benchmarking" + "_utils")).exists()


def test_root_benchmark_package_does_not_contain_legacy_entrypoints():
    old_root_files = (
        "abstract" + "_bench.py",
        "benchmark" + "_TSC.py",
        "benchmark" + "_TSER.py",
        "benchmark" + "_TSF.py",
        "feature" + "_utils.py",
        "automl" + "_forecasting.py",
    )
    for file_name in old_root_files:
        assert not (PROJECT_ROOT / "benchmark" / file_name).exists()


def test_industrial_root_keeps_only_public_runtime_shells():
    allowed_root_files = {
        "__init__.py",
        "__main__.py",
        "api.py",
        "classification.py",
        "cli.py",
        "core.py",
        "errors.py",
        "forecasting.py",
        "regression.py",
    }
    actual_root_files = {
        path.name
        for path in (PROJECT_ROOT / "benchmark" / "industrial").iterdir()
        if path.is_file()
    }

    assert actual_root_files <= allowed_root_files


def test_thematic_packages_own_helper_implementations():
    industrial_root = PROJECT_ROOT / "benchmark" / "industrial"

    assert (industrial_root / "datasets" / "discovery.py").exists()
    assert (industrial_root / "datasets" / "local_io.py").exists()
    assert (industrial_root / "evaluation" / "analytics.py").exists()
    assert (industrial_root / "evaluation" / "kernel_learning.py").exists()
    assert (industrial_root / "experiments" / "registry.py").exists()
    assert (industrial_root / "experiments" / "artifacts.py").exists()
    assert (industrial_root / "models" / "kernel_artifacts.py").exists()
    assert (industrial_root / "visualization" / "forecasting.py").exists()


def test_legacy_wrappers_are_not_part_of_industrial_namespace():
    tracked = subprocess.run(
        ["git", "ls-files", "benchmark/industrial/legacy"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tracked.stdout.strip() == ""
