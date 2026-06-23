from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEV_GUIDE_PATH = PROJECT_ROOT / "docs" / "dev_guide" / "benchmark_infrastructure.md"
KERNEL_LEARNING_RUNBOOK_PATH = (
    PROJECT_ROOT / "docs" / "dev_guide" / "kernel_learning_benchmark_runbook.md"
)
RESULTS_README_PATH = PROJECT_ROOT / "benchmark" / "results" / "README.md"
SHOWCASE_MANIFEST_PATH = (
    PROJECT_ROOT / "benchmark" / "results" / "showcase" / "showcase_manifest.json"
)


def test_benchmark_infrastructure_doc_covers_public_contract() -> None:
    text = DEV_GUIDE_PATH.read_text(encoding="utf-8")

    required_fragments = [
        "benchmark.industrial",
        "BenchmarkSuiteConfig",
        "run_registered_suite",
        "benchmark.industrial.evaluation.aggregation",
        "render_benchmark_aggregate_artifacts",
        "benchmark/results/showcase",
        "python -m benchmark.results.showcase",
        "benchmark/experiments/kernel_learning/defaults.json",
        "benchmark/results/v2_kernel_learning",
        "Adding A New Benchmark Direction",
        "What Not To Do",
    ]

    missing = [fragment for fragment in required_fragments if fragment not in text]
    assert not missing


def test_benchmark_docs_link_to_infrastructure_contract() -> None:
    expected_link = "docs/dev_guide/benchmark_infrastructure.md"

    assert expected_link in KERNEL_LEARNING_RUNBOOK_PATH.read_text(encoding="utf-8")
    assert expected_link in RESULTS_README_PATH.read_text(encoding="utf-8")


def test_result_showcase_manifest_is_documented() -> None:
    doc_text = DEV_GUIDE_PATH.read_text(encoding="utf-8")

    assert SHOWCASE_MANIFEST_PATH.exists()
    assert "benchmark/results/showcase/showcase_manifest.json" in doc_text
