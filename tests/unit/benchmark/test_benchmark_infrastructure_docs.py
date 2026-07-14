from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCS_README_PATH = PROJECT_ROOT / "docs" / "README.md"
RESULTS_README_PATH = PROJECT_ROOT / "benchmark" / "results" / "README.md"
SHOWCASE_MANIFEST_PATH = (
    PROJECT_ROOT / "benchmark" / "results" / "showcase" / "showcase_manifest.json"
)


def test_docs_readme_links_to_wiki_contract_pages() -> None:
    text = DOCS_README_PATH.read_text(encoding="utf-8")

    required_fragments = [
        "https://github.com/aimclub/Fedot.Industrial/wiki/Fedot-Industrial-KL-Documentation",
        "https://github.com/aimclub/Fedot.Industrial/wiki/Benchmark-Infrastructure",
        "https://github.com/aimclub/Fedot.Industrial/wiki/Kernel-Learning-Benchmark-Runbook",
        "https://github.com/aimclub/Fedot.Industrial/wiki/Kernel-Learning-MVP-API",
        "https://github.com/aimclub/Fedot.Industrial/wiki/Benchmark-Industrial-Migration",
        "https://github.com/aimclub/Fedot.Industrial/wiki#forecasting-refactor-plans",
        "https://github.com/aimclub/Fedot.Industrial/wiki/Forecasting-Prerelease-270426-Overview",
        "Repository docs should stay short and operational",
    ]

    missing = [fragment for fragment in required_fragments if fragment not in text]
    assert not missing


def test_benchmark_results_readme_points_to_wiki_infrastructure_contract() -> None:
    text = RESULTS_README_PATH.read_text(encoding="utf-8")

    assert "https://github.com/aimclub/Fedot.Industrial/wiki/Benchmark-Infrastructure" in text
    assert "benchmark/results/showcase/showcase_manifest.json" in text


def test_result_showcase_manifest_exists_after_docs_move() -> None:
    assert SHOWCASE_MANIFEST_PATH.exists()