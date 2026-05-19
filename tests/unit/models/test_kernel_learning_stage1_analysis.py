from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_analysis_module():
    module_path = PROJECT_ROOT / "benchmark" / "v2" / "kernel_learning_analysis.py"
    spec = importlib.util.spec_from_file_location("kernel_learning_analysis_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _record(dataset_name: str, weights: tuple[float, float], important: tuple[str, ...]) -> dict:
    return {
        "run_id": "stage1_test",
        "dataset_name": dataset_name,
        "model_name": "KernelEnsembleClassifier",
        "status": "success",
        "kernel_selection": {
            "selection_report": {
                "generator_names": ["wavelet_extractor", "recurrence_extractor"],
                "weights": list(weights),
                "scores": {"wavelet_extractor": weights[0], "recurrence_extractor": weights[1]},
                "alignments": {"wavelet_extractor": weights[0] + 0.1, "recurrence_extractor": weights[1] + 0.1},
                "complexities": {"wavelet_extractor": 1.0, "recurrence_extractor": 1.2},
                "redundancies": {"wavelet_extractor": 0.1, "recurrence_extractor": 0.2},
            },
            "important_generators": list(important),
            "important_weights": [weights[0] if name == "wavelet_extractor" else weights[1] for name in important],
            "kernel_importance": {
                "items": [
                    {
                        "name": name,
                        "weight": weights[0] if name == "wavelet_extractor" else weights[1],
                        "rank": index + 1,
                    }
                    for index, name in enumerate(important)
                ]
            },
        },
        "kernel_diagnostics": {
            "kernels": [
                {
                    "name": "wavelet_extractor",
                    "diagnostics": {
                        "min_eigenvalue": 0.01,
                        "condition_number": 10.0,
                        "kernel": "rbf",
                        "gamma": 0.1,
                        "feature_generator": {
                            "operations": ["wavelet_basis", "quantile_extractor_torch"],
                            "n_features": 12,
                            "torch_device": "cpu",
                        },
                    },
                    "complexity": {"kernel_complexity": 1.0},
                    "is_psd": True,
                    "train_kernel_shape": [4, 4],
                    "test_kernel_shape": [2, 4],
                },
                {
                    "name": "recurrence_extractor",
                    "diagnostics": {
                        "min_eigenvalue": -0.1,
                        "condition_number": 1000.0,
                        "kernel": "rbf",
                        "gamma": 0.2,
                        "feature_generator": {"operations": ["recurrence_extractor"], "n_features": 3},
                    },
                    "complexity": {"kernel_complexity": 1.2},
                    "is_psd": False,
                    "train_kernel_shape": [4, 4],
                    "test_kernel_shape": [2, 4],
                },
            ]
        },
    }


def test_stage1_analysis_builds_stable_tables_and_report(tmp_path: Path):
    module = _load_analysis_module()
    run_dir = tmp_path / "stage1_run"
    _write_jsonl(
        run_dir / "records" / "kernel_diagnostics.jsonl",
        [
            _record("DatasetB", (0.7, 0.3), ("wavelet_extractor",)),
            _record("DatasetA", (0.2, 0.8), ("recurrence_extractor",)),
        ],
    )
    aggregate_dir = run_dir / "aggregate"
    aggregate_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "dataset_name": "DatasetA",
                "model_name": "KernelEnsembleClassifier",
                "status": "success",
                "accuracy": 0.9,
                "balanced_accuracy": 0.8,
                "f1_macro": 0.75,
            },
            {
                "dataset_name": "DatasetB",
                "model_name": "KernelEnsembleClassifier",
                "status": "success",
                "accuracy": 0.8,
                "balanced_accuracy": 0.7,
                "f1_macro": 0.65,
            },
        ]
    ).to_csv(aggregate_dir / "runs.csv", index=False)

    analysis = module.render_kernel_stage1_summary_report(run_dir)

    assert analysis.summary["dataset_count"] == 2
    assert analysis.summary["psd_failure_count"] == 2
    assert list(analysis.dataset_summary["dataset_name"]) == ["DatasetA", "DatasetB"]
    assert analysis.generator_summary.iloc[0]["top1_count"] == 1
    report_text = (run_dir / "analysis" / "summary_report.md").read_text(encoding="utf-8")
    assert "| dataset_name |" in report_text
    assert (run_dir / "analysis" / "generator_importance_summary.csv").exists()


def test_stage1_analysis_falls_back_to_kernel_selection_jsonl(tmp_path: Path):
    module = _load_analysis_module()
    run_dir = tmp_path / "stage1_run"
    record = _record("DatasetA", (0.6, 0.4), ("wavelet_extractor",))
    record.pop("kernel_diagnostics")
    _write_jsonl(run_dir / "records" / "kernel_selection.jsonl", [record])

    analysis = module.analyze_kernel_stage1_run(run_dir)

    assert analysis.summary["dataset_count"] == 1
    assert set(analysis.kernel_diagnostics["generator_name"]) == {"wavelet_extractor", "recurrence_extractor"}
