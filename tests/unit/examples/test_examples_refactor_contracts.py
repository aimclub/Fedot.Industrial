from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = REPOSITORY_ROOT / "examples"
DATA_ROOT = EXAMPLES_ROOT / "utils" / "data"
ARTIFACT_ROOT = EXAMPLES_ROOT / "artifacts" / "cloud_bundle"
DATA_TASK_DIRS = ("ts_classification", "ts_regression", "forecasting", "anomaly_detection")
FORBIDDEN_PY_TOKENS = ("benchmark.v2", "benchmark_v2_manifest", "ApiTemplate")
FORBIDDEN_NOTEBOOK_TOKENS = FORBIDDEN_PY_TOKENS + ("plt.savefig",)


def test_obsolete_example_directories_are_removed() -> None:
    assert not (EXAMPLES_ROOT / "benchmark_v2").exists()
    assert not (EXAMPLES_ROOT / "outdated_examples").exists()
    assert not (EXAMPLES_ROOT / "automl_example").exists()
    assert not (EXAMPLES_ROOT / "current_api").exists()
    assert not (EXAMPLES_ROOT / "data").exists()
    assert not (EXAMPLES_ROOT / "rkhs_okhs").exists()
    assert not (EXAMPLES_ROOT / "real_world_examples" / "benchmark_example" / "analysis of results").exists()
    assert not (EXAMPLES_ROOT / "real_world_examples" / "benchmark_example" / "archive").exists()


def test_data_task_layout_has_readmes_and_tracked_fixtures() -> None:
    assert (DATA_ROOT / "README.md").is_file()

    for task_dir in DATA_TASK_DIRS:
        assert (DATA_ROOT / task_dir / "README.md").is_file()

    assert (DATA_ROOT / "ts_classification" / "ItalyPowerDemand_fake").is_dir()
    assert (DATA_ROOT / "ts_regression" / "MadridPM10Quality-no-missing").is_dir()
    assert (DATA_ROOT / "forecasting" / "nbeats").is_dir()
    assert (DATA_ROOT / "forecasting" / "ice_forecasting").is_dir()
    assert (DATA_ROOT / "anomaly_detection" / "skab").is_dir()
    assert (DATA_ROOT / "anomaly_detection" / "box_anomaly_detection").is_dir()


def test_data_loader_resolves_task_based_fixture_paths() -> None:
    from fedot_ind.tools.loader import resolve_dataset_parent_path, resolve_skab_data_root
    from fedot_ind.tools.serialisation.path_lib import EXAMPLES_DATA_PATH

    classification_parent = Path(resolve_dataset_parent_path(EXAMPLES_DATA_PATH, "ItalyPowerDemand_fake"))
    regression_parent = Path(resolve_dataset_parent_path(EXAMPLES_DATA_PATH, "MadridPM10Quality-no-missing"))
    skab_root = resolve_skab_data_root()

    assert classification_parent == DATA_ROOT / "ts_classification"
    assert regression_parent == DATA_ROOT / "ts_regression"
    assert skab_root == DATA_ROOT / "anomaly_detection" / "skab"


def test_migrated_python_examples_do_not_use_removed_api_tokens() -> None:
    scanned_files = 0
    tracked = subprocess.run(
        ["git", "ls-files", "examples"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for relative_path in tracked.stdout.splitlines():
        path = REPOSITORY_ROOT / relative_path
        if path.suffix != ".py" or not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        scanned_files += 1
        assert not any(token in content for token in FORBIDDEN_PY_TOKENS), path

    assert scanned_files > 0


def test_current_manifest_examples_use_current_json_version() -> None:
    manifest_dir = EXAMPLES_ROOT / "utils" / "current_api" / "manifests"

    for path in manifest_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["version"] == "benchmark_industrial_manifest@1"


def test_automl_and_kernel_learning_config_previews_are_typed(tmp_path: Path) -> None:
    from benchmark.industrial import TaskType
    from examples.tools_example.current_api import (
        anomaly_detection_context,
        build_forecasting_preview,
        build_kernel_learning_preview,
        invoke_tool,
        list_tool_specs,
    )

    forecasting_config = build_forecasting_preview(tmp_path / "forecasting")
    kernel_config = build_kernel_learning_preview(tmp_path / "kernel")
    anomaly_context = anomaly_detection_context()
    anomaly_preview = invoke_tool(
        "industrial_detect_anomalies",
        {"folder": "valve1", "dataset": "0", "output_dir": str(tmp_path / "anomaly")},
    )

    assert forecasting_config.task_type is TaskType.FORECASTING
    assert forecasting_config.artifact_spec.persist_on_run is False
    assert kernel_config.task_type is TaskType.TS_CLASSIFICATION
    assert kernel_config.artifact_spec.persist_on_run is False
    assert Path(anomaly_context["data_root"]) == DATA_ROOT / "anomaly_detection" / "liman" / "vibro"
    assert anomaly_preview["status"] == "dry_run"
    assert Path(anomaly_preview["data"]["target_path"]).parts[-2:] == ("valve1", "0.csv")
    tool_names = {tool["name"] for tool in list_tool_specs()}
    assert {"industrial_train_model", "industrial_run_evolution", "industrial_detect_anomalies"}.issubset(tool_names)
    assert "industrial_tsc_smoke" in tool_names
    response = invoke_tool("industrial_forecasting_config_preview", {"output_dir": str(tmp_path / "tool")})
    assert response["status"] == "dry_run"
    assert response["data"]["config"]["task_type"] == "forecasting"


def test_tools_example_exposes_mcp_ready_action_contracts(tmp_path: Path) -> None:
    from examples.tools_example import invoke_tool, list_tool_specs

    specs = list_tool_specs()
    by_name = {spec["name"]: spec for spec in specs}

    for name in (
        "industrial_load_data",
        "industrial_train_model",
        "industrial_run_evolution",
        "industrial_run_pdl_training",
        "industrial_detect_anomalies",
    ):
        assert by_name[name]["input_schema"]["type"] == "object"
        assert by_name[name]["capability"]

    train = invoke_tool(
        "industrial_train_model",
        {"task_type": "ts_classification", "dataset_name": "Lightning7", "output_dir": str(tmp_path / "train")},
    )
    pdl = invoke_tool(
        "industrial_run_pdl_training",
        {"task_type": "ts_regression", "dataset_name": "NaturalGasPricesSentiment"},
    )
    evolution = invoke_tool("industrial_run_evolution", {"datasets": ["Lightning7"]})
    anomaly = invoke_tool("industrial_detect_anomalies", {"folder": "valve1", "dataset": "0"})
    unknown = invoke_tool("does_not_exist", {})

    assert train["status"] == "dry_run"
    assert train["data"]["config"]["task_type"] == "ts_classification"
    assert pdl["status"] == "dry_run"
    assert pdl["data"]["config"]["models"][0]["adapter_name"] == "pdl_regressor"
    assert evolution["status"] == "dry_run"
    assert anomaly["status"] == "dry_run"
    assert unknown["status"] == "failed"
    assert unknown["error"]["code"] == "unknown_tool"


def test_real_world_context_entrypoints_are_importable_and_do_not_load_external_data() -> None:
    from examples.real_world_examples.benchmark_example.detection.ts_anomaly_detection_skab_bench import (
        build_skab_benchmark_context,
    )
    from examples.real_world_examples.benchmark_example.forecasting.kaggle_forecasting import (
        build_kaggle_forecasting_context,
    )
    from examples.real_world_examples.current_api import eeg_classification_context
    from examples.real_world_examples.industrial_examples.debet_forecasting.main import (
        build_debet_forecasting_context,
    )

    skab = build_skab_benchmark_context("valve1")
    kaggle = build_kaggle_forecasting_context("examples/utils/data/forecasting/kaggle_inventory")
    eeg = eeg_classification_context()
    debet = build_debet_forecasting_context()

    assert skab["context"]["task_type"] == "anomaly_detection"
    assert Path(skab["context"]["data_root"]) == DATA_ROOT / "anomaly_detection" / "skab"
    assert kaggle["context"]["task_type"] == "forecasting"
    assert kaggle["context"]["train_path"].endswith("train.csv")
    assert eeg["task_type"] == "ts_classification"
    assert Path(eeg["data_root"]) == DATA_ROOT / "ts_classification" / "eeg"
    assert debet["context"]["task_type"] == "forecasting"
    assert Path(debet["context"]["data_root"]) == DATA_ROOT / "forecasting" / "debet"


def test_real_world_analysis_notebooks_are_thin_current_api_entries() -> None:
    from examples.real_world_examples.benchmark_example.analysis_of_results import (
        available_analysis_names,
        build_analysis_diagnostics_frame,
        build_analysis_result_frame,
        build_analysis_source_metadata_frame,
        build_kernel_learning_reference_model_specs,
        build_ucr_two_stage_context,
        preflight_summary as analysis_preflight_summary,
    )

    analysis_dir = EXAMPLES_ROOT / "real_world_examples" / "benchmark_example" / "analysis_of_results"
    notebook_paths = sorted(analysis_dir.glob("*.ipynb"))

    assert set(available_analysis_names()) == {
        "analysis_multi_clf",
        "analysis_regr",
        "analysis_uni_clf",
        "m4_analysis",
        "pdl_uni_benchmark",
    }
    assert analysis_preflight_summary()["feature_generators"]
    uni = build_analysis_result_frame("analysis_uni_clf")
    regr = build_analysis_result_frame("analysis_regr")
    m4 = build_analysis_result_frame("m4_analysis")
    assert uni["dataset_name"].nunique() > 100
    assert "KernelEnsembleClassifier_score_baseline_summary" in set(uni["model_name"])
    assert "RDST" in set(regr["model_name"])
    assert "Fedot_Industrial_legacy_baseline" in set(regr["model_name"])
    assert "LaggedRidgeForecaster" in set(m4["model_name"])
    assert not build_analysis_diagnostics_frame("analysis_uni_clf").empty
    assert build_analysis_source_metadata_frame("analysis_regr")["exists_locally"].all()
    assert build_ucr_two_stage_context()["scenario"] == "ucr_two_stage"
    assert "KernelEnsembleClassifier_adaptive_all_non_topological" in {
        model.display_name for model in build_kernel_learning_reference_model_specs("ts_classification")
    }
    assert notebook_paths
    for path in notebook_paths:
        content = path.read_text(encoding="utf-8")
        assert "benchmark.industrial" in content
        assert "analysis_of_results.current_api" in content
        assert not any(token in content for token in FORBIDDEN_NOTEBOOK_TOKENS), path

    artifact_root = ARTIFACT_ROOT / "benchmark_showcase" / "analysis_of_results"
    for artifact_name in (*available_analysis_names(), "pipeline_population"):
        assert (artifact_root / artifact_name / "summary.md").is_file()
        assert any((artifact_root / artifact_name / "plots").glob("*.png"))
        assert (artifact_root / artifact_name / "tables" / "coverage.csv").is_file()
    assert (artifact_root / "analysis_uni_clf" / "tables" / "model_diagnostics.csv").is_file()
    assert (
        artifact_root
        / "forecasting_model_comparison"
        / "summary.md"
    ).is_file()


def test_real_world_notebooks_do_not_use_removed_api_tokens() -> None:
    notebook_paths = sorted((EXAMPLES_ROOT / "real_world_examples").rglob("*.ipynb"))

    assert notebook_paths
    for path in notebook_paths:
        content = path.read_text(encoding="utf-8")
        assert not any(token in content for token in FORBIDDEN_NOTEBOOK_TOKENS), path


def test_forecast_comparison_artifacts_are_not_copied_between_scenarios() -> None:
    forecast_plots = sorted(
        (ARTIFACT_ROOT / "benchmark_showcase").rglob("multi_model_forecast.png")
    )
    hashes: dict[str, tuple[Path, dict[str, object]]] = {}
    for path in forecast_plots:
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        metadata_path = path.parent.parent / "source_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        if digest in hashes:
            previous_path, previous_metadata = hashes[digest]
            assert metadata == previous_metadata, (
                f"{path} duplicates forecast plot from {previous_path} "
                "but source_metadata.json differs."
            )
        hashes[digest] = (path, metadata)


def test_real_world_data_delivery_and_domain_scenarios_are_manifest_driven(tmp_path: Path) -> None:
    from examples.real_world_examples.current_api import external_data_summary
    from examples.real_world_examples.industrial_examples import (
        build_scenario_context,
        build_scenario_forecast_preview,
        list_domain_scenarios,
        render_scenario_forecast_pack,
        render_scenario_preview_pack,
    )

    summary = external_data_summary()
    scenarios = list_domain_scenarios()
    eeg = build_scenario_context("eeg_classification")
    equipment = build_scenario_context("equipment_classification")
    manifest = render_scenario_preview_pack("eeg_classification", output_dir=tmp_path / "eeg_preview")
    forecast = build_scenario_forecast_preview("bitcoin_forecasting")
    forecast_manifest = render_scenario_forecast_pack(
        "bitcoin_forecasting",
        output_dir=tmp_path / "bitcoin_forecast",
    )

    assert summary["delivery_mode"] == "public_yandex_disk_archive_plus_optional_dvc"
    assert "real_world_archive_composition_results" in summary["source_keys"]
    assert (DATA_ROOT / "benchmark_history" / "real_world_archive" / "archive_manifest.json").is_file()
    assert "eeg_classification" in scenarios
    assert "ethereum_regression" in scenarios
    assert eeg["task_type"] == "ts_classification"
    assert Path(eeg["artifact_root"]).is_relative_to(REPOSITORY_ROOT)
    assert Path(eeg["data_path"]).is_relative_to(REPOSITORY_ROOT)
    assert "KernelEnsembleClassifier" in eeg["models"]
    assert "KernelEnsembleClassifier_adaptive_all_non_topological" in eeg["models"]
    assert "LaggedRidgeForecaster" in build_scenario_context("bitcoin_forecasting")["models"]
    assert "KernelEnsembleForecaster_embedding_nystrom_okhs" in forecast["forecasts"]
    assert equipment["domain"] == "equipment_monitoring"
    assert any(record.kind == "plot" for record in manifest)
    assert any(record.kind == "plot" for record in forecast_manifest)


def test_rkhs_okhs_forecasting_entrypoint_builds_publication_ready_preview(tmp_path: Path) -> None:
    from examples.real_world_examples.benchmark_example.rkhs_okhs.forecasting.okhs_forecasting import (
        build_rkhs_okhs_forecasting_example,
    )

    preview = build_rkhs_okhs_forecasting_example(tmp_path / "rkhs")

    assert preview["scenario"] == "rkhs_okhs_forecasting"
    assert preview["benchmark"]["task_type"] == "forecasting"
    assert "OKHSForecaster" in preview["benchmark"]["models"]
