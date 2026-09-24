from types import SimpleNamespace

import pytest

from tools import runtime_smoke


def test_probe_preserves_failure_type_and_progress(capsys):
    def failed():
        raise ImportError("missing native runtime")

    result = runtime_smoke.run_probe("native-module", failed)
    assert not result.ok and result.detail == "ImportError: missing native runtime"
    assert result.elapsed_seconds >= 0
    assert capsys.readouterr().err == "Checking native-module...\n"


def test_probe_success_is_structured(capsys):
    result = runtime_smoke.run_probe("pure", lambda: "confirmed")
    assert result.ok and result.name == "pure" and result.detail == "confirmed"
    assert capsys.readouterr().out == ""


def test_dependency_stdout_cannot_corrupt_json(capsys):
    def noisy_import():
        print("third-party initialization")
        return "ready"

    assert runtime_smoke.run_probe("dependency", noisy_import).ok
    output = capsys.readouterr()
    assert output.out == "" and "third-party initialization" in output.err


def test_resource_reader_checks_all_json_and_csv_inputs(tmp_path, monkeypatch):
    names = ("default_operation_params.json",
             "industrial_data_operation_repository.json",
             "industrial_model_repository.json")
    for name in names:
        (tmp_path / name).write_text("{}")
    (tmp_path / "ts_benchmark_metadata.csv").write_text("dataset,task\n")
    monkeypatch.setattr(runtime_smoke.resources, "files", lambda name: tmp_path)
    assert "repositories" in runtime_smoke.repository_resources()
    (tmp_path / names[1]).write_text("[]")
    with pytest.raises(ValueError, match="Invalid repository resource"):
        runtime_smoke.repository_resources()
    (tmp_path / names[1]).write_text("{}")
    (tmp_path / "ts_benchmark_metadata.csv").unlink()
    with pytest.raises(FileNotFoundError, match="metadata"):
        runtime_smoke.repository_resources()


def test_source_checkout_cannot_pass_as_an_installed_wheel(monkeypatch):
    import sys
    from pathlib import Path

    source_file = Path(runtime_smoke.__file__).resolve().parents[1] / "fedot_ind/__init__.py"
    monkeypatch.setitem(sys.modules, "fedot_ind", SimpleNamespace(__file__=str(source_file), __version__="0.5.0"))
    with pytest.raises(RuntimeError, match="Source checkout"):
        runtime_smoke.installed_package()


def test_installed_version_must_match_metadata(monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "fedot_ind",
        SimpleNamespace(
            __file__=str(
                tmp_path /
                "__init__.py"),
            __version__="0.5.0"))
    monkeypatch.setattr(runtime_smoke.metadata, "version", lambda name: "another")
    with pytest.raises(RuntimeError, match="version differs"):
        runtime_smoke.installed_package()
    monkeypatch.setattr(runtime_smoke.metadata, "version", lambda name: "0.5.0")
    assert runtime_smoke.installed_package() == str(tmp_path / "__init__.py")


@pytest.mark.parametrize("failure", [False, True])
def test_runtime_cli_keeps_json_separate_from_progress(
        monkeypatch, capsys, failure):
    import json

    names = (
        "installed_package",
        "stdlib_typing",
        "repository_resources",
        "cpu_tensor",
        "current_runtime_imports",
        "tensor_runtime_imports",
        "regression_runtime",
        "dataset_import",
        "optional_research_absence")
    for name in names:
        monkeypatch.setattr(runtime_smoke, name, lambda: "verified")
    if failure:
        def bad_import():
            raise ImportError("runtime unavailable")
        monkeypatch.setattr(runtime_smoke, "current_runtime_imports", bad_import)
    assert runtime_smoke.main(["--json"]) == int(failure)
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result["ok"] is not failure and len(result["probes"]) == 9
    assert result["profile"] == "current"
    assert output.err.count("Checking ") == 9
    assert sum(not probe["ok"] for probe in result["probes"]) == int(failure)
