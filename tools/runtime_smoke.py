"""Probe an installed wheel under python -I; progress goes to stderr."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from importlib import import_module, metadata, resources
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Sequence


@dataclass(frozen=True)
class ProbeResult:
    name: str
    ok: bool
    elapsed_seconds: float
    detail: str


def run_probe(name: str, action) -> ProbeResult:
    print(f"Checking {name}...", file=sys.stderr, flush=True)
    started = perf_counter()
    try:
        with redirect_stdout(sys.stderr):
            detail = action()
    except Exception as error:
        return ProbeResult(name, False, round(perf_counter() - started, 3), f"{type(error).__name__}: {error}")
    return ProbeResult(name, True, round(perf_counter() - started, 3), str(detail))


def installed_package() -> str:
    import fedot_ind

    path = Path(fedot_ind.__file__).resolve()
    source_path = Path(__file__).resolve().parents[1] / "fedot_ind"
    if path.is_relative_to(source_path):
        raise RuntimeError("Source checkout imported instead of the installed distribution; use python -I")
    if metadata.version("fedot-ind") != fedot_ind.__version__:
        raise RuntimeError("Runtime version differs from installed package metadata")
    return str(path)


def repository_resources() -> str:
    root = resources.files("fedot_ind.core.repository.data")
    for name in (
        "default_operation_params.json",
        "industrial_data_operation_repository.json",
            "industrial_model_repository.json"):
        raw = root.joinpath(name).read_text(encoding="utf-8")
        if not isinstance(json.loads(raw), dict):
            raise ValueError(f"Invalid repository resource: {name}")
    if not root.joinpath("ts_benchmark_metadata.csv").is_file():
        raise FileNotFoundError("Runtime benchmark metadata resource is missing")
    return "Operation parameters, data/model repositories and benchmark metadata loaded"


def stdlib_typing() -> str:
    import typing
    import sysconfig

    path = Path(typing.__file__).resolve()
    if path != (Path(sysconfig.get_path("stdlib")) / "typing.py").resolve():
        raise RuntimeError(f"Legacy typing distribution shadows the standard library: {path}")
    return str(path)


def current_runtime_imports() -> str:
    from fedot.core.data.input_data.data import InputData
    from fedot_ind.api.main import FedotIndustrial
    from fedot_ind.core.models.pdl import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor
    from fedot_ind.core.kernel_learning import KernelEnsembleClassifier, KernelEnsembleRegressor

    return ", ".join(cls.__name__ for cls in (
        InputData, FedotIndustrial, PairwiseDifferenceClassifier, PairwiseDifferenceRegressor,
        KernelEnsembleClassifier, KernelEnsembleRegressor,
    ))


def tensor_runtime_imports() -> str:
    import numpy as np
    from fedot import TensorData, create_data
    from fedot_ind.integration.fedot import (
        DataProfile, DataStage, IntegrationTask, build_data_plan, normalize_input_data,
    )

    features = np.arange(12, dtype=float).reshape(6, 2)
    target = features[:, 0] * 2 + 1
    plan = build_data_plan(
        features,
        profile=DataProfile.TENSOR,
        task=IntegrationTask.REGRESSION,
        stage=DataStage.TRAIN,
    )
    prepared = normalize_input_data(features, plan)
    train = create_data(prepared.values, target=target, task="regression")
    predicted = create_data(prepared.values.copy(), from_data=train)
    if not isinstance(train, TensorData) or not isinstance(predicted, TensorData):
        raise TypeError("FEDOT create_data did not return TensorData")
    if tuple(train.features.shape) != (6, 2) or tuple(predicted.features.shape) != (6, 2):
        raise ValueError("TensorData shape changed across train/predict preparation")
    return f"{TensorData.__name__}, profile={DataProfile.TENSOR.value}"


def regression_runtime() -> str:
    import numpy as np
    from fedot_ind.integration.fedot import create_regression_runtime

    first = np.linspace(0.0, 29.0, 30)
    train = np.column_stack((first, np.square(first + 1) / 10.0))
    target = 3 * train[:, 0] - 2 * train[:, 1] + 1
    predict = np.array([[6.0, 2.0], [7.0, 5.0]])
    expected = 3 * predict[:, 0] - 2 * predict[:, 1] + 1
    runtime = create_regression_runtime("tensor")
    try:
        runtime.fit(train, target)
        result = runtime.predict(predict)
    finally:
        runtime.close()
    np.testing.assert_allclose(np.asarray(result.values).reshape(-1), expected, rtol=1e-6, atol=1e-5)
    np.testing.assert_array_equal(result.idx, np.arange(2))
    return f"profile=tensor, predictions={result.values.shape[0]}"


def cpu_tensor() -> str:
    import torch

    data = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    result = data @ data.T
    if result.shape != (2, 2) or not torch.isfinite(result).all():
        raise ValueError("CPU tensor calculation produced invalid output")
    return f"torch {torch.__version__}, device={result.device}"


def dataset_import() -> str:
    from datasets import Dataset

    data = Dataset.from_dict({"value": [1, 2, 3], "label": [0, 1, 0]})
    if data.num_rows != 3 or data[1]["value"] != 2:
        raise ValueError("Arrow dataset roundtrip failed")
    return f"datasets {metadata.version('datasets')}, pyarrow {metadata.version('pyarrow')}"


def optional_research_absence() -> str:
    module = import_module("fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.okhs")
    return module.__name__


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    common_probes = (
        ("installed-wheel", installed_package),
        ("stdlib-typing", stdlib_typing),
        ("cpu-tensor", cpu_tensor),
    )
    integration_probes = (
        ("repository-resources", repository_resources),
        ("current-runtime", current_runtime_imports),
        ("tensor-runtime", tensor_runtime_imports),
        ("arrow-dataset", dataset_import),
        ("okhs-without-research-dependencies", optional_research_absence),
        ("integration-regression", regression_runtime),
    )
    probes = common_probes + integration_probes
    results = [run_probe(name, action) for name, action in probes]
    ok = all(result.ok for result in results)
    if args.json:
        print(json.dumps({"ok": ok, "profile": "current",
                          "probes": [asdict(result) for result in results]}, indent=2))
    else:
        for result in results:
            print(f"{result.name}: {'OK' if result.ok else 'FAILED'}: {result.detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
