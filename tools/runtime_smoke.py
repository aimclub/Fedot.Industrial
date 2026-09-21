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


def runtime_imports() -> str:
    from fedot.core.data.data import InputData
    from fedot_ind.api.main import FedotIndustrial
    from fedot_ind.core.models.pdl import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor
    from fedot_ind.core.kernel_learning import KernelEnsembleClassifier, KernelEnsembleRegressor

    return ", ".join(cls.__name__ for cls in (
        InputData, FedotIndustrial, PairwiseDifferenceClassifier, PairwiseDifferenceRegressor,
        KernelEnsembleClassifier, KernelEnsembleRegressor,
    ))


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
    probes = (
        ("installed-wheel", installed_package),
        ("stdlib-typing", stdlib_typing),
        ("repository-resources", repository_resources),
        ("cpu-tensor", cpu_tensor),
        ("current-runtime", runtime_imports),
        ("arrow-dataset", dataset_import),
        ("okhs-without-research-dependencies", optional_research_absence),
    )
    results = [run_probe(name, action) for name, action in probes]
    ok = all(result.ok for result in results)
    if args.json:
        print(json.dumps({"ok": ok, "probes": [asdict(result) for result in results]}, indent=2))
    else:
        for result in results:
            print(f"{result.name}: {'OK' if result.ok else 'FAILED'}: {result.detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
