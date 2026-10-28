"""Measure resident kernels and transfer-inclusive CPU->device->CPU calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmark.industrial.experiments.compute_profile.references import (
    digitize_reference, gasf_reference, mtf_reference,
)
from fedot_ind.core.operation.transformation.torch_backend.image.discretize import (
    _digitize_per_sample_bins_torch,
)
from fedot_ind.core.operation.transformation.torch_backend.image.gaf_transformation import GAF
from fedot_ind.core.operation.transformation.torch_backend.image.mtf_transformation import (
    _aggregate_transition_field,
)


def load_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or set(settings) != {
        "seed", "warmup", "repeats", "rtol", "atol", "cases", "memory_budget_bytes"
    }:
        raise ValueError("Unexpected profiling settings")
    for key in ("seed", "warmup", "repeats"):
        if type(settings[key]) is not int or settings[key] < (1 if key == "repeats" else 0):
            raise ValueError(f"Invalid {key}")
    for key in ("rtol", "atol"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
            raise ValueError(f"Invalid {key}")
    if type(settings["memory_budget_bytes"]) is not int or settings["memory_budget_bytes"] <= 0:
        raise ValueError("Invalid memory_budget_bytes")
    if not isinstance(settings["cases"], list) or not settings["cases"]:
        raise ValueError("cases must be a nonempty list")
    for case in settings["cases"]:
        if not isinstance(case, dict):
            raise ValueError("Each operation case must be an object")
        op = case.get("operation")
        required = {"operation", "batch", "length"}
        if op in ("digitize", "mtf"):
            required.add("bins")
        if op == "mtf":
            required.add("image_size")
        if op not in ("digitize", "mtf", "gasf") or set(case) != required:
            raise ValueError("Invalid operation case")
        if any(type(case[k]) is not int or case[k] < 1 for k in required - {"operation"}):
            raise ValueError("Invalid shape")
        if op == "mtf" and case["image_size"] > case["length"]:
            raise ValueError("image_size exceeds length")
        if op == "mtf" and (case["length"] < 2 or case["bins"] < 2):
            raise ValueError("MTF requires at least two timestamps and bins")
    return settings


def _case_inputs(case, generator):
    shape = (case["batch"], case["length"])
    if case["operation"] == "mtf":
        bins = case["bins"]
        binned = torch.randint(bins, shape, generator=generator)
        matrix = torch.rand((case["batch"], bins, bins), generator=generator)
        return (binned, matrix / matrix.sum(-1, keepdim=True))
    x = torch.randn(shape, generator=generator)
    if case["operation"] == "digitize":
        edges = torch.randn((shape[0], case["bins"] - 1), generator=generator).sort(1).values
        return x, edges
    cosine = x.tanh()
    return cosine, (1 - cosine.square()).clamp_min(0).sqrt()


def _functions(case):
    if case["operation"] == "digitize":
        return digitize_reference, _digitize_per_sample_bins_torch
    if case["operation"] == "gasf":
        return gasf_reference, GAF()._gasf
    size = case["image_size"]
    return (lambda x, p: mtf_reference(x, p, size),
            lambda x, p: _aggregate_transition_field(x, p, size))


def _synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _measure(call, device, warmup, repeats):
    for _ in range(warmup):
        result = call()
        _synchronize(device)
        del result
    timings, peaks = [], []
    for _ in range(repeats):
        _synchronize(device)
        if device == "cuda":
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = call()
        _synchronize(device)
        timings.append(time.perf_counter() - start)
        if device == "cuda":
            peaks.append(torch.cuda.max_memory_allocated() - baseline)
        del result
    return {"seconds": timings, "median_seconds": statistics.median(timings),
            "min_seconds": min(timings), "max_seconds": max(timings),
            "peak_allocated_extra_bytes": max(peaks) if peaks else None}


def run(settings, devices):
    if "cuda" in devices and not torch.cuda.is_available():
        raise RuntimeError("Explicit CUDA profiling requested, but CUDA is unavailable")
    generator = torch.Generator(device="cpu").manual_seed(settings["seed"])
    results = []
    for case in settings["cases"]:
        host = _case_inputs(case, generator)
        reference, candidate = _functions(case)
        with torch.no_grad():
            # Float64 avoids cancellation in the old MTF integral image. Integer
            # inputs retain their indices; this is an independent accuracy target.
            accuracy_inputs = tuple(x.double() if x.is_floating_point() else x for x in host)
            expected = reference(*accuracy_inputs)
            for device in devices:
                resident = tuple(x.to(device) for x in host)
                for label, fn in (("reference", reference), ("candidate", candidate)):
                    actual = fn(*resident).cpu()
                    if not torch.isfinite(actual).all():
                        raise ValueError("Nonfinite profiling output")
                    parity = torch.allclose(actual.to(expected.dtype), expected,
                                            rtol=settings["rtol"], atol=settings["atol"])
                    max_error = (actual - expected).abs().max().item()
                    output_bytes = actual.numel() * actual.element_size()
                    for mode in ("resident", "roundtrip"):
                        if mode == "resident":
                            def call(): return fn(*resident)
                        else:
                            def call(): return fn(*(x.to(device) for x in host)).cpu()
                        row = {**case, "device": device, "implementation": label, "mode": mode,
                               "input_bytes": sum(x.numel() * x.element_size() for x in host),
                               "output_bytes": output_bytes, "max_absolute_error": max_error,
                               "within_tolerance": parity}
                        row["transfer_bytes"] = (row["input_bytes"] + output_bytes
                                                 if device == "cuda" and mode == "roundtrip" else 0)
                        row.update(_measure(call, device, settings["warmup"], settings["repeats"]))
                        row["samples_per_second"] = case["batch"] / row["median_seconds"]
                        results.append(row)
                        print(f"{case} {device} {label} {mode}: {row['median_seconds']:.6f}s", flush=True)
                del resident
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("defaults.json"))
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-candidate-parity", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.config)
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "settings": settings,
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "python": platform.python_version(), "platform": platform.platform(),
                "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                "matmul_precision": torch.get_float32_matmul_precision(),
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "cpu_memory_note": "Not measured; CUDA allocator peaks exclude driver and foreign processes.",
                "background_load": "Uncontrolled desktop/background load; no foreign processes stopped."}
    metadata["revision_role"] = "Base HEAD before commit; source_sha256 identifies measured working-tree files"
    source_root = Path(__file__).resolve().parents[4]
    tracked_sources = [Path(__file__), Path(__file__).with_name("references.py")]
    tracked_sources.append(Path(__file__).with_name("placement.py"))
    tracked_sources.extend((source_root / "fedot_ind/core/operation/transformation/torch_backend/image").glob("*.py"))
    tracked_sources.extend((source_root / "fedot_ind/core/operation/transformation/torch_backend/image").glob("*.json"))
    tracked_sources.extend(
        (source_root / "benchmark/industrial/experiments/compute_profile/operation_compute").glob("*.py"))
    tracked_sources.extend(source_root / name for name in (
        "fedot_ind/core/operation/transformation/torch_backend/io.py",
        "fedot_ind/core/operation/transformation/torch_backend/rules.py",
        "fedot_ind/core/optimizer/graph_validation/contracts.py",
    ))
    metadata["source_sha256"] = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(tracked_sources)}
    try:
        metadata["revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        metadata["revision"] = None
    if torch.cuda.is_available():
        try:
            metadata["gpu_status_at_start"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                 "--format=csv,noheader"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            metadata["gpu_status_at_start"] = "nvidia-smi unavailable"
    metadata["results"] = run(settings, args.devices)
    from .placement import probe_mtf_placement
    metadata["placement"] = ([probe_mtf_placement(settings, device) for device in args.devices]
                             if any(case["operation"] == "mtf" for case in settings["cases"]) else [])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "profile.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows = [{k: v for k, v in row.items() if k != "seconds"} for row in metadata["results"]]
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.output_dir / "profile.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if args.require_candidate_parity and any(
        not row["within_tolerance"] for row in rows if row["implementation"] == "candidate"
    ):
        raise SystemExit("Candidate accuracy check failed; evidence retained in output directory")


if __name__ == "__main__":
    main()
