"""Read and verify the existing CPU runtime policy without changing Torch flags."""

from importlib import resources
import json


def expected_runtime():
    return json.loads(resources.files(__package__).joinpath('runtime_defaults.json').read_text(encoding='utf-8'))


def runtime_record(torch_module):
    return {'torch_threads': torch_module.get_num_threads(), 'cuda_visible': torch_module.cuda.is_available(),
            'torch_flags': {
                'deterministic_algorithms': torch_module.are_deterministic_algorithms_enabled(),
                'deterministic_warn_only': torch_module.is_deterministic_algorithms_warn_only_enabled(),
                'float32_matmul_precision': torch_module.get_float32_matmul_precision(),
                'cudnn_deterministic': torch_module.backends.cudnn.deterministic,
                'cudnn_benchmark': torch_module.backends.cudnn.benchmark,
                'torch_cuda_build': torch_module.version.cuda,
    }}


def runtime_matches(record):
    expected = expected_runtime()
    actual = {key: record.get(key) for key in expected}
    # Canonical JSON also distinguishes bool from int, unlike dict equality.
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)


def require_runtime(torch_module):
    if torch_module.version.cuda is not None:
        raise RuntimeError('CPU study admission requires a CPU-only Torch build')
    actual = runtime_record(torch_module)
    if not runtime_matches(actual):
        raise RuntimeError(f'CPU study admission differs from the frozen Torch policy: {actual}')
    return actual
