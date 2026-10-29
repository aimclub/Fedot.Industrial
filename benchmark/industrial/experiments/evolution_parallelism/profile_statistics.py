"""Profile saved TRAIN arrays only; never fit a model or load a dataset."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from .profiling import profile_call


def profile_statistics(inputs, config, output_dir, *, rows=16, role='refit', representation='raw'):
    import numpy as np
    import torch
    from fedot.core.operations.operation_parameters import OperationParameters
    from fedot_ind.core.operation.transformation.representation.statistical.quantile_extractor import QuantileExtractor
    from fedot_ind.core.operation.transformation.torch_backend.statistical.quantile_extractor import TorchQuantileExtractor

    if type(rows) is not int or rows < 1 or role not in ('refit', 'fold0_train', 'fold0_validation'):
        raise ValueError('Require a positive row count and an inner TRAIN input role')
    if representation not in ('raw', 'wavelet'):
        raise ValueError('Unsupported input representation')
    parameters = json.loads(Path(config).read_text('utf-8'))['settings']['operation_parameters']
    with np.load(inputs, allow_pickle=False) as arrays:
        values = arrays[role + '_features'][:rows].copy()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    flags = dict(threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                 deterministic=torch.are_deterministic_algorithms_enabled())
    if representation == 'wavelet':
        from fedot_ind.integration.fedot.compatibility import InputData
        from fedot.core.repository.tasks import Task, TaskTypesEnum
        from fedot.core.repository.dataset_types import DataTypesEnum
        from fedot_ind.core.operation.transformation.basis.wavelet import WaveletBasisImplementation
        basis = WaveletBasisImplementation(OperationParameters(**parameters['wavelet_basis']))
        task = json.loads(Path(config).read_text('utf-8'))['request']['task']
        data = InputData(features=values, idx=np.arange(len(values)), target=None,
                         task=Task(TaskTypesEnum[task]), data_type=DataTypesEnum.multi_ts)
        values = basis._transform(data)
    records = []
    for name, factory in (('quantile_extractor', QuantileExtractor), ('quantile_extractor_torch', TorchQuantileExtractor)):
        params = parameters[name]
        if params.get('use_cache') is not False:
            raise ValueError('Profiling requires the recorded cache-disabled configuration')
        instance = factory(OperationParameters(**params))
        instance._transform(SimpleNamespace(features=values))  # Warmup is outside profiling.
        output = profile_call(instance._transform, root / name, SimpleNamespace(features=values), label=name)
        np.save(root / name / 'features.npy', output, allow_pickle=False)
        records.append(dict(operation=name, parameters=params, input_shape=list(values.shape),
                            output_shape=list(output.shape), output_dtype=str(output.dtype)))
    request = dict(schema_version='industrial_statistics_profile@1', role=role, rows=rows,
                   representation=representation, warmup_calls=1, torch_flags_before=flags,
                   inputs_sha256=sha256(Path(inputs).read_bytes()).hexdigest(),
                   config_sha256=sha256(Path(config).read_bytes()).hexdigest(),
                   official_test_used=False, training_enabled=False, diagnostic_run=True, records=records)
    request['torch_flags_after'] = dict(threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                                        deterministic=torch.are_deterministic_algorithms_enabled())
    if flags != request['torch_flags_after']:
        raise RuntimeError('Global Torch settings changed')
    (root / 'request.json').write_text(json.dumps(request, indent=2) + '\n', encoding='utf-8')
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--rows', type=int, default=16)
    parser.add_argument('--representation', choices=('raw', 'wavelet'), default='raw')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(execution=False, training_enabled=False, official_test_used=False)))
        return
    if any(value is None for value in (args.inputs, args.config, args.output_dir)):
        parser.error('--execute requires --inputs, --config and a new --output-dir')
    print(json.dumps(profile_statistics(args.inputs, args.config, args.output_dir, rows=args.rows,
                                        representation=args.representation)))


if __name__ == '__main__':
    main()
