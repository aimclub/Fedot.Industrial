"""Run a synthetic reconstruction and forecasting diagnostic without model training."""

import argparse
import json
from pathlib import Path

from benchmark.industrial.evaluation.decomposition_protocol import (
    SyntheticSpec, component_errors, decomposition_control, seasonal_control, synthetic_case,
)
from benchmark.industrial.protocols.json_io import read_protocol_json


def verify_decomposition(spec):
    case = synthetic_case(spec, 'evaluation')
    parts = case['components']
    classical = decomposition_control(case['values'], case['period'])
    expected = {'trend': parts['trend'], 'seasonal': parts['seasonal'], 'residual': parts['noise'] + parts['regime']}
    return {'version': case['version'], 'partition': case['partition'], 'seed': case['seed'],
            'origin': case['origin'], 'length': spec.length, 'horizon': spec.horizon,
            'reconstruction_on_full_synthetic_series': component_errors(expected, classical),
            'forecast_from_train_prefix': seasonal_control(case)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path, nargs='?', default=Path(__file__).with_name('decomposition_smoke.json'))
    args = parser.parse_args(argv)
    spec = SyntheticSpec.parse(read_protocol_json(args.manifest))
    print(json.dumps(verify_decomposition(spec), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
