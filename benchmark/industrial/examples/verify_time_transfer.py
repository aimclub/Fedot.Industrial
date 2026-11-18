"""Compare frozen Chronos-Bolt-Tiny with last value on the pinned TIME windows."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np

from benchmark.industrial.datasets.catalog import read_catalog
from benchmark.industrial.evaluation.forecast_transfer import TrainingSources, evaluate_zero_shot, load_time_windows
from benchmark.industrial.protocols.assets import LocalAsset, checked_asset
from benchmark.industrial.protocols.json_io import read_protocol_json


def median_forecast(pipeline, inputs):
    """Make one point forecast from the observed prefix, without revealing dates or covariates to the model."""
    import torch

    history = inputs.history
    if history.ndim != 2 or history.shape[1] != 1 or not np.isfinite(history).all():
        raise ValueError('Chronos TIME smoke requires one fully observed target history')
    context = torch.tensor(history[:, 0], dtype=torch.float32)
    with torch.inference_mode():
        quantiles, _ = pipeline.predict_quantiles(
            context, prediction_length=len(inputs.future_timestamps), quantile_levels=[.5])
    return quantiles[0].cpu().numpy()


def _load_pipeline(root, model, runtime):
    import torch
    from chronos import BaseChronosPipeline

    versions = {name: version(name) for name in runtime['packages']}
    if versions != runtime['packages']:
        raise ValueError('Installed forecasting packages do not match the TIME run manifest')
    if (runtime['device'], runtime['dtype'], runtime['point_quantile']) != ('cpu', 'float32', .5):
        raise ValueError('The fixed TIME experiment uses CPU, float32 and the median')
    assets = [checked_asset(root, LocalAsset(path, digest)) for path, digest in model['assets'].items()]
    if len(assets) != 2 or {path.name for path in assets} != {'config.json', 'model.safetensors'} or len(
            {path.parent for path in assets}) != 1:
        raise ValueError('Declare exactly one local safetensors model')
    torch.set_num_threads(runtime['torch_threads'])
    started = perf_counter()
    pipeline = BaseChronosPipeline.from_pretrained(
        assets[0].parent, device_map='cpu', torch_dtype=torch.float32, local_files_only=True, use_safetensors=True)
    pipeline.model.eval()
    return pipeline, {**versions, 'torch': str(torch.__version__)}, perf_counter() - started


def compare_time_windows(root, config, catalog_path):
    release = read_catalog(catalog_path)[config['dataset_id']]
    windows = load_time_windows(release, root, target=config['target'], frequency=config['frequency'],
                                test_length=config['test_length'], horizon=config['horizon'])
    if [window.inputs.timestamps[-1] for window in windows] != config['origins']:
        raise ValueError('TIME evaluation must use exactly the declared forecast origins')
    model, runtime = config['model'], config['runtime']
    training = TrainingSources(model['release'], tuple(model['known_training_datasets']),
                               model['training_sources_complete'])
    pipeline, versions, load_seconds = _load_pipeline(root, model, runtime)
    predictions = []
    for window in windows:
        result = evaluate_zero_shot(
            lambda inputs: median_forecast(pipeline, inputs), window, training=training,
            dataset_aliases=(release.release_id,), dates_public=config['dates_public'], frequency=config['frequency'])
        if result['status'] != 'success':
            raise ValueError(f"Incomplete TIME comparison: {result['reason']}")
        predictions.append({**result, 'history_points': len(window.inputs.history),
                            'delta_mae': result['mae'] - result['last_value_mae']})
    forecast_mean = mean(row['mae'] for row in predictions)
    baseline_mean = mean(row['last_value_mae'] for row in predictions)
    return {'dataset_release': release.version, 'forecast_mode': 'frozen_weights',
            'baseline': 'last_value@1', 'device': runtime['device'], 'torch_threads': runtime['torch_threads'],
            'versions': versions, 'model_load_seconds': load_seconds,
            'median_prediction_seconds': median(row['prediction_seconds'] for row in predictions),
            'forecast_mean_mae': forecast_mean, 'last_value_mean_mae': baseline_mean,
            'delta_mean_mae': forecast_mean - baseline_mean,
            'windows_better': sum(row['delta_mae'] < 0 for row in predictions),
            'windows_worse': sum(row['delta_mae'] > 0 for row in predictions),
            'windows': predictions}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_root', type=Path)
    args = parser.parse_args(argv)
    source = Path(__file__)
    config = read_protocol_json(source.with_name('time_transfer_acceptance.json'))
    if config['version'] != 'industrial_time_transfer_acceptance@1':
        raise ValueError('Unrecognized TIME acceptance manifest')
    result = compare_time_windows(args.data_root, config, source.with_name('time_transfer_smoke.json'))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
