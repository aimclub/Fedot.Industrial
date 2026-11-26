"""Seal verified TSER provenance and role-specific datasets before evaluation."""

from dataclasses import asdict
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

from ...protocols.assets import LocalAsset
from ...protocols.json_io import parse_protocol_json, read_protocol_json
from ...protocols.schema import ProtocolError, fields, mapping
from .data import _write_role, canonical_bytes, digest
from .protocol import TSER_VERSION, tsc_protocol
from .regression import exact_series_evidence, load_regression_train, observed_temporal_panel, regression_splits


def tser_profile():
    definition = parse_protocol_json(files(__package__).joinpath('tser.json').read_text(encoding='utf-8'))
    if definition['version'] != TSER_VERSION:
        raise ProtocolError('Unsupported regression pilot profile')
    shared = tsc_protocol()
    definition['search'] = asdict(shared.search)
    definition['split_profile'] = asdict(shared.profile)
    return definition


def _prepare(data_root, dataset, definition, asset):
    if definition['sampling'] == 'observed_temporal':
        return observed_temporal_panel(data_root, dataset, asset)
    if definition['sampling'] == 'exact_series_groups':
        panel = load_regression_train(data_root, dataset, asset)
        return panel, exact_series_evidence(panel, definition['source'])
    raise ProtocolError(f'{dataset}: unsupported sampling design')


def seal_regression(data_root, output, evidence_file):
    profile = tser_profile()
    evidence = fields(read_protocol_json(evidence_file), ('version', 'datasets'), 'evidence')
    if evidence['version'] != TSER_VERSION:
        raise ProtocolError('Expected TSER sampling evidence')
    inputs = mapping(evidence['datasets'], 'datasets')
    expected = {row['name']: row for row in profile['datasets']}
    if inputs.keys() != expected.keys():
        raise ProtocolError('Do not start a partial protocol: declare all four TRAIN paths')
    sklearn = profile['split_profile']['sklearn']
    if version('scikit-learn') != sklearn:
        raise ProtocolError(f'TSER split requires scikit-learn=={sklearn}')
    prepared = []
    for dataset, definition in expected.items():
        raw = fields(inputs[dataset], ('path',), dataset)
        asset = LocalAsset(raw['path'], definition['train_sha256'])
        panel, sampling = _prepare(data_root, dataset, definition, asset)
        split = regression_splits(panel, sampling)
        if split.identity != definition['split_sha256']:
            raise ProtocolError(f'{dataset}: split does not match the frozen profile')
        prepared.append((panel, asset, sampling, split))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    records = []
    for panel, asset, sampling, split in prepared:
        destination = root / panel.dataset
        destination.mkdir()
        development = asdict(split)
        assessment = development.pop('assessment')
        base = dict(train_sha256=asset.sha256, split_sha256=split.identity, classes=(), task='regression',
                    sampling_sha256=digest(canonical_bytes(asdict(sampling))))
        fit = _write_role(destination / 'development', panel.take(split.development),
                          {**base, 'role': 'development', **development})
        score = _write_role(destination / 'assessment', panel.take(assessment),
                            {**base, 'role': 'assessment', 'assessment': assessment})
        records.append(dict(dataset=panel.dataset, development=digest(canonical_bytes(fit)),
                            assessment=digest(canonical_bytes(score))))
    seal = dict(protocol=TSER_VERSION, protocol_sha256=digest(canonical_bytes(profile)), official_test_used=False,
                datasets=records)
    (root / 'seal.json').write_bytes(canonical_bytes(seal))
    return seal
