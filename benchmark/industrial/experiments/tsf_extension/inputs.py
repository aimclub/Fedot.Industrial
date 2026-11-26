"""Reproduce the extension selection and freeze all 24 searches' TRAIN inputs."""

from dataclasses import asdict
from hashlib import sha256
from importlib.resources import files
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from ...protocols.assets import LocalAsset, read_checked_asset
from ...protocols.json_io import parse_protocol_json, read_protocol_json
from ...protocols.schema import ProtocolError
from .planning import VERSION, TrainRelease, identity, load_profile, load_train_csv, origins, select_series, setup_search


def _json_bytes(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(payload):
    return sha256(payload).hexdigest()


def _previous_inputs(root):
    lock = read_protocol_json(Path(root) / 'lock.json')
    _, content = read_checked_asset(root, LocalAsset('manifest.json', lock['manifest_sha256']))
    manifest = parse_protocol_json(content.decode())
    if manifest['schema'] != 'industrial_tsf_variant_b@1' or manifest['official_test_used'] is not False:
        raise ProtocolError('Require a verified preceding TRAIN-only B series')
    ids, hashes = set(), set()
    for case, row in manifest['cases'].items():
        _, content = read_checked_asset(root, LocalAsset(f'{case}/assessment.npy', row['assessment_input']['sha256']))
        previous = np.load(BytesIO(content), allow_pickle=False)
        hashes.add(identity(previous))
        ids.add((row['group'], case.removeprefix(f"{row['group']}_")))
    return ids, hashes, lock['manifest_sha256']


def _write_array(path, values):
    asset = BytesIO()
    np.save(asset, values, allow_pickle=False)
    data = asset.getvalue()
    path.write_bytes(data)
    return _digest(data)


def seal(data_root, previous_bundle, output):
    collections, seeds = load_profile()
    limit = parse_protocol_json(files(__package__).joinpath('extension.json').read_text('utf-8'))['history_limit']
    previous_ids, previous_hashes, previous_identity = _previous_inputs(previous_bundle)
    planned = []
    for collection in collections:
        release = TrainRelease(LocalAsset(collection.filename, collection.sha256),
                               collection.source, collection.horizon)
        series = load_train_csv(data_root, collection, release)
        trimmed = {name: values[-limit:] for name, values in series.items()}
        exclusions = {name for group, name in previous_ids if group == collection.id.replace(' ', '')}
        selection, reasons = select_series(collection, trimmed, exclusions, previous_hashes)
        planned.append((collection, series, trimmed, selection, reasons))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, rejected = [], {}
    for collection, full, trimmed, selection, reasons in planned:
        directory = output / collection.id.replace(' ', '_')
        directory.mkdir()
        (directory / 'development').mkdir()
        (directory / 'assessment').mkdir()
        rejected[collection.id] = reasons
        for identifier in selection:
            values = trimmed[identifier]
            prefix, search = setup_search(values, collection)
            _, assess = origins(len(values), collection)
            offset = len(full[identifier]) - len(values)
            local = f'{identifier}.npy'
            if not identifier.isalnum():
                raise ProtocolError('Series identifier cannot be used as a local artifact name')
            rows.append(dict(
                id=identifier, collection=asdict(collection), offset=offset,
                search=search, assessment=assess,
                source_search=[offset + begin for begin in search],
                source_assessment=[offset + begin for begin in assess], seeds=seeds,
                development_sha256=_write_array(directory / 'development' / local, prefix),
                assessment_sha256=_write_array(directory / 'assessment' / local, values),
            ))
    manifest = dict(protocol=VERSION, previous_manifest_sha256=previous_identity, official_test_used=False,
                    history_limit=limit, searches=len(rows) * len(seeds), rows=rows, exclusions=rejected)
    payload = _json_bytes(manifest)
    (output / 'manifest.json').write_bytes(payload)
    (output / 'lock.json').write_bytes(_json_bytes({'manifest_sha256': _digest(payload)}))
    return manifest
