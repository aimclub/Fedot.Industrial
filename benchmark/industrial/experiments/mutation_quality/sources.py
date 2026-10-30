"""Explicit source delivery and narrow Monash Electricity TRAIN extraction."""

import csv
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

from .preflight import file_digest


def source_catalog():
    value = json.loads(Path(__file__).with_name('external_sources.json').read_text('utf-8'))
    if value['schema_version'] != 'industrial_mutation_quality_sources@1':
        raise ValueError('Unsupported source catalog')
    return value


def deliver(spec, directory):
    path = directory / spec['file']
    if not path.exists():
        # No implicit extraction or execution of remote code; retain partial evidence on failure.
        with urlopen(spec['url'], timeout=90) as response, path.open('xb') as output:
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > spec['maximum_bytes']:
                    raise ValueError('Source exceeds its declared size bound')
                output.write(block)
    if file_digest(path) != spec['sha256']:
        raise ValueError(f'Source checksum mismatch; do not overwrite evidence: {path}')
    return path


def train_row(line, reserved_tail, expected_length):
    prefix, separator, values = line.strip().rpartition(':')
    if not separator or type(reserved_tail) is not int or not 0 < reserved_tail < expected_length:
        raise ValueError('Invalid Electricity record or boundary')
    # Future tokens are discarded as text, never parsed as numbers or used for eligibility.
    parts = values.rsplit(',', reserved_tail)
    if len(parts) != reserved_tail + 1:
        raise ValueError('Insufficient reserved tail')
    training = parts[0].split(',')
    if len(training) != expected_length - reserved_tail:
        raise ValueError('Unexpected Electricity series length')
    array = np.asarray(training, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite TRAIN values; no implicit imputation')
    return prefix.split(':', 1)[0], training


def extract_electricity_train(archive, directory, spec):
    destination = directory / spec['train_file']
    provenance = destination.with_suffix('.provenance.json')
    if destination.exists() or provenance.exists():
        raise FileExistsError('Derived TRAIN already exists; verify or choose a new output directory')
    seen = set()
    with ZipFile(archive) as zipped:
        if zipped.namelist() != [spec['member']] or zipped.getinfo(spec['member']).file_size > 150_000_000:
            raise ValueError('Unexpected archive members or expanded size')
        with io.TextIOWrapper(zipped.open(spec['member']), encoding='utf-8', newline=None) as stream:
            attributes, frequency, started = [], None, False
            with destination.open('x', encoding='utf-8', newline='') as output:
                writer = csv.writer(output)
                writer.writerow(['series_id', *range(spec['expected_length'] - spec['reserved_tail'])])
                for raw in stream:
                    line = raw.strip()
                    if not line or line.startswith('#'):
                        continue
                    if not started:
                        if line.lower().startswith('@attribute '):
                            attributes.append(line.split()[1:])
                        if line.lower().startswith('@frequency '):
                            frequency = line.split()[1]
                        if line.lower() == '@data':
                            if attributes != [['series_name', 'string'], ['start_timestamp', 'date']] or frequency != 'hourly':
                                raise ValueError('Unsupported TSF schema; this converter is Electricity-specific')
                            started = True
                        continue
                    name, training = train_row(line, spec['reserved_tail'], spec['expected_length'])
                    if name in seen:
                        raise ValueError('Duplicate Electricity series identifier')
                    seen.add(name)
                    writer.writerow([name, *training])
    if len(seen) != spec['expected_series']:
        raise ValueError('Unexpected Electricity series count')
    record = dict(source_page=spec['page'], archive_sha256=file_digest(archive),
                  train_sha256=file_digest(destination), reserved_tail=spec['reserved_tail'],
                  series=len(seen), original_length=spec['expected_length'],
                  train_length=spec['expected_length'] - spec['reserved_tail'],
                  boundary_reference=spec['boundary_reference'], future_tokens_parsed=False,
                  official_test_used=False, archive_role='transport_only_unpartitioned_source')
    provenance.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


def prepare_sources(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    catalog = source_catalog()
    m4 = deliver(catalog['m4_hourly'], directory)
    archive = deliver(catalog['electricity'], directory)
    spec = catalog['electricity']
    if (directory / spec['train_file']).exists():
        record = verify_train_provenance(directory / spec['train_file'], spec)
    else:
        record = extract_electricity_train(archive, directory, spec)
    return dict(m4_hourly_sha256=file_digest(m4), electricity=record)


def verify_train_provenance(path, spec):
    record = json.loads(path.with_suffix('.provenance.json').read_text('utf-8'))
    if (record['train_sha256'] != file_digest(path) or record['archive_sha256'] != spec['sha256']
            or record['reserved_tail'] != spec['reserved_tail'] or record['series'] != spec['expected_series']
            or record['original_length'] != spec['expected_length']
            or record['official_test_used'] is not False or record['future_tokens_parsed'] is not False):
        raise ValueError('Monash TRAIN provenance mismatch')
    return record
