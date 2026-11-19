from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_DATA_DIR = PROJECT_ROOT / 'examples' / 'utils' / 'data'
DEFAULT_LEGACY_DATA_DIR = PROJECT_ROOT / 'fedot_ind' / 'data'
LOCAL_TASK_DATA_DIRS = (
    'ts_classification',
    'ts_regression',
    'forecasting',
    'anomaly_detection',
)


class LocalDatasetParseError(ValueError):
    pass


@dataclass(frozen=True)
class LocalSplitData:
    train_features: np.ndarray
    train_target: np.ndarray
    test_features: np.ndarray
    test_target: np.ndarray
    metadata: dict[str, Any]


def resolve_local_split_paths(
        dataset_name: str,
        *,
        data_root: str | Path | None = None,
        train_path: str | Path | None = None,
        test_path: str | Path | None = None,
) -> tuple[Path, Path]:
    if train_path is not None and test_path is not None:
        return Path(train_path), Path(test_path)

    roots = [Path(data_root)] if data_root is not None else [DEFAULT_LOCAL_DATA_DIR, DEFAULT_LEGACY_DATA_DIR]
    candidate_dirs: list[Path] = []
    seen_dirs: set[Path] = set()
    for root in roots:
        for candidate in (root / dataset_name, *(root / task_dir / dataset_name for task_dir in LOCAL_TASK_DATA_DIRS)):
            if candidate in seen_dirs:
                continue
            seen_dirs.add(candidate)
            candidate_dirs.append(candidate)

    searched_dirs = []
    for dataset_dir in candidate_dirs:
        searched_dirs.append(str(dataset_dir))
        if not dataset_dir.exists():
            continue
        train_base = dataset_dir / f'{dataset_name}_TRAIN'
        test_base = dataset_dir / f'{dataset_name}_TEST'
        for extension in ('.tsv', '.csv', '.ts'):
            train_candidate = train_base.with_suffix(extension)
            test_candidate = test_base.with_suffix(extension)
            if train_candidate.exists() and test_candidate.exists():
                return train_candidate, test_candidate

    searched = ', '.join(searched_dirs)
    raise LocalDatasetParseError(f'Could not resolve local TRAIN/TEST files for {dataset_name}. Searched: {searched}')


def load_local_supervised_split(
        dataset_name: str,
        *,
        data_root: str | Path | None = None,
        train_path: str | Path | None = None,
        test_path: str | Path | None = None,
) -> LocalSplitData:
    train_file, test_file = resolve_local_split_paths(
        dataset_name,
        data_root=data_root,
        train_path=train_path,
        test_path=test_path,
    )

    if train_file.suffix != test_file.suffix:
        raise LocalDatasetParseError('TRAIN/TEST files use different suffixes.')

    suffix = train_file.suffix.lower()
    if suffix == '.tsv':
        train_features, train_target = _load_tabular_split(train_file, separator='\t')
        test_features, test_target = _load_tabular_split(test_file, separator='\t')
        split_provenance = 'local_tsv'
    elif suffix == '.csv':
        train_features, train_target = _load_tabular_split(train_file, separator=',')
        test_features, test_target = _load_tabular_split(test_file, separator=',')
        split_provenance = 'local_csv'
    elif suffix == '.ts':
        train_features, train_target, parser_metadata = _load_ts_split(train_file)
        test_features, test_target, test_metadata = _load_ts_split(test_file)
        parser_metadata.update(
            {
                key: value
                for key, value in test_metadata.items()
                if key not in parser_metadata
            }
        )
        split_provenance = 'local_ts'
    else:  # pragma: no cover
        raise LocalDatasetParseError(f'Unsupported local split suffix: {suffix}')

    metadata = {
        'source_train_file': train_file.name,
        'source_test_file': test_file.name,
        'source_format': suffix.lstrip('.'),
        'split_provenance': split_provenance,
    }
    if suffix == '.ts':
        metadata.update(parser_metadata)

    return LocalSplitData(
        train_features=np.asarray(train_features, dtype=float),
        train_target=np.asarray(train_target),
        test_features=np.asarray(test_features, dtype=float),
        test_target=np.asarray(test_target),
        metadata=metadata,
    )


def _load_tabular_split(path: Path, *, separator: str, content: bytes | None = None) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path if content is None else BytesIO(content), sep=separator, header=None)
    if frame.shape[1] < 2:
        raise LocalDatasetParseError(f'Tabular split file must contain target and at least one feature: {path}')
    target = frame.iloc[:, 0].to_numpy()
    features = frame.iloc[:, 1:].to_numpy(dtype=float)
    return features, target


def load_local_ts_cases(path: Path, *, content: bytes | None = None):
    """Read channel-preserving cases without requiring equal record lengths."""
    return _load_ts_split(path, preserve_channels=True, content=content)


def _load_ts_split(path: Path, *, preserve_channels=False, content: bytes | None = None):
    last_error: UnicodeDecodeError | None = None
    for encoding in ('utf-8', 'utf-8-sig', 'cp1252', 'latin-1'):
        try:
            return _load_ts_split_with_encoding(
                path, encoding=encoding, preserve_channels=preserve_channels, content=content)
        except UnicodeDecodeError as error:
            last_error = error
    if last_error is not None:
        raise LocalDatasetParseError(f'Could not decode .ts file with supported encodings: {path}') from last_error
    raise LocalDatasetParseError(f'Could not parse .ts file: {path}')


def _load_ts_split_with_encoding(path: Path, *, encoding: str, preserve_channels=False, content: bytes | None = None):
    timestamps = False
    target_label = False
    class_label = False
    dimensions = None
    metadata: dict[str, Any] = {}
    data_started = False
    cases: list[tuple[np.ndarray, ...]] = []
    targets: list[Any] = []

    source_stream = path.open('r', encoding=encoding) if content is None else StringIO(content.decode(encoding))
    with source_stream as source:
        for raw_line in source:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            lower = line.lower()
            if not data_started:
                if lower.startswith('@timestamps'):
                    timestamps = _parse_boolean_tag(line)
                elif lower.startswith('@targetlabel'):
                    target_label = _parse_boolean_tag(line)
                elif lower.startswith('@classlabel'):
                    class_label = _parse_boolean_tag(line)
                elif lower.startswith('@dimensions') or lower.startswith('@dimension'):
                    parts = line.split()
                    if len(parts) >= 2:
                        dimensions = int(parts[1])
                elif lower.startswith('@serieslength'):
                    parts = line.split()
                    if len(parts) >= 2:
                        metadata['series_length'] = int(parts[1])
                elif lower.startswith('@problemname'):
                    parts = line.split(maxsplit=1)
                    metadata['problem_name'] = parts[1] if len(parts) > 1 else path.stem
                elif lower.startswith('@univariate'):
                    metadata['univariate'] = _parse_boolean_tag(line)
                elif lower.startswith('@equallength'):
                    metadata['equal_length'] = _parse_boolean_tag(line)
                elif lower.startswith('@data'):
                    data_started = True
                continue

            if target_label == class_label:
                raise LocalDatasetParseError(f'.ts file must define exactly one target kind: {path}')

            split_index = line.rfind(':')
            if split_index < 0:
                raise LocalDatasetParseError(f'Could not split series and target in line from {path}')

            series_part = line[:split_index]
            target_part = line[split_index + 1:].strip()
            dimension_parts = _split_ts_dimension_parts(series_part, timestamps=timestamps)
            if dimensions is not None and len(dimension_parts) != dimensions:
                raise LocalDatasetParseError(
                    f'Unexpected number of dimensions in {path}: expected {dimensions}, got {len(dimension_parts)}'
                )

            channels = []
            for dimension_part in dimension_parts:
                cleaned = dimension_part.strip()
                if not cleaned:
                    values = []
                elif timestamps:
                    values = _parse_timestamped_dimension_values(cleaned, path=path)
                else:
                    values = [_parse_ts_value(token) for token in cleaned.split(',')]
                channels.append(np.asarray(values, dtype=float))

            cases.append(tuple(channels))
            if target_label:
                targets.append(float(target_part))
            else:
                targets.append(target_part)

    if not cases:
        raise LocalDatasetParseError(f'No cases were parsed from .ts file: {path}')

    metadata['dimensions'] = dimensions or len(cases[0])
    metadata['timestamps'] = bool(timestamps)
    metadata['target_type'] = 'regression' if target_label else 'classification'
    target_array = np.asarray(targets, dtype=float if target_label else object)
    if preserve_channels:
        return tuple(cases), target_array, metadata

    case_vectors = [np.concatenate(case) for case in cases]
    lengths = {vector.size for vector in case_vectors}
    if len(lengths) != 1:
        raise LocalDatasetParseError(
            f'Variable-length .ts records are not supported by the flat Industrial benchmark adapters yet: {path}'
        )

    features = np.vstack(case_vectors)
    return features, target_array, metadata


def _split_ts_dimension_parts(series_part: str, *, timestamps: bool) -> list[str]:
    if not timestamps:
        return series_part.split(':')

    parts: list[str] = []
    depth = 0
    start_index = 0
    for index, character in enumerate(series_part):
        if character == '(':
            depth += 1
        elif character == ')':
            depth = max(0, depth - 1)
        elif character == ':' and depth == 0:
            parts.append(series_part[start_index:index])
            start_index = index + 1
    parts.append(series_part[start_index:])
    return parts


def _parse_timestamped_dimension_values(dimension_part: str, *, path: Path) -> list[float]:
    matches = list(re.finditer(r'\(([^,]+),([^)]+)\)', dimension_part))
    if not matches:
        raise LocalDatasetParseError(f'Could not parse timestamped values from .ts file: {path}')
    return [_parse_ts_value(match.group(2)) for match in matches]


def _parse_ts_value(value: str) -> float:
    return np.nan if value.strip() == '?' else float(value)


def _parse_boolean_tag(line: str) -> bool:
    parts = line.split()
    if len(parts) < 2:
        raise LocalDatasetParseError(f'Malformed boolean metadata tag: {line}')
    return parts[1].strip().lower() == 'true'
