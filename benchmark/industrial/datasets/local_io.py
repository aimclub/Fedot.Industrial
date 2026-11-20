from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, DecimalException
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
        parser_metadata['uniform_channel_layout'] = (
            parser_metadata['uniform_channel_layout'] and test_metadata['uniform_channel_layout']
            and parser_metadata['dimensions'] == test_metadata['dimensions']
            and train_features.shape[1] == test_features.shape[1]
        )
        parser_metadata['aligned_channel_grid'] = (
            parser_metadata['aligned_channel_grid'] and test_metadata['aligned_channel_grid']
            and parser_metadata['timestamps'] == test_metadata['timestamps']
            and parser_metadata.get('timestamp_cadence') == test_metadata.get('timestamp_cadence')
        )
        parser_metadata.update(
            {
                key: value
                for key, value in test_metadata.items()
                if key not in parser_metadata
            }
        )
        if not parser_metadata['aligned_channel_grid']:
            parser_metadata.pop('timestamp_cadence', None)
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


def _load_tabular_split(path: Path, *, separator: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, sep=separator, header=None)
    if frame.shape[1] < 2:
        raise LocalDatasetParseError(f'Tabular split file must contain target and at least one feature: {path}')
    target = frame.iloc[:, 0].to_numpy()
    features = frame.iloc[:, 1:].to_numpy(dtype=float)
    return features, target


def _load_ts_split(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    last_error: UnicodeDecodeError | None = None
    for encoding in ('utf-8', 'utf-8-sig', 'cp1252', 'latin-1'):
        try:
            return _load_ts_split_with_encoding(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    if last_error is not None:
        raise LocalDatasetParseError(f'Could not decode .ts file with supported encodings: {path}') from last_error
    raise LocalDatasetParseError(f'Could not parse .ts file: {path}')


def _load_ts_split_with_encoding(path: Path, *, encoding: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    timestamps = False
    target_label = False
    class_label = False
    dimensions = None
    metadata: dict[str, Any] = {}
    data_started = False
    case_vectors: list[np.ndarray] = []
    channel_layouts: set[tuple[int, ...]] = set()
    cadences: set[tuple[str, str | None] | None] = set()
    targets: list[Any] = []

    with path.open('r', encoding=encoding) as source:
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

            if not target_label and not class_label:
                raise LocalDatasetParseError(f'.ts file must define @targetlabel or @classlabel: {path}')

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

            flattened: list[float] = []
            channel_lengths = []
            timestamp_axes = []
            for dimension_part in dimension_parts:
                cleaned = dimension_part.strip()
                previous = len(flattened)
                if timestamps:
                    axis, values = _parse_timestamped_dimension(cleaned, path=path) if cleaned else ((), ())
                    timestamp_axes.append(axis)
                    flattened.extend(values)
                else:
                    flattened.extend(float(token) for token in cleaned.split(',') if token)
                channel_lengths.append(len(flattened) - previous)

            channel_layouts.add(tuple(channel_lengths))
            if timestamps:
                cadences.add(_shared_cadence(timestamp_axes))
            case_vectors.append(np.asarray(flattened, dtype=float))
            if target_label:
                targets.append(float(target_part))
            else:
                targets.append(target_part)

    if not case_vectors:
        raise LocalDatasetParseError(f'No cases were parsed from .ts file: {path}')

    lengths = {vector.size for vector in case_vectors}
    if len(lengths) != 1:
        raise LocalDatasetParseError(
            f'Variable-length .ts records are not supported by the flat Industrial benchmark adapters yet: {path}'
        )

    layout = next(iter(channel_layouts)) if len(channel_layouts) == 1 else ()
    metadata['dimensions'] = dimensions or len(layout) or 1
    metadata['uniform_channel_layout'] = bool(layout and layout[0] > 0 and len(set(layout)) == 1)
    metadata['aligned_channel_grid'] = not timestamps or (len(cadences) == 1 and None not in cadences)
    if timestamps and metadata['aligned_channel_grid']:
        metadata['timestamp_cadence'] = next(iter(cadences))
    metadata['timestamps'] = bool(timestamps)
    metadata['target_type'] = 'regression' if target_label else 'classification'
    features = np.vstack(case_vectors)
    target_array = np.asarray(targets, dtype=float if target_label else object)
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


def _parse_timestamped_dimension(dimension_part: str, *, path: Path) -> tuple[tuple[str, ...], tuple[float, ...]]:
    entries = re.findall(r'\(([^,]+),([^)]+)\)', dimension_part)
    if not entries:
        raise LocalDatasetParseError(f'Could not parse timestamped values from .ts file: {path}')
    return tuple(stamp.strip() for stamp, _ in entries), tuple(float(value) for _, value in entries)


def _ordered_axis(stamps):
    try:
        coordinates = tuple(Decimal(stamp) for stamp in stamps)
    except DecimalException:
        if not all(re.match(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}(?:$|[ T])", stamp) for stamp in stamps):
            return None
        try:
            dates = tuple(pd.Timestamp(stamp) for stamp in stamps)
            kinds = {"datetime_naive" if date.tz is None else "datetime_utc" for date in dates}
            if len(kinds) != 1 or any(pd.isna(date) for date in dates):
                return None
            return kinds.pop(), tuple(date.value for date in dates)
        except (ValueError, OverflowError):
            return None
    if not all(value.is_finite() for value in coordinates):
        return None
    return "numeric", coordinates


def _shared_cadence(axes):
    """Panel methods need coincident channels; different records may have different origins."""
    if not axes or not axes[0]:
        return None
    expected = _ordered_axis(axes[0])
    if expected is None or any(axis != axes[0] and _ordered_axis(axis) != expected for axis in axes[1:]):
        return None
    kind, coordinates = expected
    if len(coordinates) == 1:
        return kind, None
    try:
        step = coordinates[1] - coordinates[0]
        if step <= 0 or any(right - left != step for left, right in zip(coordinates, coordinates[1:])):
            return None
        return kind, str(step.normalize() if kind == "numeric" else step)
    except DecimalException:
        return None


def _parse_boolean_tag(line: str) -> bool:
    parts = line.split()
    if len(parts) < 2:
        raise LocalDatasetParseError(f'Malformed boolean metadata tag: {line}')
    return parts[1].strip().lower() == 'true'
