import json

import pytest
from hypothesis import given, strategies as st

from benchmark.industrial.evaluation.dask_profiles import read_dask_profile


def test_profile_columns_preserve_sampling_mass_without_double_counting(tmp_path):
    columns = {'name': ['root', 'fit', 'predict'], 'filename': ['', 'model.py', 'model.py'],
               'line_number': [0, 1, 2], 'width': [1., .6, .3], 'bottom': [0, 1, 1], 'time': ['1s'] * 3}
    document = {'roots': [{'name': 'ColumnDataSource', 'attributes': {
        'data': {'type': 'map', 'entries': list(columns.items())}}}]}
    path = tmp_path / 'profile.html'
    path.write_text('<script>throw new Error("must not execute")</script>'
                    '<script type="application/json">' + json.dumps(document) + '</script>', encoding='utf-8')
    frame = read_dask_profile(path)
    assert frame.exclusive_sample_share.tolist() == pytest.approx([.1, .6, .3])
    assert frame.exclusive_sample_share.sum() == pytest.approx(1)


def test_unsupported_profile_is_explicit(tmp_path):
    path = tmp_path / 'empty.html'
    path.write_text('<html>No profile</html>', encoding='utf-8')
    with pytest.raises(ValueError, match='No supported'):
        read_dask_profile(path)


@given(st.lists(st.tuples(st.integers(1, 20), st.lists(st.integers(1, 10), max_size=4)),
                min_size=1, max_size=5), st.integers(0, 10))
def test_nested_sample_partition_conserves_total_mass(children, extra):
    import tempfile
    from pathlib import Path
    widths, depths = [1.], [0]
    total = sum(weight for weight, _ in children) + extra
    for weight, grandchildren in children:
        parent = weight / total
        widths.append(parent)
        depths.append(1)
        for leaf in grandchildren:
            widths.append(parent * leaf / (sum(grandchildren) + 1))
            depths.append(2)
    count = len(widths)
    columns = dict(name=['frame'] * count, filename=['module.py'] * count,
                   line_number=[1] * count, width=widths, bottom=depths, time=['1s'] * count)
    document = {'name': 'ColumnDataSource', 'attributes': {
        'data': {'type': 'map', 'entries': list(columns.items())}}}
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / 'profile.html'
        path.write_text('<script type="application/json">' + json.dumps(document) + '</script>', encoding='utf-8')
        frame = read_dask_profile(path)
    assert frame.exclusive_sample_share.sum() == pytest.approx(1)
    assert frame.exclusive_sample_share.min() >= 0
