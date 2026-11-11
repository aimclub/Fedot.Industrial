"""Read sampled stack columns embedded in Dask/Bokeh profile HTML, without JS."""
from html.parser import HTMLParser
import json
from pathlib import Path

import pandas as pd


class _JsonScripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.parts = []
        self.documents = []

    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.active = dict(attrs).get('type') == 'application/json'
            self.parts = []

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == 'script' and self.active:
            self.documents.append(json.loads(''.join(self.parts)))
            self.active = False


def _objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _objects(child)


def read_dask_profile(path):
    """Bokeh 3.4 profile columns; shares are sampling estimates, not exact CPU."""
    parser = _JsonScripts()
    parser.feed(Path(path).read_text('utf-8'))
    for obj in _objects(parser.documents):
        if obj.get('name') != 'ColumnDataSource':
            continue
        data = obj.get('attributes', {}).get('data', {})
        columns = dict(data.get('entries', ())) if data.get('type') == 'map' else data
        if not {'name', 'filename', 'width', 'bottom'} <= columns.keys():
            continue
        frame = pd.DataFrame({name: columns[name] for name in
                              ('name', 'filename', 'line_number', 'width', 'bottom', 'time')})
        parents, children = [], [0.] * len(frame)
        widths, depths = frame.width.tolist(), frame.bottom.tolist()
        for index, (width, depth) in enumerate(zip(widths, depths)):
            while parents and depths[parents[-1]] >= depth:
                parents.pop()
            if parents:
                children[parents[-1]] += width
            parents.append(index)
        frame['exclusive_sample_share'] = frame.width - pd.Series(children)
        if (frame.exclusive_sample_share < -1e-9).any():
            raise ValueError('Invalid nested sampled profile widths')
        frame['exclusive_sample_share'] = frame.exclusive_sample_share.clip(lower=0)
        return frame.rename(columns={'width': 'inclusive_sample_share', 'bottom': 'stack_depth'})
    raise ValueError(f'No supported sampled profile columns in {path}')
