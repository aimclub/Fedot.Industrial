"""Explicit resource loading, kept out of the pure policy decisions."""
import json
from pathlib import Path

from .contracts import parse_staged_search


def load_staged_search_defaults():
    return parse_staged_search(json.loads(Path(__file__).with_name('defaults.json').read_text('utf-8')))
