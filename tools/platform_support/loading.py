"""File effects and transport decoding, kept outside policy interpretation."""

import json
from pathlib import Path
from typing import Callable

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 developer environments use tomli.
    import tomli as tomllib

from .schema import Compatibility, Issue, ParseResult, parse_compatibility
from .verification import Project, parse_project


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _decode(path: Path, decoder: Callable) -> ParseResult[object]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return ParseResult(None, (Issue(str(path), "read", str(error)),))
    try:
        return ParseResult(decoder(text), ())
    except (ValueError, tomllib.TOMLDecodeError) as error:
        return ParseResult(None, (Issue(str(path), "syntax", str(error)),))


def load_project(root: Path) -> ParseResult[Project]:
    decoded = _decode(root / "pyproject.toml", tomllib.loads)
    return parse_project(decoded.value) if decoded.ok else ParseResult(None, decoded.issues)


def load_compatibility() -> ParseResult[Compatibility]:
    decoded = _decode(Path(__file__).with_name("compatibility.json"), _decode_json)
    return parse_compatibility(decoded.value) if decoded.ok else ParseResult(None, decoded.issues)


def _decode_json(text: str) -> object:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=unique_object)


def load_requirements(root: Path, filename: str = "requirements.txt") -> ParseResult[str]:
    try:
        return ParseResult((root / filename).read_text(encoding="utf-8"), ())
    except (OSError, UnicodeError) as error:
        return ParseResult(None, (Issue(filename, "read", str(error)),))
