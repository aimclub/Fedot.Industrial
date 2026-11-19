"""Access to declared local bytes without a downloader or cache mutation."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PureWindowsPath
import re


class AssetUnavailable(OSError):
    pass


class AssetCorrupt(ValueError):
    pass


class AssetIncompatible(ValueError):
    pass


@dataclass(frozen=True)
class LocalAsset:
    path: str
    sha256: str

    def __post_init__(self):
        path = self.path
        if (not isinstance(path, str) or not path or path != path.strip()
                or any(not part or part != part.rstrip(' .') or PureWindowsPath(part).is_reserved()
                       for part in path.split('/'))
                or re.search(r'[\\:*?"<>|\x00-\x1f]', path)):
            raise AssetIncompatible('Artifact must be a portable relative file path')
        digest = self.sha256
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-fA-F]{64}', digest) is None:
            raise AssetIncompatible('Artifact SHA-256 must contain 64 hexadecimal characters')
        object.__setattr__(self, 'sha256', digest.lower())


def file_digest(path):
    digest = sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def local_root(root):
    root = Path(root)
    if root.anchor.startswith(('\\\\', '//')):
        raise AssetIncompatible('Data root must be local, not a UNC path')
    try:
        canonical = root.resolve()
    except RuntimeError as error:
        raise AssetIncompatible('Data root path cannot be resolved') from error
    except OSError as error:
        raise AssetUnavailable('Data root cannot be read') from error
    if canonical.anchor.startswith(('\\\\', '//')):
        raise AssetIncompatible('Data root resolves to a non-local path')
    return canonical


def _asset_path(root, asset: LocalAsset):
    root = local_root(root)
    try:
        path = (root / asset.path).resolve()
        if not path.is_relative_to(root):
            raise AssetIncompatible('Artifact escapes the local data root')
        if not path.is_file():
            raise AssetUnavailable('Local artifact is absent or is not a regular file')
    except AssetUnavailable:
        raise
    except OSError as error:
        raise AssetUnavailable(f'Local artifact cannot be read: {type(error).__name__}') from error
    except RuntimeError as error:
        raise AssetIncompatible('Local artifact path cannot be resolved') from error
    return path


def _verify_digest(digest, asset):
    if digest != asset.sha256:
        raise AssetCorrupt('Local artifact SHA-256 does not match the declared release')


def checked_asset(root, asset: LocalAsset):
    """Return a verified regular file strictly within the declared root."""
    path = _asset_path(root, asset)
    try:
        _verify_digest(file_digest(path), asset)
    except OSError as error:
        raise AssetUnavailable(f'Local artifact cannot be read: {type(error).__name__}') from error
    return path


def read_checked_asset(root, asset: LocalAsset):
    """Return the same bytes that passed verification, for snapshot-based parsers."""
    path = _asset_path(root, asset)
    try:
        content = path.read_bytes()
    except OSError as error:
        raise AssetUnavailable(f'Local artifact cannot be read: {type(error).__name__}') from error
    _verify_digest(sha256(content).hexdigest(), asset)
    return path, content
