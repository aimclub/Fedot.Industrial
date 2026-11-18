"""Read-only external releases. The catalog never downloads or opens dataset contents."""

from dataclasses import dataclass

from ..protocols.assets import LocalAsset, checked_asset
from ..protocols.json_io import read_protocol_json
from ..protocols.schema import Access, ProtocolError, choice, fields, mapping, string


CATALOG_VERSION = 'industrial_dataset_catalog@1'


@dataclass(frozen=True)
class CatalogRelease:
    release_id: str
    version: str
    source: str
    access: Access
    license: str | None
    assets: tuple[tuple[str, LocalAsset], ...]

    def files(self, root, roles):
        if self.license is None:
            raise ProtocolError(f'{self.release_id}: dataset license is unknown')
        assets = dict(self.assets)
        missing = set(roles) - assets.keys()
        if missing:
            raise ProtocolError(f'{self.release_id}: missing file roles {sorted(missing)}')
        return {role: checked_asset(root, assets[role]) for role in roles}


def parse_catalog(payload):
    catalog = fields(payload, ('version', 'releases'), 'catalog')
    if catalog['version'] != CATALOG_VERSION:
        raise ProtocolError('Unsupported external dataset catalog version')
    releases = mapping(catalog['releases'], 'catalog.releases')
    if not releases:
        raise ProtocolError('Dataset catalog must declare a release')
    entries = {}
    for release_id, row in releases.items():
        name = f'releases.{string(release_id, "release ID")}'
        row = fields(row, ('version', 'source', 'access', 'license', 'assets'), name)
        entries[release_id] = CatalogRelease(
            release_id, string(row['version'], f'{name}.version'), string(row['source'], f'{name}.source'),
            choice(row['access'], Access, f'{name}.access'),
            None if row['license'] is None else string(row['license'], f'{name}.license'),
            _assets(row['assets'], name))
    return entries


def _assets(value, name):
    rows = mapping(value, f'{name}.assets')
    if not rows:
        raise ProtocolError(f'{name} has no declared files')
    result = []
    for role, row in sorted(rows.items()):
        path = f'{name}.assets.{string(role, "file role")}'
        row = fields(row, ('path', 'sha256'), path)
        result.append((role, LocalAsset(row['path'], row['sha256'])))
    if len({asset.path for _, asset in result}) != len(result):
        raise ProtocolError(f'{name} reuses a path for different file roles')
    return tuple(result)


def read_catalog(path):
    """Read a local release description without resolving its assets."""
    return parse_catalog(read_protocol_json(path))
