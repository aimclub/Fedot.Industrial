"""Revise source bytes only; preserve an existing experiment's data and decisions."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from shutil import copyfile
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

from ..freeze import seal, verify_freeze, write_json
from ..preflight import file_digest
from ..runner import freeze_sources


def revise_sources(previous, directory, *, expected_seal, changes):
    """Require the exact reviewed path->digest delta, not an unrestricted refreeze."""
    previous, directory = Path(previous).resolve(), Path(directory).resolve()
    verify_freeze(previous, expected_seal_sha256=expected_seal)
    old = json.loads((previous / 'lock.json').read_text('utf-8'))
    discovered = freeze_sources()
    # Include explicitly sealed protocol files outside the ordinary source catalog.
    current = {**discovered, **{path: file_digest(Path(path)) for path in old['sources']}}
    observed = {path: digest for path, digest in current.items()
                if path not in old['sources'] or old['sources'][path]['sha256'] != digest}
    if not changes or observed != changes:
        raise ValueError('Source revision differs from the exact reviewed changes')
    value = deepcopy(old)
    value['created_at'] = datetime.now(timezone.utc).isoformat()
    value['git_revision_context'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    value['source_revision'] = dict(parent_seal=expected_seal, parent_bundle=str(previous), changes=changes,
                                    data_and_protocol_unchanged=True)
    directory.mkdir(parents=True, exist_ok=False)
    for member in previous.rglob('*'):
        relative = member.relative_to(previous)
        if member.is_file() and relative.as_posix() not in ('seal.json', 'lock.json', 'sources.zip'):
            destination = directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            copyfile(member, destination)
    sources = {}
    with ZipFile(directory / 'sources.zip', 'x', compression=ZIP_DEFLATED) as archive:
        for index, (path, digest) in enumerate(sorted(current.items())):
            member = f'sources/{index:04d}{Path(path).suffix}'
            if file_digest(Path(path)) != digest:
                raise ValueError('Source changed during revision')
            archive.write(path, arcname=member)
            sources[path] = dict(member=member, sha256=digest)
    value['sources'] = sources
    write_json(directory / 'lock.json', value)
    seal(directory)
    return verify_freeze(directory, check_current=True)
