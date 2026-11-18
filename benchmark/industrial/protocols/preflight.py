"""Read-only checks of explicitly supplied local dataset artifacts."""

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path

from .planning import ResolvedProtocolPlan
from .schema import ProtocolError


class PreflightStatus(str, Enum):
    READY = 'ready'
    UNAVAILABLE = 'unavailable'
    CORRUPT = 'corrupt'
    INCOMPATIBLE = 'incompatible'
    UNKNOWN_LICENSE = 'unknown_license'


@dataclass(frozen=True)
class DatasetCheck:
    dataset_id: str
    status: PreflightStatus
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    checks: tuple[DatasetCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.status is PreflightStatus.READY for check in self.checks)

    def to_dict(self) -> dict:
        return {'ready': self.ready, 'checks': [asdict(check) for check in self.checks]}


def _check_artifact(dataset_id, release, root):
    def result(status, detail):
        return DatasetCheck(dataset_id, status, detail)

    if root is None:
        return result(PreflightStatus.UNAVAILABLE, 'No local data root supplied')
    try:
        artifact = (root / release.artifact).resolve()
        if not artifact.is_relative_to(root):
            return result(PreflightStatus.INCOMPATIBLE, 'Artifact escapes the local data root')
        if not artifact.is_file():
            return result(PreflightStatus.UNAVAILABLE, 'Local artifact is absent or is not a regular file')
        digest = sha256()
        with artifact.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
    except OSError as error:
        return result(PreflightStatus.UNAVAILABLE, f'Local artifact cannot be read: {type(error).__name__}')
    except RuntimeError:
        return result(PreflightStatus.INCOMPATIBLE, 'Local artifact path cannot be resolved')
    if digest.hexdigest() != release.sha256:
        return result(PreflightStatus.CORRUPT, 'Local artifact SHA-256 does not match the declared release')
    return result(PreflightStatus.READY, 'Local artifact SHA-256 matches the declared release')


def preflight_protocol(plan: ResolvedProtocolPlan, data_root: str | Path | None = None) -> PreflightReport:
    """Verify local bytes and license declarations; never fetch external artifacts."""
    root = Path(data_root) if data_root is not None else None
    if root is not None:
        if root.anchor.startswith(('\\\\', '//')):
            raise ProtocolError('Data root must be local, not a UNC path')
        root = root.resolve()
        if root.anchor.startswith(('\\\\', '//')):
            raise ProtocolError('Data root resolves to a non-local path')
    checks = []
    for profile in plan.profiles:
        artifact = _check_artifact(profile.dataset_id, profile.release, root)
        if profile.release.license is None:
            checks.append(DatasetCheck(profile.dataset_id, PreflightStatus.UNKNOWN_LICENSE,
                                       'No license declaration for the dataset release'))
            if artifact.status is PreflightStatus.READY:
                continue
        checks.append(artifact)
    return PreflightReport(tuple(checks))
