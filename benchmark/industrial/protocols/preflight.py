"""Read-only checks of explicitly supplied local dataset artifacts."""

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from .assets import AssetCorrupt, AssetIncompatible, AssetUnavailable, LocalAsset, checked_asset
from .planning import ResolvedProtocolPlan


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
        checked_asset(root, LocalAsset(release.artifact, release.sha256))
    except AssetUnavailable as error:
        return result(PreflightStatus.UNAVAILABLE, str(error))
    except AssetCorrupt as error:
        return result(PreflightStatus.CORRUPT, str(error))
    except AssetIncompatible as error:
        return result(PreflightStatus.INCOMPATIBLE, str(error))
    return result(PreflightStatus.READY, 'Local artifact SHA-256 matches the declared release')


def preflight_protocol(plan: ResolvedProtocolPlan, data_root: str | Path | None = None) -> PreflightReport:
    """Verify local bytes and license declarations; never fetch external artifacts."""
    root = Path(data_root) if data_root is not None else None
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
