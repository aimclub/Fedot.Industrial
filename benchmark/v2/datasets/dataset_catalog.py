from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
from pathlib import Path
from typing import Any, Optional, List, Dict, Literal,  Iterable, Optional, Sequence

from benchmark.v2.core import CovariateMode, LeakagePolicy, Track


class DatasetFamily(str, Enum):
    M4 = "m4"
    M5 = "m5"
    MONASH = "monash"
    LTSF = "ltsf"
    GIFT_EVAL = "gift_eval"
    TFB = "tfb"
    PROFT = "proft"
    FRESH = "fresh"


class Domain(str, Enum):
    ECONOMIC = "economic"
    FINANCE = "finance"
    RETAIL = "retail"
    ENERGY = "energy"
    TRAFFIC = "traffic"
    WEATHER = "weather"
    HEALTH = "health"
    DEMAND = "demand"
    SALES = "sales"
    STOCK = "stock"
    MACRO = "macro"


class TargetType(str, Enum):
    UNIVARIATE = "univariate"
    MULTIVARIATE = "multivariate"
    PANEL = "panel"


class Frequency(str, Enum):
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"
    YEARLY = "Y"
    HOURLY = "H"
    MINUTELY = "T"
    SECONDLY = "S"
    TEN_MINUTES = "10min"
    FIFTEEN_MINUTES = "15min"
    FIVE_MINUTES = "5min"


class DatasetCapability(str, Enum):
    HIERARCHICAL = "hierarchical"
    PANEL = "panel"
    STATIC_COVARIATES = "static_covariates"
    KNOWN_FUTURE_COVARIATES = "known_future_covariates"
    PROBABILISTIC = "probabilistic"
    MULTIVARIATE = "multivariate"


@dataclass
class DatasetMetadata:
    dataset_id: str
    family: DatasetFamily | str
    domain: Domain | str
    frequency: Frequency | str
    default_horizon: int
    seasonal_period: int
    target_type: TargetType | str
    covariates: list[CovariateMode | str] = field(default_factory=list)
    leakage_status: LeakagePolicy | str = LeakagePolicy.STRICT
    track: Track | str = Track.DEFAULT
    requires_download: bool = False
    description: str = ""
    default_adapter_options: dict[str, Any] = field(default_factory=dict)
    capabilities: list[DatasetCapability | str] = field(default_factory=list)
    version: str = "1.0"

    def __post_init__(self):
        if isinstance(self.family, str):
            self.family = DatasetFamily(self.family)
        if isinstance(self.domain, str):
            self.domain = Domain(self.domain)
        if isinstance(self.frequency, str):
            self.frequency = Frequency(self.frequency)
        if isinstance(self.target_type, str):
            self.target_type = TargetType(self.target_type)
        if isinstance(self.leakage_status, str):
            self.leakage_status = LeakagePolicy(self.leakage_status)
        if isinstance(self.track, str):
            self.track = Track(self.track)

        self.covariates = [
            CovariateMode(cov) if isinstance(cov, str) else cov
            for cov in self.covariates
        ]

        self.capabilities = [
            DatasetCapability(cap) if isinstance(cap, str) else cap
            for cap in self.capabilities
        ]

        if self.default_horizon <= 0:
            raise ValueError(
                f"default_horizon must be positive, got {self.default_horizon}"
            )
        if self.seasonal_period < 1:
            raise ValueError(
                f"seasonal_period must be at least 1, got {self.seasonal_period}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "family": self.family.value,
            "domain": self.domain.value,
            "frequency": self.frequency.value,
            "default_horizon": self.default_horizon,
            "seasonal_period": self.seasonal_period,
            "target_type": self.target_type.value,
            "covariates": [c.value for c in self.covariates],
            "leakage_status": self.leakage_status.value,
            "track": self.track.value,
            "requires_download": self.requires_download,
            "description": self.description,
            "default_adapter_options": self.default_adapter_options,
            "capabilities": [c.value for c in self.capabilities],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetMetadata:
        return cls(**data)


@dataclass
class DatasetFilter:
    families: Optional[list[DatasetFamily | str]] = None
    domains: Optional[list[Domain | str]] = None
    task_types: Optional[list[TargetType | str]] = None
    min_horizon: Optional[int] = None
    max_horizon: Optional[int] = None
    tracks: Optional[list[Track | str]] = None
    leakage_policies: Optional[list[LeakagePolicy | str]] = None
    requires_download: Optional[bool] = None
    dataset_ids: Optional[list[str]] = None
    capabilities: Optional[list[DatasetCapability | str]] = None

    def __post_init__(self):
        self._normalize()

    def _normalize(self):
        def norm(items, enum_cls):
            if items is None:
                return None
            return [enum_cls(x) if isinstance(x, str) else x for x in items]

        self.families = norm(self.families, DatasetFamily)
        self.domains = norm(self.domains, Domain)
        self.task_types = norm(self.task_types, TargetType)
        self.tracks = norm(self.tracks, Track)
        self.leakage_policies = norm(self.leakage_policies, LeakagePolicy)
        self.capabilities = norm(self.capabilities, DatasetCapability)

    def matches(self, meta: DatasetMetadata) -> bool:
        if self.families and meta.family not in self.families:
            return False
        if self.domains and meta.domain not in self.domains:
            return False
        if self.task_types and meta.target_type not in self.task_types:
            return False
        if self.min_horizon is not None and meta.default_horizon < self.min_horizon:
            return False
        if self.max_horizon is not None and meta.default_horizon > self.max_horizon:
            return False
        if self.tracks and meta.track not in self.tracks:
            return False
        if self.leakage_policies and meta.leakage_status not in self.leakage_policies:
            return False
        if self.requires_download is not None and meta.requires_download != self.requires_download:
            return False
        if self.dataset_ids and meta.dataset_id not in self.dataset_ids:
            return False
        if self.capabilities:
            if not all(cap in meta.capabilities for cap in self.capabilities):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        def serialize_list(items):
            if items is None:
                return None
            return [x.value if isinstance(x, Enum) else x for x in items]

        return {
            "families": serialize_list(self.families),
            "domains": serialize_list(self.domains),
            "task_types": serialize_list(self.task_types),
            "min_horizon": self.min_horizon,
            "max_horizon": self.max_horizon,
            "tracks": serialize_list(self.tracks),
            "leakage_policies": serialize_list(self.leakage_policies),
            "requires_download": self.requires_download,
            "dataset_ids": self.dataset_ids,
            "capabilities": serialize_list(self.capabilities),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetFilter:
        return cls(**data)


class DatasetCatalog:
    def __init__(self, manifest_path: Optional[Path] = None):
        self._datasets: dict[str, DatasetMetadata] = {}
        if manifest_path:
            self.load_manifest(manifest_path)
        else:
            default_path = Path(__file__).parent / "dataset_manifest.yaml"
            if default_path.exists():
                self.load_manifest(default_path)

    def load_manifest(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")
        suffix = path.suffix.lower()
        if suffix in (".json",):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif suffix in (".yaml", ".yml"):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported manifest format: {suffix}")

        if not isinstance(data, list):
            raise ValueError("Manifest must contain a list of dataset definitions.")

        for entry in data:
            meta = DatasetMetadata.from_dict(entry)
            if meta.dataset_id in self._datasets:
                raise ValueError(f"Duplicate dataset_id: {meta.dataset_id}")
            self._datasets[meta.dataset_id] = meta

    def save_manifest(self, path: Path) -> None:
        data = [meta.to_dict() for meta in self._datasets.values()]
        suffix = path.suffix.lower()
        if suffix in (".json",):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif suffix in (".yaml", ".yml"):
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported manifest format: {suffix}")

    def list_datasets(self, filter_criteria: Optional[DatasetFilter] = None) -> list[DatasetMetadata]:
        if filter_criteria is None:
            return list(self._datasets.values())
        return [meta for meta in self._datasets.values() if filter_criteria.matches(meta)]

    def get_dataset(self, dataset_id: str) -> Optional[DatasetMetadata]:
        return self._datasets.get(dataset_id)

    def get_datasets_by_family(self, family: DatasetFamily | str) -> list[DatasetMetadata]:
        if isinstance(family, str):
            family = DatasetFamily(family.lower())
        return [meta for meta in self._datasets.values() if meta.family == family]

    def get_ltsf_benchmark_suite(self) -> list[DatasetMetadata]:
        return self.get_datasets_by_family(DatasetFamily.LTSF)

    def get_datasets_with_capability(self, capability: DatasetCapability | str) -> list[DatasetMetadata]:
        if isinstance(capability, str):
            capability = DatasetCapability(capability)
        return [meta for meta in self._datasets.values() if capability in meta.capabilities]
