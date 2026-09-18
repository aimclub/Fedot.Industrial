"""Dask runtime startup service for ``FedotIndustrial``."""

from dataclasses import dataclass
from typing import Any

from fedot_ind.core.architecture.abstraction.decorators import DaskServer


@dataclass(frozen=True)
class DaskRuntime:
    """Started Dask runtime handles."""

    client: Any
    cluster: Any


class DaskRuntimeInitializer:
    """Start the Dask runtime and return explicit handles."""

    def __init__(self, dask_server_cls=DaskServer):
        self.dask_server_cls = dask_server_cls

    def start(self, *, distributed_config: dict, logger: Any) -> DaskRuntime:
        logger.info('-' * 50)
        logger.info('Initialising Dask Server')
        dask_server = self.dask_server_cls(distributed_config)
        logger.info(f'Link Dask Server - {dask_server.client.dashboard_link}')
        return DaskRuntime(client=dask_server.client, cluster=dask_server.cluster)
