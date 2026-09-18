"""Input processing service for ``FedotIndustrial``."""

from copy import deepcopy
from typing import Any, Optional

from fedot_ind.api.flow import ProcessedInputBundle
from fedot_ind.api.utils.checkers_collections import DataCheck


class IndustrialInputProcessor:
    """Prepare user input data for fitting or prediction."""

    def process(
            self,
            input_data: Any,
            *,
            task: str,
            task_params: Optional[dict] = None,
            fit_stage: bool = True,
            industrial_task_params: Any = None,
            default_fedot_context: bool = False,
    ) -> ProcessedInputBundle:
        data_copy = deepcopy(input_data)
        data_check = DataCheck(
            input_data=data_copy,
            task=task,
            task_params=task_params,
            fit_stage=fit_stage,
            industrial_task_params=industrial_task_params,
        )
        checked_data = data_check.check_input_data()
        target_encoder = data_check.get_target_encoder()
        if default_fedot_context:
            checked_data.features = checked_data.features.squeeze()
        return ProcessedInputBundle(
            data=checked_data,
            target_encoder=target_encoder,
            fit_stage=fit_stage,
        )
