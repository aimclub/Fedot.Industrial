from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.core.models.detection.anomaly_detector import AnomalyDetector
from fedot_ind.core.repository.constanst_repository import VALID_LINEAR_DETECTION_PIPELINE


class StatisticalDetector(AnomalyDetector):
    """Statistical anomaly detector is build on QuantileExtractor and sklearn.OneClassSVM.

    Args:
        params: additional parameters for a statistical model

            .. details:: Possible parameters:

                    - ``scale_ts`` -> a flag indicating whether to add a scaling node or not
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.node_list = VALID_LINEAR_DETECTION_PIPELINE['stat_detector']
        self.scale_ts = self.params.get('scale', False)

    def build_model(self):
        model_impl = PipelineBuilder()
        if self.scale_ts:
            self.node_list.insert(0, 'scaling')
        for node in self.node_list:
            model_impl.add_node(node)
        model_impl = model_impl.build()
        return model_impl
