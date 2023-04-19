import pathlib
from typing import Optional, Tuple

from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher, OptionalEvalResult
from golem.core.optimisers.graph import OptGraph

from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels


class IndustrialDispatcher(MultiprocessingDispatcher):
    def evaluate_single(self, graph: OptGraph, uid_of_individual: str, with_time_limit: bool = True,
                        cache_key: Optional[str] = None,
                        logs_initializer: Optional[Tuple[int, pathlib.Path]] = None) -> OptionalEvalResult:
        # we should do that due to this is copy of initial process, and it does not have industrial models in itself
        if self._n_jobs != 1:
            with IndustrialModels():
                eval_res = super().evaluate_single(graph, uid_of_individual, with_time_limit, cache_key, logs_initializer)
        else:
            eval_res = super().evaluate_single(graph, uid_of_individual, with_time_limit, cache_key, logs_initializer)
        return eval_res
