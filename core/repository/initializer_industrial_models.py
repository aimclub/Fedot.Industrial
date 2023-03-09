import pathlib
from pathlib import Path
from typing import Optional, Tuple

from fedot.core.dag.verification_rules import ERROR_PREFIX
from fedot.core.optimisers.gp_comp.evaluation import OptionalEvalResult, MultiprocessingDispatcher
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.populational_optimizer import PopulationalOptimizer
from fedot.core.pipelines.tuning.search_space import SearchSpace
from fedot.core.pipelines.verification import class_rules
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum
from gtda.pipeline import Pipeline

from core.architecture.utils.utils import PROJECT_PATH
from core.repository.IndustrialDispatcher import init_pop
from core.tuning.search_space import get_industrial_search_space


def has_no_data_flow_conflicts_in_industrial_pipeline(pipeline: Pipeline):
    """ Function checks the correctness of connection between nodes """
    task = Task(TaskTypesEnum.classification)
    basis_models = get_operations_for_task(task=task, mode='data_operation', tags=["basis"])
    extractor = get_operations_for_task(task=task, mode='data_operation', tags=["extractor"])
    other = get_operations_for_task(task=task, forbidden_tags=["basis", "extractor"])

    for node in pipeline.nodes:
        # Operation name in the current node
        current_operation = node.operation.operation_type
        parent_nodes = node.nodes_from
        if parent_nodes:
            if current_operation in basis_models:
                raise ValueError(
                    f'{ERROR_PREFIX} Pipeline has incorrect subgraph with wrong parent nodes combination')
            # There are several parents for current node or at least 1
            for parent in parent_nodes:
                parent_operation = parent.operation.operation_type
                if current_operation in extractor:
                    if parent_operation not in basis_models:
                        raise ValueError(
                            f'{ERROR_PREFIX} Pipeline has incorrect subgraph with wrong parent nodes combination')
                elif current_operation in other:
                    if parent_operation in basis_models:
                        raise ValueError(
                            f'{ERROR_PREFIX} Pipeline has incorrect subgraph with wrong parent nodes combination')
        else:
            # Only basis models can be primary
            if current_operation not in basis_models:
                raise ValueError(
                    f'{ERROR_PREFIX} Pipeline has incorrect subgraph with wrong parent nodes combination')
    return True


def initialize_industrial_models():
    OperationTypesRepository.__repository_dict__.update({'data_operation':
                                                             {'file': Path(PROJECT_PATH, 'core', 'repository', 'data',
                                                                           'data_operation_repository.json'),
                                                              'initialized_repo': None,
                                                              'default_tags': []}})
    OperationTypesRepository.init_default_repositories()
    # DefaultOperationParamsRepository.__repository_name__ = Path(PROJECT_PATH, 'core', 'repository', 'data',
    #                                                                        'default_operations_params.json')
    class_rules.append(has_no_data_flow_conflicts_in_industrial_pipeline)

    setattr(SearchSpace, "get_parameters_dict", get_industrial_search_space)
    #setattr(PopulationalOptimizer, '__init__', init_pop)

