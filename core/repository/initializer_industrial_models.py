import random
from pathlib import Path
from typing import Sequence

from fedot.api.api_utils.api_composer import ApiComposer
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.dag.verification_rules import ERROR_PREFIX
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import SearchSpace
from fedot.core.pipelines.verification import class_rules
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum

from core.architecture.utils.utils import PROJECT_PATH
from core.tuning.search_space import get_industrial_search_space


def add_preprocessing(pipeline: Pipeline, requirements, params, **kwargs) -> Pipeline:
    task = Task(TaskTypesEnum.classification)
    basis_models = get_operations_for_task(task=task, mode='data_operation', tags=["basis"])
    extractors = get_operations_for_task(task=task, mode='data_operation', tags=["extractor"])

    basis_model = PipelineNode(random.choice(basis_models))
    extractor_model = PipelineNode(random.choice(extractors), nodes_from=[basis_model])

    node_to_mutate = random.choice(pipeline.nodes)
    if node_to_mutate.nodes_from:
        node_to_mutate.nodes_from.append(extractor_model)
    else:
        node_to_mutate.nodes_from = [extractor_model]
    pipeline.nodes.append(basis_model)
    pipeline.nodes.append(extractor_model)

    return pipeline


def _get_default_mutations(task_type: TaskTypesEnum) -> Sequence[MutationTypesEnum]:
    mutations = [parameter_change_mutation,
                 MutationTypesEnum.single_change,
                 MutationTypesEnum.single_drop,
                 MutationTypesEnum.single_add,
                 add_preprocessing
                 ]

    return mutations


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
    setattr(ApiComposer, "_get_default_mutations", _get_default_mutations)
    # setattr(PopulationalOptimizer, '__init__', init_pop)