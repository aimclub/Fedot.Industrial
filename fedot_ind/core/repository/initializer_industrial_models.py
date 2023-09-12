import pathlib
import random
from typing import Sequence

from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation, boosting_mutation
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum
from golem.core.dag.verification_rules import ERROR_PREFIX
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum

from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.tuning.search_space import get_industrial_search_space


def add_preprocessing(pipeline: Pipeline, **kwargs) -> Pipeline:
    task = Task(TaskTypesEnum.classification)
    basis_models = get_operations_for_task(task=task, mode='data_operation', tags=["basis"])
    extractors = get_operations_for_task(task=task, mode='data_operation', tags=["extractor"])
    models = get_operations_for_task(task=task, mode='model')
    basis_model = PipelineNode(random.choice(basis_models))
    extractor_model = PipelineNode(random.choice(extractors), nodes_from=[basis_model])

    try:
        node_to_mutate = list(filter(lambda x: x.name in models, pipeline.nodes))[0]
    except:
        pipeline.show()
    if node_to_mutate.nodes_from:
        node_to_mutate.nodes_from.append(extractor_model)
    else:
        node_to_mutate.nodes_from = [extractor_model]
    pipeline.nodes.append(basis_model)
    pipeline.nodes.append(extractor_model)

    return pipeline


def _get_default_industrial_mutations(task_type: TaskTypesEnum) -> Sequence[MutationTypesEnum]:
    mutations = [parameter_change_mutation,
                 MutationTypesEnum.single_change,
                 add_preprocessing
                 ]
    return mutations


def _get_default_mutations(task_type: TaskTypesEnum) -> Sequence[MutationTypesEnum]:
    mutations = [parameter_change_mutation,
                 MutationTypesEnum.single_change,
                 MutationTypesEnum.single_drop,
                 MutationTypesEnum.single_add]

    # TODO remove workaround after boosting mutation fix
    if task_type == TaskTypesEnum.ts_forecasting:
        mutations.append(boosting_mutation)
    # TODO remove workaround after validation fix
    if task_type is not TaskTypesEnum.ts_forecasting:
        mutations.append(MutationTypesEnum.single_edge)

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


class IndustrialModels:
    def __init__(self):
        self.industrial_data_operation_path = pathlib.Path(PROJECT_PATH, 'fedot_ind',
                                                           'core',
                                                           'repository',
                                                           'data',
                                                           'industrial_data_operation_repository.json')
        self.base_data_operation_path = pathlib.Path('data_operation_repository.json')

    def __enter__(self):
        """
        Switching to industrial models
        """
        OperationTypesRepository.__repository_dict__.update({'data_operation':
                                                                 {'file': self.industrial_data_operation_path,
                                                                  'initialized_repo': None,
                                                                  'default_tags': []}})
        OperationTypesRepository.assign_repo('data_operation', self.industrial_data_operation_path)
        setattr(PipelineSearchSpace, "get_parameters_dict", get_industrial_search_space)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Switching to fedot models.
        """
        OperationTypesRepository.__repository_dict__.update({'data_operation':
                                                                 {'file': self.base_data_operation_path,
                                                                  'initialized_repo': None,
                                                                  'default_tags': [
                                                                      OperationTypesRepository.DEFAULT_DATA_OPERATION_TAGS]}})
        OperationTypesRepository.assign_repo('data_operation', self.base_data_operation_path)

        # setattr(ApiComposer, "_get_default_mutations", _get_default_mutations)
