import pathlib
from random import sample

from fedot.core.composer.metrics import F1, Accuracy
from fedot.api.api_utils.api_composer import ApiComposer
from fedot.api.api_utils.api_params_repository import ApiParamsRepository
from fedot.core.data.merge.data_merger import ImageDataMerger
from fedot.core.operations.operation import Operation
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.pipelines.verification import class_rules
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task
from golem.core.dag.graph_utils import graph_has_cycle
from golem.core.optimisers.genetic.operators.crossover import Crossover
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams

from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.repository.industrial_implementations.abstract import merge_predicts, preprocess_predicts, \
    predict_for_fit, predict, predict_operation
from fedot_ind.core.repository.industrial_implementations.metric import metric_f1, metric_acc
from fedot_ind.core.repository.industrial_implementations.optimisation import _get_default_industrial_mutations, \
    MutationStrengthEnumIndustrial, has_no_data_flow_conflicts_in_industrial_pipeline, _crossover_by_type
from fedot_ind.core.repository.model_repository import AtomizedModel
from fedot_ind.core.tuning.search_space import get_industrial_search_space

class IndustrialMutations:
    def __init__(self, task_type):
        self.node_adapter = PipelineAdapter()
        self.task_type = Task(task_type)
        self.industrial_data_operations = [*list(AtomizedModel.INDUSTRIAL_PREPROC_MODEL.value.keys()),
                                           *list(AtomizedModel.INDUSTRIAL_CLF_PREPROC_MODEL.value.keys()),
                                           *list(AtomizedModel.FEDOT_PREPROC_MODEL.value.keys())]

    def transform_to_pipeline_node(self, node):
        return self.node_adapter._transform_to_pipeline_node(node)

    def transform_to_opt_node(self, node):
        return self.node_adapter._transform_to_opt_node(node)

    def single_edge_mutation(self,
                             graph: OptGraph,
                             requirements: GraphRequirements,
                             graph_gen_params: GraphGenerationParams,
                             parameters: 'GPAlgorithmParameters'
                             ) -> OptGraph:
        """
        This mutation adds new edge between two random nodes in graph.

        :param graph: graph to mutate
        """

        def nodes_not_cycling(source_node: OptNode, target_node: OptNode):
            parents = source_node.nodes_from
            while parents:
                if target_node not in parents:
                    grandparents = []
                    for parent in parents:
                        grandparents.extend(parent.nodes_from)
                    parents = grandparents
                else:
                    return False
            return True

        for _ in range(parameters.max_num_of_operator_attempts):
            if len(graph.nodes) < 2 or graph.depth > requirements.max_depth:
                return graph

            source_node, target_node = sample(graph.nodes, 2)
            if source_node not in target_node.nodes_from:
                if graph_has_cycle(graph):
                    graph.connect_nodes(source_node, target_node)
                    break
                else:
                    if nodes_not_cycling(source_node, target_node):
                        graph.connect_nodes(source_node, target_node)
                        break
        return graph

class IndustrialModels:
    def __init__(self):
        self.industrial_data_operation_path = pathlib.Path(PROJECT_PATH, 'fedot_ind',
                                                           'core',
                                                           'repository',
                                                           'data',
                                                           'industrial_data_operation_repository.json')
        self.base_data_operation_path = pathlib.Path(
            'data_operation_repository.json')

        self.industrial_model_path = pathlib.Path(PROJECT_PATH, 'fedot_ind',
                                                  'core',
                                                  'repository',
                                                  'data',
                                                  'industrial_model_repository.json')
        self.base_model_path = pathlib.Path('model_repository.json')

    def setup_repository(self):
        OperationTypesRepository.__repository_dict__.update(
            {'data_operation': {'file': self.industrial_data_operation_path,
                                'initialized_repo': True,
                                'default_tags': []}})

        OperationTypesRepository.assign_repo(
            'data_operation', self.industrial_data_operation_path)

        OperationTypesRepository.__repository_dict__.update(
            {'model': {'file': self.industrial_model_path,
                       'initialized_repo': True,
                       'default_tags': []}})
        OperationTypesRepository.assign_repo(
            'model', self.industrial_model_path)

        setattr(PipelineSearchSpace, "get_parameters_dict",
                get_industrial_search_space)
        setattr(ApiParamsRepository, "_get_default_mutations",
                _get_default_industrial_mutations)
        setattr(Crossover, '_crossover_by_type', _crossover_by_type)
        setattr(ImageDataMerger, "preprocess_predicts", preprocess_predicts)
        setattr(ImageDataMerger, "merge_predicts", merge_predicts)
        setattr(F1, "metric", metric_f1)
        setattr(Accuracy, "metric", metric_acc)
        setattr(Operation, "_predict", predict_operation)
        setattr(Operation, "predict", predict)
        setattr(Operation, "predict_for_fit", predict_for_fit)

        # class_rules.append(has_no_data_flow_conflicts_in_industrial_pipeline)
        MutationStrengthEnum = MutationStrengthEnumIndustrial
        return OperationTypesRepository

    def __enter__(self):
        """
        Switching to industrial models
        """
        OperationTypesRepository.__repository_dict__.update(
            {'data_operation': {'file': self.industrial_data_operation_path,
                                'initialized_repo': True,
                                'default_tags': []}})

        OperationTypesRepository.assign_repo(
            'data_operation', self.industrial_data_operation_path)

        OperationTypesRepository.__repository_dict__.update(
            {'model': {'file': self.industrial_model_path,
                       'initialized_repo': True,
                       'default_tags': []}})
        OperationTypesRepository.assign_repo(
            'model', self.industrial_model_path)

        setattr(PipelineSearchSpace, "get_parameters_dict",
                get_industrial_search_space)
        setattr(ApiComposer, "_get_default_mutations",
                _get_default_industrial_mutations)
        class_rules.append(has_no_data_flow_conflicts_in_industrial_pipeline)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Switching to fedot models.
        """
        OperationTypesRepository.__repository_dict__.update(
            {'data_operation': {'file': self.base_data_operation_path,
                                'initialized_repo': None,
                                'default_tags': [
                                    OperationTypesRepository.DEFAULT_DATA_OPERATION_TAGS]}})
        OperationTypesRepository.assign_repo(
            'data_operation', self.base_data_operation_path)

        OperationTypesRepository.__repository_dict__.update(
            {'model': {'file': self.base_model_path,
                       'initialized_repo': None,
                       'default_tags': []}})
        OperationTypesRepository.assign_repo('model', self.base_model_path)
