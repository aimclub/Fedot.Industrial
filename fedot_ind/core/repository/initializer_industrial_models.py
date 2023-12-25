import pathlib
import random
from enum import Enum
from typing import Sequence
from random import choice, sample
from typing import List, Iterable, Union, Optional

import numpy as np
from fedot.core.composer.metrics import QualityMetric, from_maximised_metric, F1

from fedot.core.data.array_utilities import atleast_4d

from fedot.api.api_utils.api_composer import ApiComposer
from fedot.api.api_utils.api_params_repository import ApiParamsRepository
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation, boosting_mutation
from fedot.core.data.merge.data_merger import ImageDataMerger
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.pipelines.verification import class_rules, common_rules
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum
from golem.core.dag.graph import ReconnectType
from golem.core.dag.graph_utils import graph_has_cycle
from golem.core.dag.verification_rules import ERROR_PREFIX
from golem.core.optimisers.advisor import RemoveType
from golem.core.optimisers.genetic.operators.base_mutations import MutationStrengthEnum
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, AlgorithmParameters

from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.repository.model_repository import INDUSTRIAL_PREPROC_MODEL, AtomizedModel
from fedot_ind.core.tuning.search_space import get_industrial_search_space

import numpy as np
from sklearn.metrics import f1_score

from fedot.core.data.data import InputData, OutputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TaskTypesEnum


class MutationStrengthEnumIndustrial(Enum):
    weak = 1.0
    mean = 3.0
    strong = 5.0


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

    def add_intermediate_node(self,
                              graph: OptGraph,
                              node_to_mutate: OptNode,
                              node_factory: OptNodeFactory) -> OptGraph:
        # add between node and parent
        new_node = node_factory.get_parent_node(self.transform_to_opt_node(node_to_mutate), is_primary=False)
        new_node = self.transform_to_pipeline_node(new_node)
        if not new_node:
            return graph

        # rewire old children to new parent
        new_node.nodes_from = node_to_mutate.nodes_from
        node_to_mutate.nodes_from = [new_node]

        # add new node to graph
        graph.add_node(new_node)
        return graph

    def add_separate_parent_node(self,
                                 graph: OptGraph,
                                 node_to_mutate: PipelineNode,
                                 node_factory: OptNodeFactory) -> OptGraph:
        # add as separate parent
        new_node = node_factory.get_parent_node(self.transform_to_opt_node(node_to_mutate), is_primary=True)
        new_node = self.transform_to_pipeline_node(new_node)
        if not new_node:
            # there is no possible operators
            return graph
        if node_to_mutate.nodes_from:
            node_to_mutate.nodes_from.append(new_node)
        else:
            node_to_mutate.nodes_from = [new_node]
        graph.nodes.append(new_node)
        return graph

    def add_as_child(self,
                     graph: OptGraph,
                     node_to_mutate: OptNode,
                     node_factory: OptNodeFactory) -> OptGraph:
        # add as child
        old_node_children = graph.node_children(node_to_mutate)
        new_node_child = choice(old_node_children) if old_node_children else None

        while True:
            new_node = node_factory.get_node(is_primary=False)
            if new_node.name not in self.industrial_data_operations:
                break
        if not new_node:
            return graph
        new_node = self.transform_to_pipeline_node(new_node)
        graph.add_node(new_node)
        graph.connect_nodes(node_parent=node_to_mutate, node_child=new_node)
        if new_node_child:
            graph.connect_nodes(node_parent=new_node, node_child=new_node_child)
            graph.disconnect_nodes(node_parent=node_to_mutate, node_child=new_node_child,
                                   clean_up_leftovers=True)

        return graph

    def single_add(self,
                   graph: OptGraph,
                   requirements: GraphRequirements,
                   graph_gen_params: GraphGenerationParams,
                   parameters: AlgorithmParameters
                   ) -> OptGraph:
        """
        Add new node between two sequential existing modes

        :param graph: graph to mutate
        """

        if graph.depth >= requirements.max_depth:
            # add mutation is not possible
            return graph

        node_to_mutate = choice(graph.nodes)

        single_add_strategies = [
            self.add_as_child,
            self.add_separate_parent_node
        ]
        if node_to_mutate.nodes_from:
            single_add_strategies.append(self.add_intermediate_node)
        strategy = choice(single_add_strategies)

        result = strategy(graph, node_to_mutate, graph_gen_params.node_factory)
        return result

    def single_change(self,
                      graph: OptGraph,
                      requirements: GraphRequirements,
                      graph_gen_params: GraphGenerationParams,
                      parameters: AlgorithmParameters
                      ) -> OptGraph:
        """
        Change node between two sequential existing modes.

        :param graph: graph to mutate
        """
        node = choice(graph.nodes)
        new_node = graph_gen_params.node_factory.exchange_node(self.transform_to_opt_node(node))
        if not new_node:
            return graph
        graph.update_node(node, self.transform_to_pipeline_node(new_node))
        return graph

    def single_drop(self,
                    graph: OptGraph,
                    requirements: GraphRequirements,
                    graph_gen_params: GraphGenerationParams,
                    parameters: AlgorithmParameters
                    ) -> OptGraph:
        """
        Drop single node from graph.

        :param graph: graph to mutate
        """
        if len(graph.nodes) < 2:
            return graph
        node_to_del = choice(graph.nodes)
        node_name = node_to_del.name
        removal_type = graph_gen_params.advisor.can_be_removed(node_to_del)
        if removal_type == RemoveType.with_direct_children:
            # TODO refactor workaround with data_source
            graph.delete_node(node_to_del)
            nodes_to_delete = \
                [n for n in graph.nodes
                 if n.descriptive_id.count('data_source') == 1 and node_name in n.descriptive_id]
            for child_node in nodes_to_delete:
                graph.delete_node(child_node, reconnect=ReconnectType.all)
        elif removal_type == RemoveType.with_parents:
            graph.delete_subtree(node_to_del)
        elif removal_type == RemoveType.node_rewire:
            graph.delete_node(node_to_del, reconnect=ReconnectType.all)
        elif removal_type == RemoveType.node_only:
            graph.delete_node(node_to_del, reconnect=ReconnectType.none)
        elif removal_type == RemoveType.forbidden:
            pass
        else:
            raise ValueError("Unknown advice (RemoveType) returned by Advisor ")
        return graph

    def add_preprocessing(self,
                          pipeline: Pipeline, **kwargs) -> Pipeline:

        basis_models = get_operations_for_task(task=self.task_type, mode='data_operation', tags=["basis"])
        extractors = get_operations_for_task(task=self.task_type, mode='data_operation', tags=["extractor"])
        extractors = [x for x in extractors if x != 'dimension_reduction']
        models = get_operations_for_task(task=self.task_type, mode='model')
        models = [x for x in models if x != 'fedot_cls']
        basis_model = PipelineNode(random.choice(basis_models))
        extractor_model = PipelineNode(random.choice(extractors), nodes_from=[basis_model])
        node_to_mutate = list(filter(lambda x: x.name in models, pipeline.nodes))[0]
        if node_to_mutate.nodes_from:
            node_to_mutate.nodes_from.append(extractor_model)
        else:
            node_to_mutate.nodes_from = [extractor_model]
        pipeline.nodes.append(basis_model)
        pipeline.nodes.append(extractor_model)

        return pipeline


def _get_default_industrial_mutations(task_type: TaskTypesEnum, params) -> Sequence[MutationTypesEnum]:
    ind_mutations = IndustrialMutations(task_type=task_type)
    mutations = [
        parameter_change_mutation,
        ind_mutations.single_change,
        ind_mutations.add_preprocessing,
        # IndustrialMutations().single_drop,
        ind_mutations.single_add
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

    for idx, node in enumerate(pipeline.nodes):
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
                if current_operation in basis_models and pipeline.nodes[
                    idx + 1].operation.operation_type not in extractor:
                    raise ValueError(
                        f'{ERROR_PREFIX} Pipeline has incorrect subgraph with wrong parent nodes combination. '
                        f'Basis output should contain feature transformation')

        else:
            continue
    return True


def preprocess_predicts(*args) -> List[np.array]:
    predicts = args[1]
    if len(predicts[0].shape) <= 3:
        return predicts
    else:
        reshaped_predicts = list(map(atleast_4d, predicts))

        # And check image sizes
        img_wh = [predict.shape[1:3] for predict in reshaped_predicts]
        invalid_sizes = len(set(img_wh)) > 1  # Can merge only images of the same size
        if invalid_sizes:
            raise ValueError("Can't merge images of different sizes: " + str(img_wh))
        return reshaped_predicts


def merge_predicts(*args) -> np.array:
    predicts = args[1]
    predicts = [x.reshape(-1, 1) if len(x.shape) == 1 else x for x in predicts]

    channel_shape, elem_shape = [(x.shape[1], x.shape[2]) if len(x.shape) > 2 else (1, x.shape[0]) for x in predicts][0]

    chanel_concat = [x.shape[1] == channel_shape if len(x.shape) > 2
                     else 1 == channel_shape for x in predicts]

    element_wise_concat = [x.shape[2] == elem_shape if len(x.shape) > 2
                           else x.shape[1] == elem_shape for x in predicts]

    if all(chanel_concat) and all(element_wise_concat):
        try:
            return np.concatenate(predicts, axis=1)
        except Exception:
            return np.concatenate(predicts, axis=0)
    elif not all(chanel_concat) and not all(element_wise_concat):
        prediction_2d = np.concatenate([x.reshape(x.shape[0], x.shape[1] * x.shape[2]) if len(x.shape) > 2
                                        else x for x in predicts], axis=1)
        return prediction_2d.reshape(prediction_2d.shape[0], 1, prediction_2d.shape[1])
    else:
        return np.concatenate(predicts, axis=1)


@staticmethod
@from_maximised_metric
def metric(reference: InputData, predicted: OutputData) -> float:
    n_classes = reference.num_classes
    default_value = 0
    output_mode = 'labels'
    binary_averaging_mode = 'binary'
    multiclass_averaging_mode = 'weighted'
    if n_classes > 2:
        additional_params = {'average': multiclass_averaging_mode}
    else:
        u, count = np.unique(np.ravel(reference.target), return_counts=True)
        count_sort_ind = np.argsort(count)
        pos_label = u[count_sort_ind[0]].item()
        additional_params = {'average': binary_averaging_mode, 'pos_label': pos_label}
    try:
        return f1_score(y_true=reference.target, y_pred=predicted.predict,
                        **additional_params)
    except Exception:
        additional_params = {'average': multiclass_averaging_mode}
        if predicted.predict.shape[1] > reference.target.shape[1]:
            predicted.predict = np.argmax(predicted.predict, axis=1)
        return f1_score(y_true=reference.target, y_pred=predicted.predict,
                        **additional_params)


class IndustrialModels:
    def __init__(self):
        self.industrial_data_operation_path = pathlib.Path(PROJECT_PATH, 'fedot_ind',
                                                           'core',
                                                           'repository',
                                                           'data',
                                                           'industrial_data_operation_repository.json')
        self.base_data_operation_path = pathlib.Path('data_operation_repository.json')

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

        OperationTypesRepository.assign_repo('data_operation', self.industrial_data_operation_path)

        OperationTypesRepository.__repository_dict__.update(
            {'model': {'file': self.industrial_model_path,
                       'initialized_repo': True,
                       'default_tags': []}})
        OperationTypesRepository.assign_repo('model', self.industrial_model_path)

        setattr(PipelineSearchSpace, "get_parameters_dict", get_industrial_search_space)
        setattr(ApiParamsRepository, "_get_default_mutations", _get_default_industrial_mutations)
        setattr(ImageDataMerger, "preprocess_predicts", preprocess_predicts)
        setattr(ImageDataMerger, "merge_predicts", merge_predicts)
        setattr(F1, "merge_predicts", metric)
        class_rules.append(has_no_data_flow_conflicts_in_industrial_pipeline)
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

        OperationTypesRepository.assign_repo('data_operation', self.industrial_data_operation_path)

        OperationTypesRepository.__repository_dict__.update(
            {'model': {'file': self.industrial_model_path,
                       'initialized_repo': True,
                       'default_tags': []}})
        OperationTypesRepository.assign_repo('model', self.industrial_model_path)

        setattr(PipelineSearchSpace, "get_parameters_dict", get_industrial_search_space)
        setattr(ApiComposer, "_get_default_mutations", _get_default_industrial_mutations)
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
        OperationTypesRepository.assign_repo('data_operation', self.base_data_operation_path)

        OperationTypesRepository.__repository_dict__.update(
            {'model': {'file': self.base_model_path,
                       'initialized_repo': None,
                       'default_tags': []}})
        OperationTypesRepository.assign_repo('model', self.base_model_path)
