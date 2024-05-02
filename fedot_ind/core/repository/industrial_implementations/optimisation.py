import itertools
from copy import deepcopy
from itertools import chain
from math import ceil
from random import choice, sample
from typing import Sequence
from typing import Tuple
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.operations.atomized_model import AtomizedModel
from fedot.core.composer.gp_composer.specific_operators import boosting_mutation, parameter_change_mutation
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task
from fedot.core.repository.tasks import TaskTypesEnum
from golem.core.adapter import register_native
from golem.core.dag.graph import ReconnectType
from golem.core.dag.graph_utils import graph_has_cycle
from golem.core.dag.graph_utils import node_depth, nodes_from_layer
from golem.core.dag.verification_rules import ERROR_PREFIX
from golem.core.optimisers.advisor import RemoveType
from golem.core.optimisers.genetic.gp_operators import equivalent_subtree, replace_subtrees
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverCallable, CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import AlgorithmParameters
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.utilities.data_structures import ComparableEnum as Enum
from fedot.core.operations.atomized_model import AtomizedModel as Atom
from fedot_ind.core.repository.constanst_repository import EXCLUDED_OPERATION_MUTATION
from fedot_ind.core.repository.model_repository import AtomizedModel, TEMPORARY_EXCLUDED, \
    default_industrial_availiable_operation


class MutationStrengthEnumIndustrial(Enum):
    weak = 1.0
    mean = 3.0
    strong = 5.0


class IndustrialMutations:
    def __init__(self, task_type):
        self.node_adapter = PipelineAdapter()
        self.task_type = Task(task_type)
        self.excluded_mutation = EXCLUDED_OPERATION_MUTATION[self.task_type.task_type.value]
        self.industrial_data_operations = default_industrial_availiable_operation(self.task_type.task_type.value)
        self.excluded = [list(TEMPORARY_EXCLUDED[x].keys())
                         for x in TEMPORARY_EXCLUDED.keys()]
        self.excluded = (list(itertools.chain(*self.excluded)))
        self.excluded = self.excluded + self.excluded_mutation
        self.industrial_data_operations = [operation for operation in self.industrial_data_operations if operation
                                           not in self.excluded]

    def transform_to_pipeline_node(self, node):
        return self.node_adapter._transform_to_pipeline_node(node)

    def transform_to_opt_node(self, node):
        return self.node_adapter._transform_to_opt_node(node)

    def single_edge_mutation(self,
                             graph: OptGraph,
                             requirements: GraphRequirements,
                             graph_gen_params: GraphGenerationParams,
                             parameters: GPAlgorithmParameters
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
        new_node = node_factory.get_parent_node(
            self.transform_to_opt_node(node_to_mutate), is_primary=False)
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
        new_node = node_factory.get_parent_node(
            self.transform_to_opt_node(node_to_mutate), is_primary=True)
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
        new_node_child = choice(
            old_node_children) if old_node_children else None

        while True:
            new_node = node_factory.get_node(is_primary=False)
            if new_node.name in self.industrial_data_operations:
                break
        if not new_node:
            return graph

        new_node = self.transform_to_pipeline_node(new_node)

        if graph.depth == 1:
            graph.add_node(new_node)
            graph.connect_nodes(node_parent=new_node,
                                node_child=node_to_mutate)
        else:
            graph.add_node(new_node)
            graph.connect_nodes(node_parent=node_to_mutate,
                                node_child=new_node)

        if new_node_child:
            graph.connect_nodes(node_parent=new_node,
                                node_child=new_node_child)
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
        new_node = graph_gen_params.node_factory.exchange_node(
            self.transform_to_opt_node(node))
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
            raise ValueError(
                "Unknown advice (RemoveType) returned by Advisor ")
        return graph

    def add_preprocessing(self,
                          pipeline: Pipeline, **kwargs) -> Pipeline:

        basis_models = get_operations_for_task(
            task=self.task_type, mode='data_operation', tags=["basis"])
        extractors = get_operations_for_task(
            task=self.task_type, mode='data_operation', tags=["extractor"])
        extractors = [x for x in extractors if x in self.industrial_data_operations]
        models = get_operations_for_task(task=self.task_type, mode='model')
        models = [x for x in models if x not in self.excluded_mutation]
        basis_model = PipelineNode(choice(basis_models))
        extractor_model = PipelineNode(
            choice(extractors), nodes_from=[basis_model])
        node_to_mutate = list(
            filter(lambda x: x.name in models, pipeline.nodes))[0]
        if node_to_mutate.nodes_from:
            node_to_mutate.nodes_from.append(extractor_model)
        else:
            node_to_mutate.nodes_from = [extractor_model]
        pipeline.nodes.append(basis_model)
        pipeline.nodes.append(extractor_model)

        return pipeline

    def add_lagged(self, pipeline: Pipeline, **kwargs) -> Pipeline:
        lagged = ['lagged', 'ridge']
        current_operation = list(reversed([x.name for x in pipeline.nodes]))
        if 'lagged' in current_operation:
            return pipeline
        else:
            pipeline = PipelineBuilder().add_sequence(*lagged, branch_idx=0).\
                add_sequence(*current_operation, branch_idx=1).join_branches('ridge').build()
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
    # TODO remove workaround after boosting mutation fix
    if task_type == TaskTypesEnum.ts_forecasting:
        mutations.append(boosting_mutation)
        #mutations.append(ind_mutations.add_lagged)
        mutations.remove(ind_mutations.add_preprocessing)
        mutations.remove(ind_mutations.single_add)
    # TODO remove workaround after validation fix
    if task_type is not TaskTypesEnum.ts_forecasting:
        mutations.append(MutationTypesEnum.single_edge)
    return mutations


class IndustrialCrossover:
    @register_native
    def subtree_crossover(self,
                          graph_1: OptGraph,
                          graph_2: OptGraph,
                          max_depth: int, inplace: bool = True) -> Tuple[OptGraph, OptGraph]:
        """Performed by the replacement of random subtree
        in first selected parent to random subtree from the second parent"""

        if not inplace:
            graph_1 = deepcopy(graph_1)
            graph_2 = deepcopy(graph_2)
        else:
            graph_1 = graph_1
            graph_2 = graph_2

        random_layer_in_graph_first = choice(range(graph_1.depth))
        min_second_layer = 1 if random_layer_in_graph_first == 0 and graph_2.depth > 1 else 0
        random_layer_in_graph_second = choice(
            range(min_second_layer, graph_2.depth))

        node_from_graph_first = choice(nodes_from_layer(
            graph_1, random_layer_in_graph_first))
        node_from_graph_second = choice(nodes_from_layer(
            graph_2, random_layer_in_graph_second))

        replace_subtrees(graph_1, graph_2, node_from_graph_first, node_from_graph_second,
                         random_layer_in_graph_first, random_layer_in_graph_second, max_depth)

        return graph_1, graph_2

    @register_native
    def one_point_crossover(self,
                            graph_first: OptGraph,
                            graph_second: OptGraph,
                            max_depth: int) -> Tuple[OptGraph, OptGraph]:
        """Finds common structural parts between two trees, and after that randomly
        chooses the location of nodes, subtrees of which will be swapped"""
        pairs_of_nodes = equivalent_subtree(graph_first, graph_second)
        if pairs_of_nodes:
            node_from_graph_first, node_from_graph_second = choice(
                pairs_of_nodes)

            layer_in_graph_first = graph_first.depth - \
                                   node_depth(node_from_graph_first)
            layer_in_graph_second = graph_second.depth - \
                                    node_depth(node_from_graph_second)

            replace_subtrees(graph_first, graph_second, node_from_graph_first, node_from_graph_second,
                             layer_in_graph_first, layer_in_graph_second, max_depth)
        return graph_first, graph_second

    @register_native
    def exchange_edges_crossover(self,
                                 graph_first: OptGraph,
                                 graph_second: OptGraph,
                                 max_depth):
        """Parents exchange a certain number of edges with each other. The number of
        edges is defined as half of the minimum number of edges of both parents, rounded up"""

        def find_edges_in_other_graph(edges, graph: OptGraph):
            new_edges = []
            for parent, child in edges:
                parent_new = graph.get_nodes_by_name(str(parent))
                if parent_new:
                    parent_new = parent_new[0]
                else:
                    parent_new = OptNode(str(parent))
                    graph.add_node(parent_new)
                child_new = graph.get_nodes_by_name(str(child))
                if child_new:
                    child_new = child_new[0]
                else:
                    child_new = OptNode(str(child))
                    graph.add_node(child_new)
                new_edges.append((parent_new, child_new))
            return new_edges

        edges_1 = graph_first.get_edges()
        edges_2 = graph_second.get_edges()
        count = ceil(min(len(edges_1), len(edges_2)) / 2)
        choice_edges_1 = sample(edges_1, count)
        choice_edges_2 = sample(edges_2, count)

        for parent, child in choice_edges_1:
            child.nodes_from.remove(parent)
        for parent, child in choice_edges_2:
            child.nodes_from.remove(parent)

        old_edges1 = graph_first.get_edges()
        old_edges2 = graph_second.get_edges()

        new_edges_2 = find_edges_in_other_graph(choice_edges_1, graph_second)
        new_edges_1 = find_edges_in_other_graph(choice_edges_2, graph_first)

        for parent, child in new_edges_1:
            if (parent, child) not in old_edges1:
                child.nodes_from.append(parent)
        for parent, child in new_edges_2:
            if (parent, child) not in old_edges2:
                child.nodes_from.append(parent)

        return graph_first, graph_second

    @register_native
    def exchange_parents_one_crossover(self,
                                       graph_first: OptGraph, graph_second: OptGraph, max_depth: int):
        """For the selected node for the first parent, change the parent nodes to
        the parent nodes of the same node of the second parent. Thus, the first child is obtained.
        The second child is a copy of the second parent"""

        def find_nodes_in_other_graph(nodes, graph: OptGraph):
            new_nodes = []
            for node in nodes:
                new_node = graph.get_nodes_by_name(str(node))
                if new_node:
                    new_node = new_node[0]
                else:
                    new_node = OptNode(str(node))
                    graph.add_node(new_node)
                new_nodes.append(new_node)
            return new_nodes

        edges = graph_second.get_edges()
        nodes_with_parent_or_child = list(set(chain(*edges)))
        if nodes_with_parent_or_child:

            selected_node = choice(nodes_with_parent_or_child)
            parents = selected_node.nodes_from

            node_from_first_graph = find_nodes_in_other_graph(
                [selected_node], graph_first)[0]

            node_from_first_graph.nodes_from = []
            old_edges1 = graph_first.get_edges()

            if parents:
                parents_in_first_graph = find_nodes_in_other_graph(
                    parents, graph_first)
                for parent in parents_in_first_graph:
                    if (parent, node_from_first_graph) not in old_edges1:
                        node_from_first_graph.nodes_from.append(parent)

        return graph_first, graph_second

    @register_native
    def exchange_parents_both_crossover(self,
                                        graph_first: OptGraph, graph_second: OptGraph, max_depth: int):
        """For the selected node for the first parent, change the parent nodes to
        the parent nodes of the same node of the second parent. Thus, the first child is obtained.
        The second child is formed in a similar way"""

        parents_in_first_graph = []
        parents_in_second_graph = []

        def find_nodes_in_other_graph(nodes, graph: OptGraph):
            new_nodes = []
            for node in nodes:
                new_node = graph.get_nodes_by_name(str(node))
                if new_node:
                    new_node = new_node[0]
                else:
                    new_node = OptNode(str(node))
                    graph.add_node(new_node)
                new_nodes.append(new_node)
            return new_nodes

        edges = graph_second.get_edges()
        nodes_with_parent_or_child = list(set(chain(*edges)))
        if nodes_with_parent_or_child:

            selected_node2 = choice(nodes_with_parent_or_child)
            parents2 = selected_node2.nodes_from
            if parents2:
                parents_in_first_graph = find_nodes_in_other_graph(
                    parents2, graph_first)

            selected_node1 = find_nodes_in_other_graph(
                [selected_node2], graph_first)[0]
            parents1 = selected_node1.nodes_from
            if parents1:
                parents_in_second_graph = find_nodes_in_other_graph(
                    parents1, graph_second)

            for p in parents1:
                selected_node1.nodes_from.remove(p)
            for p in parents2:
                selected_node2.nodes_from.remove(p)

            old_edges1 = graph_first.get_edges()
            old_edges2 = graph_second.get_edges()

            for parent in parents_in_first_graph:
                if (parent, selected_node1) not in old_edges1:
                    selected_node1.nodes_from.append(parent)

            for parent in parents_in_second_graph:
                if (parent, selected_node2) not in old_edges2:
                    selected_node2.nodes_from.append(parent)

        return graph_first, graph_second


def has_no_data_flow_conflicts_in_industrial_pipeline(pipeline: Pipeline):
    """ Function checks the correctness of connection between nodes """
    task = Task(TaskTypesEnum.classification)
    basis_models = get_operations_for_task(
        task=task, mode='data_operation', tags=["basis"])
    extractor = get_operations_for_task(
        task=task, mode='data_operation', tags=["extractor"])
    other = get_operations_for_task(
        task=task, forbidden_tags=["basis", "extractor"])

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
                if current_operation in basis_models and \
                        pipeline.nodes[idx + 1].operation.operation_type not in extractor:
                    raise ValueError(
                        f'{ERROR_PREFIX} Pipeline has incorrect subgraph with wrong parent nodes combination. '
                        f'Basis output should contain feature transformation')

        else:
            continue
    return True


def _crossover_by_type(self, crossover_type: CrossoverTypesEnum) -> None:
    ind_crossover = IndustrialCrossover()
    return None
