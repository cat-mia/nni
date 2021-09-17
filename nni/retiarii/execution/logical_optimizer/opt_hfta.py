# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math

from nni.retiarii.operation_def.torch_op_def import HFTAOperation
from typing import List, Dict, Tuple

from nni.retiarii.utils import uid
from nni.retiarii.evaluator.pytorch.cgo.evaluator import MultiModelSupervisedLearningModule
from nni.common.device import GPUDevice

from ...graph import Graph, Model, Node
from .interface import AbstractOptimizer
from .logical_plan import (AbstractLogicalNode, LogicalGraph, LogicalPlan,
                           OriginNode)


_supported_evaluators = [MultiModelSupervisedLearningModule]


class HFTANode(AbstractLogicalNode):
    def __init__(self, logical_graph: LogicalGraph, node_id: int,
                 nodes_to_batch: List[Node], _internal=False):
        super().__init__(logical_graph, node_id,
                         "HFTA_batch_" + nodes_to_batch[0].name,
                         nodes_to_batch[0].operation)
        self.origin_nodes: List[OriginNode] = nodes_to_batch.copy()
        self.related_models = [_.original_graph.model for _ in self.origin_nodes]

    def assemble(self, multi_model_placement: Dict[Model, GPUDevice], max_model_num) -> List[Tuple[Node, GPUDevice]]:
        new_nodes = []
        for node in self.origin_nodes:
            if node.original_graph.model in multi_model_placement:
                for predecessor in node.predecessors():
                    if '_inputs' in predecessor.name :# predecessor is input node
                        input_reshape = True
                        break
                # if len(multi_model_placement) <= max_model_num, generate 2 or more HFTANodes
                if len(multi_model_placement) <= max_model_num:
                    hfta_nodes_num = math.ceil(len(multi_model_placement) / float(max_model_num))
                    for _ in hfta_nodes_num:
                        new_nodes.append(Node())
                hfta_operation = HFTAOperation(node.operation, #params, min(max_model_num, len(multi_model_placement)), 
                input_reshape)
                new_node = Node(node.original_graph, node.id,
                                f'M_{node.original_graph.model.model_id}_{node.name}',
                                hfta_operation) # hfta operation
                return new_node, multi_model_placement[node.original_graph.model]
        raise ValueError(f'HFTABatchNode {self.name} does not contain nodes from multi_model')



class HFTAOptimizer(AbstractOptimizer):
    def __init__(self) -> None:
        pass

    def _check_supported_evaluator(self, evaluator):
        for e in _supported_evaluators:
            if isinstance(evaluator, e):
                return True
        return False

    def _check_hfta_models(self, logical_plan: LogicalPlan):
        input_node = logical_plan.logical_graph.get_nodes_by_type("_inputs")#.input_node
        assert(len(input_node)==1)
        # bfs traverse
        queue = [input_node]
        visited = set()
        visited.add(input_node)
        second_layer_nodes = input_node.successors()
        last_node_graph = None
        node_graph = first_graph = second_layer_nodes[0].original_graph
        first_graph_operation = [] # current layer
        index = -1
        while len(queue) > 0:
            cur_node = queue.pop(0)
            next_nodes = cur_node.successors()
            
            for node in next_nodes:
                if node not in visited:
                    queue.append(node)
                    visited.add(node)
                    
                    node_graph = node.original_graph

                    if node_graph == first_graph:
                        if last_node_graph != first_graph:
                            # traverse to a new layer
                            first_graph_operation = []
                        first_graph_operation.append(node.operation)
                    
                    if last_node_graph == first_graph and node_graph != first_graph:
                        operation_num = len(first_graph_operation)

                    if last_node_graph != node_graph: # traverse to next graph
                        if index != -1 and index != operation_num - 1: # operation numbers are not equal
                            return False
                        # start comparing
                        index = 0
                        
                    if node_graph != first_graph:
                        assert(index <= operation_num)
                        if node.operation != first_graph_operation[index]:
                            return False
                        index += 1

                    last_node_graph = node_graph
        return True

    # def convert(self, logical_plan: LogicalPlan) -> None:
    #     if _check_hfta_models(logical_plan):
    #         # batch all nodes
    #         