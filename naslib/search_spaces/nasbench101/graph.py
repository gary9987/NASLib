import numpy as np
import copy
import random
import spektral
import torch
import torch.nn.functional as F
from typing import *
from enum import Enum


from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench101.conversions import convert_spec_to_model, convert_spec_to_tuple, \
    convert_tuple_to_spec
from naslib.utils import get_dataset_api

INPUT = "input"
OUTPUT = "output"
CONV3X3 = "conv3x3-bn-relu"
CONV1X1 = "conv1x1-bn-relu"
MAXPOOL3X3 = "maxpool3x3"
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9

MODEL_TYPE = Enum('MODEL_TYPE', 'KERAS ENSEMBLE')

class NasBench101SearchSpace(Graph):
    """
    Contains the interface to the tabular benchmark of nasbench 101.
    """

    QUERYABLE = True

    def __init__(self, model_type: MODEL_TYPE, n_classes=10):
        super().__init__()
        self.num_classes = n_classes
        self.space_name = "nasbench101"
        self.spec = None
        self.labeled_archs = None
        self.instantiate_model = True
        self.sample_without_replacement = False
        self.model_type = model_type

        self.add_edge(1, 2)

    def convert_to_cell(self, matrix: np.ndarray, ops: list) -> dict:

        if len(matrix) < 7:
            # the nasbench spec can have an adjacency matrix of n x n for n<7, 
            # but in the nasbench api, it is always 7x7 (possibly containing blank rows)
            # so this method will add a blank row/column

            new_matrix = np.zeros((7, 7), dtype='int8')
            new_ops = []
            n = matrix.shape[0]
            for i in range(7):
                for j in range(7):
                    if j < n - 1 and i < n:
                        new_matrix[i][j] = matrix[i][j]
                    elif j == n - 1 and i < n:
                        new_matrix[i][-1] = matrix[i][j]

            for i in range(7):
                if i < n - 1:
                    new_ops.append(ops[i])
                elif i < 6:
                    new_ops.append('conv3x3-bn-relu')
                else:
                    new_ops.append('output')
            return {
                'matrix': new_matrix,
                'ops': new_ops
            }

        else:
            return {
                'matrix': matrix,
                'ops': ops
            }

    def convert_to_graph(self, matrix: np.ndarray, ops: list) -> spektral.data.Graph:
        nodes = 67
        features_dict = {'input': 0, 'conv1x1-bn-relu': 1, 'conv3x3-bn-relu': 2, 'maxpool3x3': 3, 'output': 4,
                         'Classifier': 5, 'maxpool2x2': 6}

        # Node features X
        new_x = np.zeros((nodes, len(features_dict)), dtype=float)  # nodes * (features + metadata + num_layer)

        for now_layer in range(11 + 1):
            if now_layer == 0 or now_layer == 4 or now_layer == 8:
                if now_layer == 0:
                    offset_idx = 0
                    new_x[offset_idx][features_dict['conv3x3-bn-relu']] = 1  # stem is a 'conv3x3-bn-relu' type
                elif now_layer == 4:
                    offset_idx = 22
                    new_x[offset_idx][features_dict['maxpool2x2']] = 1
                else:  # now_layer == 8
                    offset_idx = 44
                    new_x[offset_idx][features_dict['maxpool2x2']] = 1

            else:
                now_group = now_layer // 4 + 1
                node_start_no = now_group + 7 * (now_layer - now_group)

                for i in range(len(ops)):
                    if i == len(ops) - 1:
                        new_x[6 + node_start_no][features_dict[ops[i]]] = 1
                    else:
                        new_x[i + node_start_no][features_dict[ops[i]]] = 1

        new_x[66][features_dict['Classifier']] = 1

        # Adjacency matrix A
        adj_matrix = np.zeros((nodes, nodes), dtype=float)
        # 0 convbn 128
        # 1 cell
        # 8 cell
        # 15 cell
        # 22 maxpool
        # 23 cell
        # 30 cell
        # 37 cell
        # 44 maxpool
        # 45 cell
        # 52 cell
        # 59 cell

        for now_layer in range(11+1):
            if now_layer == 0:
                adj_matrix[0][1] = 1  # stem to input node
            elif now_layer == 4:
                adj_matrix[21][22] = 1  # output to maxpool
                adj_matrix[22][23] = 1  # maxpool to input
            elif now_layer == 8:
                adj_matrix[43][44] = 1  # output to maxpool
                adj_matrix[44][45] = 1  # maxpool to input
            else:
                now_group = now_layer // 4 + 1
                node_start_no = now_group + 7 * (now_layer - now_group)
                for i in range(matrix.shape[0]):
                    if i == matrix.shape[0] - 1:
                        if now_layer == 11:
                            # to classifier
                            adj_matrix[6 + node_start_no][nodes - 1] = 1
                        else:
                            # output node to next input node
                            adj_matrix[6 + node_start_no][i + node_start_no + (7 - matrix.shape[0]) + 1] = 1
                    else:
                        for j in range(matrix.shape[1]):
                            if matrix[i][j] == 1:
                                if j == matrix.shape[0] - 1:
                                    # X node to output node
                                    adj_matrix[i + node_start_no][6 + node_start_no] = 1
                                else:
                                    adj_matrix[i + node_start_no][j + node_start_no] = 1

        return spektral.data.Graph(x=new_x, e=None, a=adj_matrix, y=None)

    def convert_graph_to_keras_model_input(self, graph: spektral.data.Graph) -> Tuple[np.ndarray, np.ndarray]:
        '''
                data: (x, a)
                x: (1, 67, feature)
                a: (1, 67, 67)
                '''
        x = np.expand_dims(graph.x, axis=0)
        a = np.expand_dims(graph.a, axis=0)
        return (x, a)

    def convert_graph_to_ndarry_input(self, graph: spektral.data.Graph) -> np.ndarray:
        ret = graph.a.reshape((1, graph.a.shape[0] * graph.a.shape[1]))
        ret = np.concatenate((ret, graph.x.reshape((1, graph.x.shape[0] * graph.x.shape[1]))), axis=1)
        return np.array([np.squeeze(ret)])

    def surrogate_query(self, model, data) -> Dict[Metric, Union[int, float]]:

        #validation_accuracy = float(model.predict(data)[0][0])
        validation_accuracy = float(model.predict(data))
        #print(validation_accuracy)
        return {Metric.TRAIN_ACCURACY: -1, Metric.VAL_ACCURACY: validation_accuracy, Metric.TEST_ACCURACY: -1}

    def query(self,
              metric: Metric,
              dataset: str = "cifar10",
              path: str = None,
              epoch: int = -1,
              full_lc: bool = False,
              dataset_api: dict = None) -> Union[list, float]:
        """
        Query results from nasbench 101
        """
        assert isinstance(metric, Metric)
        assert dataset in ["cifar10", None], "Unknown dataset: {}".format(dataset)

        if metric in [Metric.ALL, Metric.HP]:
            raise NotImplementedError()
        if dataset_api is None:
            raise NotImplementedError("Must pass in dataset_api to query nasbench101")
        assert epoch in [
            -1,
            108,
            200,
        ], f"Metric is not available at epoch {epoch}. NAS-Bench-101 does not have full learning curve information. Available epochs are [4, 12, 36, and 108]."

        metric_to_nb101 = {
            Metric.TRAIN_ACCURACY: "train_accuracy",
            Metric.VAL_ACCURACY: "validation_accuracy",
            Metric.TEST_ACCURACY: "test_accuracy",
            Metric.TRAIN_TIME: "training_time",
            Metric.PARAMETERS: "trainable_parameters",
        }

        if self.get_spec() is None:
            raise NotImplementedError(
                "Cannot yet query directly from the naslib object"
            )

        api_spec = dataset_api["api"].ModelSpec(**self.spec)

        if not dataset_api["nb101_data"].is_valid(api_spec):
            return -1

        graph_data = self.convert_to_graph(**self.spec)
        if self.model_type == MODEL_TYPE.KERAS:
            data = self.convert_graph_to_keras_model_input(graph_data)
        elif self.model_type == MODEL_TYPE.ENSEMBLE:
            data = self.convert_graph_to_ndarry_input((graph_data))
        query_results = self.surrogate_query(dataset_api['surrogate'], data)

        if full_lc:
            raise NotImplementedError()

        if metric == Metric.RAW:
            return query_results
        elif metric == Metric.TRAIN_TIME:
            return -1
        else:
            print(query_results[metric])
            return query_results[metric]

    def get_spec(self) -> dict:
        return self.spec

    def get_hash(self) -> tuple:
        return convert_spec_to_tuple(self.get_spec())

    def set_spec(self, spec: Union[str, dict, tuple], dataset_api: dict = None) -> None:
        # TODO: convert the naslib object to this spec
        # convert_spec_to_naslib(spec, self)
        # assert self.spec is None, f"An architecture has already been assigned to this instance of {self.__class__.__name__}. Instantiate a new instance to be able to sample a new model or set a new architecture."
        assert isinstance(spec, str) or isinstance(spec, tuple) or isinstance(spec,
                                                                              dict), "The spec has to be a string (hash of the architecture), a dict with the matrix and operations, or a tuple (NASLib representation)."

        if isinstance(spec, str):
            """
            TODO: I couldn't find a better solution here.
            We need the arch iterator to return strings because the matrix/ops
            representation is too large for 400k elements. But having the `spec' be 
            strings would require passing in dataset_api for all of this search 
            space's methods. So the solution is to optionally pass in the dataset 
            api in set_spec and check whether `spec' is a string or a dict.
            """
            assert dataset_api is not None, "To set the hash string as the spec, the NAS-Bench-101 API must be passed as the dataset_api argument"
            fix, comp = dataset_api["nb101_data"].get_metrics_from_hash(spec)
            spec = self.convert_to_cell(fix['module_adjacency'], fix['module_operations'])
        elif isinstance(spec, tuple):
            spec = convert_tuple_to_spec(spec)

        if self.instantiate_model:
            assert self.spec is None, f"An architecture has already been assigned to this instance of {self.__class__.__name__}. Instantiate a new instance to be able to sample a new model or set a new architecture."
            model = convert_spec_to_model(spec)
            self.edges[1, 2].set('op', model)

        self.spec = spec

    def get_arch_iterator(self, dataset_api: dict) -> Iterator:
        return dataset_api["nb101_data"].hash_iterator()

    def sample_random_labeled_architecture(self) -> None:
        assert self.labeled_archs is not None, "Labeled archs not provided to sample from"

        while True:
            op_indices = random.choice(self.labeled_archs)
            if len(op_indices) == 56:
                break

        if self.sample_without_replacement == True:
            self.labeled_archs.pop(self.labeled_archs.index(op_indices))

        self.set_spec(op_indices)

    def sample_random_architecture(self, dataset_api: dict, load_labeled: bool = False) -> None:
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        From the NASBench repository:
        one-hot adjacency matrix
        draw [0,1] for each slot in the adjacency matrix
        """

        if load_labeled == True:
            return self.sample_random_labeled_architecture()

        while True:
            matrix = np.random.choice([0, 1], size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT

            spec = dataset_api["api"].ModelSpec(matrix=matrix, ops=ops)
            if dataset_api["nb101_data"].is_valid(spec):
                break

        self.set_spec({"matrix": matrix, "ops": ops})

    def mutate(self, parent: Graph, dataset_api: dict, edits: int = 1) -> None:
        """
        This will mutate the parent architecture spec.
        Code inspird by https://github.com/google-research/nasbench
        """
        parent_spec = parent.get_spec()
        spec = copy.deepcopy(parent_spec)
        matrix, ops = spec['matrix'], spec['ops']
        for _ in range(edits):
            while True:
                new_matrix = copy.deepcopy(matrix)
                new_ops = copy.deepcopy(ops)
                for src in range(0, NUM_VERTICES - 1):
                    for dst in range(src + 1, NUM_VERTICES):
                        if np.random.random() < 1 / NUM_VERTICES:
                            new_matrix[src][dst] = 1 - new_matrix[src][dst]
                for ind in range(1, NUM_VERTICES - 1):
                    if np.random.random() < 1 / len(OPS):
                        available = [op for op in OPS if op != new_ops[ind]]
                        new_ops[ind] = np.random.choice(available)
                new_spec = dataset_api['api'].ModelSpec(new_matrix, new_ops)
                if dataset_api['nb101_data'].is_valid(new_spec):
                    break

        self.set_spec({'matrix': new_matrix, 'ops': new_ops})

    def get_nbhd(self, dataset_api: dict) -> list:
        # return all neighbors of the architecture
        spec = self.get_spec()
        matrix, ops = spec["matrix"], spec["ops"]
        nbhd = []

        def add_to_nbhd(new_matrix: np.ndarray, new_ops: list, nbhd: list) -> list:
            new_spec = {"matrix": new_matrix, "ops": new_ops}
            model_spec = dataset_api["api"].ModelSpec(new_matrix, new_ops)
            if dataset_api["nb101_data"].is_valid(model_spec):
                nbr = NasBench101SearchSpace()
                nbr.set_spec(new_spec)
                nbr_model = torch.nn.Module()
                nbr_model.arch = nbr
                nbhd.append(nbr_model)
            return nbhd

        # add op neighbors
        for vertex in range(1, OP_SPOTS + 1):
            if is_valid_vertex(matrix, vertex):
                available = [op for op in OPS if op != ops[vertex]]
                for op in available:
                    new_matrix = copy.deepcopy(matrix)
                    new_ops = copy.deepcopy(ops)
                    new_ops[vertex] = op
                    nbhd = add_to_nbhd(new_matrix, new_ops, nbhd)

        # add edge neighbors
        for src in range(0, NUM_VERTICES - 1):
            for dst in range(src + 1, NUM_VERTICES):
                new_matrix = copy.deepcopy(matrix)
                new_ops = copy.deepcopy(ops)
                new_matrix[src][dst] = 1 - new_matrix[src][dst]
                new_spec = {"matrix": new_matrix, "ops": new_ops}

                if matrix[src][dst] and is_valid_edge(matrix, (src, dst)):
                    nbhd = add_to_nbhd(new_matrix, new_ops, nbhd)

                if not matrix[src][dst] and is_valid_edge(new_matrix, (src, dst)):
                    nbhd = add_to_nbhd(new_matrix, new_ops, nbhd)

        random.shuffle(nbhd)
        return nbhd

    def get_loss_fn(self) -> Callable:
        return F.cross_entropy

    def get_type(self) -> str:
        return "nasbench101"

    def forward_before_global_avg_pool(self, x: torch.Tensor) -> list:
        outputs = []

        def hook_fn(module, input_t, output_t):
            # print(f'Input tensor shape: {input_t[0].shape}')
            # print(f'Output tensor shape: {output_t.shape}')
            outputs.append(output_t)

        model = self.edges[1, 2]['op'].model
        model.layers[-1].register_forward_hook(hook_fn)

        self.forward(x, None)

        assert len(outputs) == 1
        return outputs[0]


def get_utilized(matrix):
    # return the sets of utilized edges and nodes
    # first, compute all paths
    n = np.shape(matrix)[0]
    sub_paths = []
    for j in range(0, n):
        sub_paths.append([[(0, j)]]) if matrix[0][j] else sub_paths.append([])

    # create paths sequentially
    for i in range(1, n - 1):
        for j in range(1, n):
            if matrix[i][j]:
                for sub_path in sub_paths[i]:
                    sub_paths[j].append([*sub_path, (i, j)])
    paths = sub_paths[-1]

    utilized_edges = []
    for path in paths:
        for edge in path:
            if edge not in utilized_edges:
                utilized_edges.append(edge)

    utilized_nodes = []
    for i in range(NUM_VERTICES):
        for edge in utilized_edges:
            if i in edge and i not in utilized_nodes:
                utilized_nodes.append(i)

    return utilized_edges, utilized_nodes


def is_valid_vertex(matrix: np.ndarray, vertex: int) -> bool:
    edges, nodes = get_utilized(matrix)
    return vertex in nodes


def is_valid_edge(matrix: np.ndarray, edge: tuple) -> bool:
    edges, nodes = get_utilized(matrix)
    return edge in edges


if __name__ == '__main__':
    import keras
    import tensorflow as tf
    dataset_api = get_dataset_api('nasbench101', None)
    weight_path = '../../../examples/gin_conv_batch_filterTrue_a1_size95500_r1_m64_b256_dropout0.2_lr0.001_mlp(64, 64, 64, 64)_0'
    dataset_api['surrogate'] = keras.models.load_model(weight_path, custom_objects={
        'weighted_mse': tf.keras.losses.MeanSquaredError()})
    search_space = NasBench101SearchSpace(model_type=MODEL_TYPE.ENSEMBLE)

    for i in range(1):
        graph = search_space.clone()
        graph.sample_random_architecture(dataset_api=dataset_api)

        graph_hash = graph.get_hash()
        print(graph_hash)

        print(convert_tuple_to_spec(graph_hash))
