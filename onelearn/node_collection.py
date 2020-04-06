# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import numpy as np
from numba import jitclass, njit
from .types import void, float32, boolean, uint32, uint8, get_array_2d_type
from .utils import resize_array, get_type


# TODO: write the docstrings

spec_node_collection = [
    # The index of the node in the tree
    ("index", uint32[::1]),
    # Is the node a leaf ?
    ("is_leaf", boolean[::1]),
    # Depth of the node in the tree
    ("depth", uint8[::1]),
    # Number of samples in the node
    ("n_samples", uint32[::1]),
    # Index of the parent
    ("parent", uint32[::1]),
    # Index of the left child
    ("left", uint32[::1]),
    # Index of the right child
    ("right", uint32[::1]),
    # Index of the feature used for the split
    ("feature", uint32[::1]),
    # The label of the sample saved in the node
    ("y_t", float32[::1]),
    # Logarithm of the aggregation weight for the node
    ("weight", float32[::1]),
    # Logarithm of the aggregation weight for the sub-tree starting at this node
    ("log_weight_tree", float32[::1]),
    # Threshold used for the split
    ("threshold", float32[::1]),
    # Time of creation of the node
    ("time", float32[::1]),
    # Counts the number of sample seen in each class
    ("counts", get_array_2d_type(uint32)),
    # Is the range of the node is memorized ?
    ("memorized", boolean[::1]),
    # Minimum range of the data points in the node
    ("memory_range_min", get_array_2d_type(float32)),
    # Maximum range of the data points in the node
    ("memory_range_max", get_array_2d_type(float32)),
    # List of the samples contained in the range of the node (this allows to
    # compute the range whenever the range memory isn't used)
    # TODO: this is going to be a problem, we don't know in advance how many points end up in the node
    # ('samples', typed.List(uint32)),
    # Number of features
    ("n_features", uint32),
    # Number of classes
    ("n_classes", uint32),
    # Number of nodes actually used
    ("n_nodes", uint32),
    # For how many samples do we allocate nodes in advance ?
    ("n_samples_increment", uint32),
    # Number of nodes currently allocated in memory
    ("n_nodes_reserved", uint32),
    ("n_nodes_computed", uint32),
]


@jitclass(spec_node_collection)
class NodeCollection(object):
    def __init__(self, n_features, n_classes, n_samples_increment):
        # One for root + and twice the number of samples
        n_nodes_reserved = 2 * n_samples_increment + 1
        self.n_samples_increment = n_samples_increment
        self.n_features = n_features
        self.n_classes = n_classes
        # TODO: group together arrays of the same type for faster computations
        #  and copies ?
        self.index = np.zeros(n_nodes_reserved, dtype=uint32)
        self.is_leaf = np.ones(n_nodes_reserved, dtype=boolean)
        self.depth = np.zeros(n_nodes_reserved, dtype=uint8)
        self.n_samples = np.zeros(n_nodes_reserved, dtype=uint32)
        self.parent = np.zeros(n_nodes_reserved, dtype=uint32)
        self.left = np.zeros(n_nodes_reserved, dtype=uint32)
        self.right = np.zeros(n_nodes_reserved, dtype=uint32)
        self.feature = np.zeros(n_nodes_reserved, dtype=uint32)
        self.y_t = np.zeros(n_nodes_reserved, dtype=float32)
        self.weight = np.zeros(n_nodes_reserved, dtype=float32)
        self.log_weight_tree = np.zeros(n_nodes_reserved, dtype=float32)
        self.threshold = np.zeros(n_nodes_reserved, dtype=float32)
        self.time = np.zeros(n_nodes_reserved, dtype=float32)
        self.counts = np.zeros((n_nodes_reserved, n_classes), dtype=uint32)
        self.memorized = np.zeros(n_nodes_reserved, dtype=boolean)
        self.memory_range_min = np.zeros((n_nodes_reserved, n_features), dtype=float32)
        self.memory_range_max = np.zeros((n_nodes_reserved, n_features), dtype=float32)
        self.n_nodes_reserved = n_nodes_reserved
        self.n_nodes = 0
        self.n_nodes_computed = 0


@njit(void(get_type(NodeCollection)))
def reserve_nodes(node_collection):
    n_nodes_reserved = (
        node_collection.n_nodes_reserved + node_collection.n_samples_increment
    )
    n_nodes = node_collection.n_nodes
    if n_nodes_reserved > node_collection.n_nodes_reserved:
        node_collection.index = resize_array(
            node_collection.index, n_nodes, n_nodes_reserved
        )
        # By default, a node is a leaf when newly created
        node_collection.is_leaf = resize_array(
            node_collection.is_leaf, n_nodes, n_nodes_reserved, fill=1
        )
        node_collection.depth = resize_array(
            node_collection.depth, n_nodes, n_nodes_reserved
        )
        node_collection.n_samples = resize_array(
            node_collection.n_samples, n_nodes, n_nodes_reserved
        )
        node_collection.parent = resize_array(
            node_collection.parent, n_nodes, n_nodes_reserved
        )
        node_collection.left = resize_array(
            node_collection.left, n_nodes, n_nodes_reserved
        )
        node_collection.right = resize_array(
            node_collection.right, n_nodes, n_nodes_reserved
        )
        node_collection.feature = resize_array(
            node_collection.feature, n_nodes, n_nodes_reserved
        )
        node_collection.y_t = resize_array(
            node_collection.y_t, n_nodes, n_nodes_reserved
        )
        node_collection.weight = resize_array(
            node_collection.weight, n_nodes, n_nodes_reserved
        )
        node_collection.log_weight_tree = resize_array(
            node_collection.log_weight_tree, n_nodes, n_nodes_reserved
        )
        node_collection.threshold = resize_array(
            node_collection.threshold, n_nodes, n_nodes_reserved
        )
        node_collection.time = resize_array(
            node_collection.time, n_nodes, n_nodes_reserved
        )
        node_collection.counts = resize_array(
            node_collection.counts, n_nodes, n_nodes_reserved
        )
        node_collection.memorized = resize_array(
            node_collection.memorized, n_nodes, n_nodes_reserved
        )
        node_collection.memory_range_min = resize_array(
            node_collection.memory_range_min, n_nodes, n_nodes_reserved
        )
        node_collection.memory_range_max = resize_array(
            node_collection.memory_range_max, n_nodes, n_nodes_reserved
        )

    node_collection.n_nodes_reserved = n_nodes_reserved


@njit(uint32(get_type(NodeCollection), uint32, float32))
def add_node(node_collection, parent, time):
    """
    Add a node with specified parent and creation time. The other fields
    will be provided later on

    Returns
    -------
    output : `int` index of the added node
    """
    if node_collection.n_nodes >= node_collection.n_nodes_reserved:
        # We don't have memory for this extra node, so let's create some
        reserve_nodes(node_collection)

    node_index = node_collection.n_nodes
    node_collection.index[node_index] = node_index
    node_collection.parent[node_index] = parent
    node_collection.time[node_index] = time
    node_collection.n_nodes += 1
    # This is a new node, so it's necessary computed for now
    node_collection.n_nodes_computed += 1
    return node_collection.n_nodes - 1


@njit(void(get_type(NodeCollection), uint32, uint32))
def copy_node(node_collection, first, second):
    # We must NOT copy the index
    node_collection.is_leaf[second] = node_collection.is_leaf[first]
    node_collection.depth[second] = node_collection.depth[first]
    node_collection.n_samples[second] = node_collection.n_samples[first]
    node_collection.parent[second] = node_collection.parent[first]
    node_collection.left[second] = node_collection.left[first]
    node_collection.right[second] = node_collection.right[first]
    node_collection.feature[second] = node_collection.feature[first]
    node_collection.y_t[second] = node_collection.y_t[first]
    node_collection.weight[second] = node_collection.weight[first]
    node_collection.log_weight_tree[second] = node_collection.log_weight_tree[first]
    node_collection.threshold[second] = node_collection.threshold[first]
    node_collection.time[second] = node_collection.time[first]
    node_collection.counts[second, :] = node_collection.counts[first, :]
    node_collection.memorized[second] = node_collection.memorized[first]
    node_collection.memory_range_min[second, :] = node_collection.memory_range_min[
        first, :
    ]
    node_collection.memory_range_max[second, :] = node_collection.memory_range_max[
        first, :
    ]
