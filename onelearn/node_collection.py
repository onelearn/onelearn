# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import numpy as np
from numba import njit
from numba.experimental import jitclass
from .types import void, float32, boolean, uint32, uint8, int32, get_array_2d_type
from .utils import resize_array, get_type


# Specification of base node attributes
spec_nodes = [
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
    # TODO: is it used ?
    # Logarithm of the aggregation weight for the node
    ("weight", float32[::1]),
    # Logarithm of the aggregation weight for the sub-tree starting at this node
    ("log_weight_tree", float32[::1]),
    # Threshold used for the split
    ("threshold", float32[::1]),
    # Time of creation of the node
    ("time", float32[::1]),
    # Minimum range of the data points in the node
    ("memory_range_min", get_array_2d_type(float32)),
    # Maximum range of the data points in the node
    ("memory_range_max", get_array_2d_type(float32)),
    # Number of features
    ("n_features", uint32),
    # Number of nodes actually used
    ("n_nodes", uint32),
    # For how many nodes do we allocate nodes in advance ?
    ("n_samples_increment", uint32),
    # Number of nodes currently allocated in memory
    ("n_nodes_capacity", uint32),
]


# TODO: group together arrays of the same type for faster computations
#  and copies ?


spec_nodes_classifier = spec_nodes + [
    # Counts the number of sample seen in each class
    ("counts", get_array_2d_type(uint32)),
    # Number of classes
    ("n_classes", uint32),
]


@jitclass(spec_nodes_classifier)
class NodesClassifier(object):
    """A collection of nodes for classification.

    Attributes
    ----------
    n_features : :obj:`int`
        Number of features used during training.

    n_nodes : :obj:`int`
        Number of nodes saved in the collection.

    n_samples_increment : :obj:`int`
        The minimum amount of memory which is pre-allocated each time extra memory is
        required for new nodes.

    n_nodes_capacity : :obj:`int`
        Number of nodes that can be currently saved in the object.

    """

    def __init__(
        self, n_features, n_classes, n_samples_increment, n_nodes, n_nodes_capacity
    ):
        """Instantiates a `NodesClassifier` instance.

        Parameters
        ----------
        n_features : :obj:`int`
            Number of features used during training.

        n_classes : :obj:`int`
            Number of expected classes in the labels.

        n_samples_increment : :obj:`int`
            The minimum amount of memory which is pre-allocated each time extra memory
            is required for new nodes.
        """
        init_nodes(self, n_features, n_samples_increment, n_nodes, n_nodes_capacity)
        init_nodes_classifier(self, n_classes)


spec_nodes_regressor = spec_nodes + [
    # Current mean of the labels in the node
    ("mean", float32[::1]),
]


@jitclass(spec_nodes_regressor)
class NodesRegressor(object):
    """A collection of nodes for regression.

    Attributes
    ----------
    n_features : :obj:`int`
        Number of features used during training.

    n_nodes : :obj:`int`
        Number of nodes saved in the collection.

    n_samples_increment : :obj:`int`
        The minimum amount of memory which is pre-allocated each time extra memory is
        required for new nodes.

    n_nodes_capacity : :obj:`int`
        Number of nodes that can be currently saved in the object.

    """

    def __init__(self, n_features, n_samples_increment, n_nodes, n_nodes_capacity):
        """Instantiates a `NodesClassifier` instance.

        Parameters
        ----------
        n_features : :obj:`int`
            Number of features used during training.

        n_classes : :obj:`int`
            Number of expected classes in the labels.

        n_samples_increment : :obj:`int`
            The minimum amount of memory which is pre-allocated each time extra memory
            is required for new nodes.

        """
        init_nodes(self, n_features, n_samples_increment, n_nodes, n_nodes_capacity)
        init_nodes_regressor(self)


@njit(
    [
        void(get_type(NodesClassifier), uint32, uint32),
        void(get_type(NodesRegressor), uint32, uint32),
    ]
)
def init_nodes_arrays(nodes, n_nodes_capacity, n_features):
    """Initializes the nodes arrays given their capacity

    Parameters
    ----------
    nodes : :obj:`NodesClassifier` or :obj:`NodesRegressor`
        Object to be initialized

    n_nodes_capacity : :obj:`int`
        Desired nodes capacity

    n_features : :obj:`int`
        Number of features used during training.
    """
    nodes.index = np.zeros(n_nodes_capacity, dtype=uint32)
    nodes.is_leaf = np.ones(n_nodes_capacity, dtype=boolean)
    nodes.depth = np.zeros(n_nodes_capacity, dtype=uint8)
    nodes.n_samples = np.zeros(n_nodes_capacity, dtype=uint32)
    nodes.parent = np.zeros(n_nodes_capacity, dtype=uint32)
    nodes.left = np.zeros(n_nodes_capacity, dtype=uint32)
    nodes.right = np.zeros(n_nodes_capacity, dtype=uint32)
    nodes.feature = np.zeros(n_nodes_capacity, dtype=uint32)
    nodes.weight = np.zeros(n_nodes_capacity, dtype=float32)
    nodes.log_weight_tree = np.zeros(n_nodes_capacity, dtype=float32)
    nodes.threshold = np.zeros(n_nodes_capacity, dtype=float32)
    nodes.time = np.zeros(n_nodes_capacity, dtype=float32)
    nodes.memory_range_min = np.zeros((n_nodes_capacity, n_features), dtype=float32)
    nodes.memory_range_max = np.zeros((n_nodes_capacity, n_features), dtype=float32)


@njit(
    [
        void(get_type(NodesClassifier), uint32, uint32, uint32, uint32),
        void(get_type(NodesRegressor), uint32, uint32, uint32, uint32),
    ]
)
def init_nodes(nodes, n_features, n_samples_increment, n_nodes, n_nodes_capacity):
    """Initializes a `Nodes` instance.

    Parameters
    ----------
    nodes : :obj:`NodesClassifier` or :obj:`NodesRegressor`
        Object to be initialized

    n_features : :obj:`int`
        Number of features used during training.

    n_samples_increment : :obj:`int`
        The minimum amount of memory which is pre-allocated each time extra memory is
        required for new nodes.

    n_nodes : :obj:`int`
        Blabla

    n_nodes_capacity : :obj:`int`
        Initial required node capacity. If 0, we use 2 * n_samples_increment + 1,
        otherwise we use the given value (useful for serialization).
    """

    if n_nodes_capacity == 0:
        # One for root + and twice the number of samples
        n_nodes_capacity = 2 * n_samples_increment + 1
    nodes.n_samples_increment = n_samples_increment
    nodes.n_features = n_features
    nodes.n_nodes_capacity = n_nodes_capacity
    nodes.n_nodes = n_nodes
    # Initialize node attributes
    init_nodes_arrays(nodes, n_nodes_capacity, n_features)


@njit(void(get_type(NodesClassifier), uint32))
def init_nodes_classifier(nodes, n_classes):
    """Initializes a `NodesClassifier` instance.

    Parameters
    ----------
    n_classes : :obj:`int`
        Number of expected classes in the labels.

    """
    nodes.counts = np.zeros((nodes.n_nodes_capacity, n_classes), dtype=uint32)
    nodes.n_classes = n_classes


@njit(void(get_type(NodesRegressor)))
def init_nodes_regressor(nodes):
    """Initializes a `NodesRegressor` instance.

    """
    nodes.mean = np.zeros(nodes.n_nodes_capacity, dtype=float32)


@njit(
    [void(get_type(NodesClassifier)), void(get_type(NodesRegressor)),]
)
def reserve_nodes(nodes):
    """Reserves memory for nodes.

    Parameters
    ----------
    nodes : :obj:`Nodes`
        The collection of nodes.
    """
    n_nodes_capacity = nodes.n_nodes_capacity + 2 * nodes.n_samples_increment + 1
    n_nodes = nodes.n_nodes
    # TODO: why is this test useful ?
    if n_nodes_capacity > nodes.n_nodes_capacity:
        nodes.index = resize_array(nodes.index, n_nodes, n_nodes_capacity)
        # By default, a node is a leaf when newly created
        nodes.is_leaf = resize_array(nodes.is_leaf, n_nodes, n_nodes_capacity, fill=1)
        nodes.depth = resize_array(nodes.depth, n_nodes, n_nodes_capacity)
        nodes.n_samples = resize_array(nodes.n_samples, n_nodes, n_nodes_capacity)
        nodes.parent = resize_array(nodes.parent, n_nodes, n_nodes_capacity)
        nodes.left = resize_array(nodes.left, n_nodes, n_nodes_capacity)
        nodes.right = resize_array(nodes.right, n_nodes, n_nodes_capacity)
        nodes.feature = resize_array(nodes.feature, n_nodes, n_nodes_capacity)
        nodes.weight = resize_array(nodes.weight, n_nodes, n_nodes_capacity)
        nodes.log_weight_tree = resize_array(
            nodes.log_weight_tree, n_nodes, n_nodes_capacity
        )
        nodes.threshold = resize_array(nodes.threshold, n_nodes, n_nodes_capacity)
        nodes.time = resize_array(nodes.time, n_nodes, n_nodes_capacity)

        nodes.memory_range_min = resize_array(
            nodes.memory_range_min, n_nodes, n_nodes_capacity
        )
        nodes.memory_range_max = resize_array(
            nodes.memory_range_max, n_nodes, n_nodes_capacity
        )

    nodes.n_nodes_capacity = n_nodes_capacity


@njit(void(get_type(NodesClassifier)))
def reserve_nodes_classifier(nodes):
    """Reserves memory for classifier nodes.

    Parameters
    ----------
    nodes : :obj:`NodesClassifier`
        The collection of classifier nodes.

    """
    reserve_nodes(nodes)
    nodes.counts = resize_array(nodes.counts, nodes.n_nodes, nodes.n_nodes_capacity)


@njit(void(get_type(NodesRegressor)))
def reserve_nodes_regressor(nodes):
    """Reserves memory for regressor nodes.

    Parameters
    ----------
    nodes : :obj:`NodesRegressor`
        The collection of regressor nodes.

    """
    reserve_nodes(nodes)
    nodes.mean = resize_array(nodes.mean, nodes.n_nodes, nodes.n_nodes_capacity)


@njit(
    [
        uint32(get_type(NodesClassifier), uint32, float32),
        uint32(get_type(NodesRegressor), uint32, float32),
    ]
)
def add_node(nodes, parent, time):
    """Adds a node with specified parent and creation time. This functions assumes that
    a node has been already allocated by "child" functions `add_node_classifier` and
    `add_node_regressor`.

    Parameters
    ----------
    nodes : :obj:`Nodes`
        The collection of nodes.

    parent : :obj:`int`
        The index of the parent of the new node.

    time : :obj:`float`
        The creation time of the new node.

    Returns
    -------
    output : `int`
        Index of the new node.

    """
    node_index = nodes.n_nodes
    nodes.index[node_index] = node_index
    nodes.parent[node_index] = parent
    nodes.time[node_index] = time
    nodes.n_nodes += 1
    return nodes.n_nodes - 1


@njit(uint32(get_type(NodesClassifier), uint32, float32))
def add_node_classifier(nodes, parent, time):
    """Adds a node with specified parent and creation time.

    Parameters
    ----------
    nodes : :obj:`Nodes`
        The collection of nodes.

    parent : :obj:`int`
        The index of the parent of the new node.

    time : :obj:`float`
        The creation time of the new node.

    Returns
    -------
    output : `int`
        Index of the new node.

    """
    if nodes.n_nodes >= nodes.n_nodes_capacity:
        # We don't have memory for this extra node, so let's create some
        reserve_nodes_classifier(nodes)

    return add_node(nodes, parent, time)


@njit(uint32(get_type(NodesRegressor), uint32, float32))
def add_node_regressor(nodes, parent, time):
    """Adds a node with specified parent and creation time.

    Parameters
    ----------
    nodes : :obj:`Nodes`
        The collection of nodes.

    parent : :obj:`int`
        The index of the parent of the new node.

    time : :obj:`float`
        The creation time of the new node.

    Returns
    -------
    output : `int`
        Index of the new node.

    """
    if nodes.n_nodes >= nodes.n_nodes_capacity:
        # We don't have memory for this extra node, so let's create some
        reserve_nodes_regressor(nodes)

    return add_node(nodes, parent, time)


@njit(
    [
        void(get_type(NodesClassifier), uint32, uint32),
        void(get_type(NodesRegressor), uint32, uint32),
    ]
)
def copy_node(nodes, first, second):
    """Copies the node at index ``first`` into the node at index ``second``.

    Parameters
    ----------
    nodes : :obj:`Nodes`
        The collection of nodes.

    first : :obj:`int`
        The index of the node to be copied in ``second``.

    second : :obj:`int`
        The index of the node containing the copy of ``first``.

    """
    # We must NOT copy the index
    nodes.is_leaf[second] = nodes.is_leaf[first]
    nodes.depth[second] = nodes.depth[first]
    nodes.n_samples[second] = nodes.n_samples[first]
    nodes.parent[second] = nodes.parent[first]
    nodes.left[second] = nodes.left[first]
    nodes.right[second] = nodes.right[first]
    nodes.feature[second] = nodes.feature[first]
    nodes.weight[second] = nodes.weight[first]
    nodes.log_weight_tree[second] = nodes.log_weight_tree[first]
    nodes.threshold[second] = nodes.threshold[first]
    nodes.time[second] = nodes.time[first]
    nodes.memory_range_min[second, :] = nodes.memory_range_min[first, :]
    nodes.memory_range_max[second, :] = nodes.memory_range_max[first, :]


@njit(void(get_type(NodesClassifier), uint32, uint32))
def copy_node_classifier(nodes, first, second):
    """Copies the node at index `first` into the node at index `second`.

    Parameters
    ----------
    nodes : :obj:`NodesClassifier`
        The collection of nodes

    first : :obj:`int`
        The index of the node to be copied in ``second``

    second : :obj:`int`
        The index of the node containing the copy of ``first``

    """
    copy_node(nodes, first, second)
    nodes.counts[second, :] = nodes.counts[first, :]


@njit(void(get_type(NodesRegressor), uint32, uint32))
def copy_node_regressor(nodes, first, second):
    """Copies the node at index `first` into the node at index `second`.

    Parameters
    ----------
    nodes : :obj:`NodesRegressor`
        The collection of nodes

    first : :obj:`int`
        The index of the node to be copied in ``second``

    second : :obj:`int`
        The index of the node containing the copy of ``first``

    """
    copy_node(nodes, first, second)
    nodes.mean[second] = nodes.mean[first]


def nodes_classifier_to_dict(nodes):
    d = {}
    for key, _ in spec_nodes_classifier:
        d[key] = getattr(nodes, key)
    return d


def nodes_regressor_to_dict(nodes):
    d = {}
    for key, dtype in spec_nodes_regressor:
        d[key] = getattr(nodes, key)
    return d


def dict_to_nodes(nodes, nodes_dict):
    nodes.index[:] = nodes_dict["index"]
    nodes.is_leaf[:] = nodes_dict["is_leaf"]
    nodes.depth[:] = nodes_dict["depth"]
    nodes.n_samples[:] = nodes_dict["n_samples"]
    nodes.parent[:] = nodes_dict["parent"]
    nodes.left[:] = nodes_dict["left"]
    nodes.right[:] = nodes_dict["right"]
    nodes.feature[:] = nodes_dict["feature"]
    nodes.weight[:] = nodes_dict["weight"]
    nodes.log_weight_tree[:] = nodes_dict["log_weight_tree"]
    nodes.threshold[:] = nodes_dict["threshold"]
    nodes.time[:] = nodes_dict["time"]
    nodes.memory_range_min[:] = nodes_dict["memory_range_min"]
    nodes.memory_range_max[:] = nodes_dict["memory_range_max"]


def dict_to_nodes_classifier(nodes, nodes_dict):
    dict_to_nodes(nodes, nodes_dict)
    nodes.counts[:] = nodes_dict["counts"]


def dict_to_nodes_regressor(nodes, nodes_dict):
    dict_to_nodes(nodes, nodes_dict)
    nodes.mean[:] = nodes_dict["mean"]
