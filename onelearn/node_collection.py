# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

import numpy as np
from numba import jitclass
from numba.types import float32, boolean, uint32, uint8

from .utils import resize_array


# TODO: declare methods are njitted functions
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
    ("counts", uint32[:, ::1]),
    # Is the range of the node is memorized ?
    ("memorized", boolean[::1]),
    # Minimum range of the data points in the node
    ("memory_range_min", float32[:, ::1]),
    # Maximum range of the data points in the node
    ("memory_range_max", float32[:, ::1]),
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
    # Number of nodes allocated in memory
    ("n_nodes_reserved", uint32),
    ("n_nodes_computed", uint32),
]


@jitclass(spec_node_collection)
class NodeCollection(object):
    def __init__(self, n_features, n_classes, n_nodes_reserved):
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

    def reserve_nodes(self, n_nodes_reserved):
        n_nodes = self.n_nodes
        if n_nodes_reserved > self.n_nodes_reserved:
            self.index = resize_array(self.index, n_nodes, n_nodes_reserved)
            # By default, a node is a leaf when newly created
            self.is_leaf = resize_array(self.is_leaf, n_nodes, n_nodes_reserved, True)
            self.depth = resize_array(self.depth, n_nodes, n_nodes_reserved)
            self.n_samples = resize_array(self.n_samples, n_nodes, n_nodes_reserved)
            self.parent = resize_array(self.parent, n_nodes, n_nodes_reserved)
            self.left = resize_array(self.left, n_nodes, n_nodes_reserved)
            self.right = resize_array(self.right, n_nodes, n_nodes_reserved)
            self.feature = resize_array(self.feature, n_nodes, n_nodes_reserved)
            self.y_t = resize_array(self.y_t, n_nodes, n_nodes_reserved)
            self.weight = resize_array(self.weight, n_nodes, n_nodes_reserved)
            self.log_weight_tree = resize_array(
                self.log_weight_tree, n_nodes, n_nodes_reserved
            )
            self.threshold = resize_array(self.threshold, n_nodes, n_nodes_reserved)
            self.time = resize_array(self.time, n_nodes, n_nodes_reserved)
            self.counts = resize_array(self.counts, n_nodes, n_nodes_reserved)
            self.memorized = resize_array(self.memorized, n_nodes, n_nodes_reserved)
            self.memory_range_min = resize_array(
                self.memory_range_min, n_nodes, n_nodes_reserved
            )
            self.memory_range_max = resize_array(
                self.memory_range_max, n_nodes, n_nodes_reserved
            )

        self.n_nodes_reserved = n_nodes_reserved

    def add_node(self, parent, time):
        """
        Add a node with specified parent and creation time. The other fields
        will be provided later on

        Returns
        -------
        output : `int` index of the added node
        """
        if self.n_nodes >= self.n_nodes_reserved:
            raise RuntimeError("self.n_nodes >= self.n_nodes_reserved")
        if self.n_nodes >= self.index.shape[0]:
            raise RuntimeError("self.n_nodes >= self.index.shape[0]")

        node_index = self.n_nodes
        self.index[node_index] = node_index
        self.parent[node_index] = parent
        self.time[node_index] = time
        self.n_nodes += 1
        # This is a new node, so it's necessary computed for now
        self.n_nodes_computed += 1
        return self.n_nodes - 1

    def copy_node(self, first, second):
        # We must NOT copy the index
        self.is_leaf[second] = self.is_leaf[first]
        self.depth[second] = self.depth[first]
        self.n_samples[second] = self.n_samples[first]
        self.parent[second] = self.parent[first]
        self.left[second] = self.left[first]
        self.right[second] = self.right[first]
        self.feature[second] = self.feature[first]
        self.y_t[second] = self.y_t[first]
        self.weight[second] = self.weight[first]
        self.log_weight_tree[second] = self.log_weight_tree[first]
        self.threshold[second] = self.threshold[first]
        self.time[second] = self.time[first]
        self.counts[second, :] = self.counts[first, :]
        self.memorized[second] = self.memorized[first]
        self.memory_range_min[second, :] = self.memory_range_min[first, :]
        self.memory_range_max[second, :] = self.memory_range_max[first, :]

    def print(self):
        n_nodes = self.n_nodes
        n_nodes_reserved = self.n_nodes_reserved
        print("n_nodes_reserved:", n_nodes_reserved, " n_nodes: ", n_nodes)
        print("index: ", self.index[:n_nodes])
        print("parent: ", self.parent[:n_nodes])
        print("time: ", self.time[:n_nodes])
