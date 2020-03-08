# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

import numpy as np
from numba import jitclass
from numba.types import float32, boolean, uint32, string

from .node_collection import NodeCollection
from .sample import SamplesCollection

spec_tree_classifier = [
    ("n_classes", uint32),
    ("n_features", uint32),
    ("step", float32),
    ("loss", string),
    ("use_aggregation", boolean),
    ("dirichlet", float32),
    ("split_pure", boolean),
    ("samples", SamplesCollection.class_type.instance_type),
    ("nodes", NodeCollection.class_type.instance_type),
    ("intensities", float32[::1]),
    ("iteration", uint32),
]

# TODO: write all the docstrings


@jitclass(spec_tree_classifier)
class TreeClassifier(object):
    def __init__(
        self,
        n_classes,
        n_features,
        step,
        loss,
        use_aggregation,
        dirichlet,
        split_pure,
        samples,
    ):
        self.n_classes = n_classes
        self.n_features = n_features
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.dirichlet = dirichlet
        self.split_pure = split_pure
        self.samples = samples

        # TODO: something more clever than this ?
        default_reserve = 512
        self.nodes = NodeCollection(n_features, n_classes, default_reserve)
        self.intensities = np.empty(n_features, dtype=float32)
        self.iteration = 0
        self.add_root()

    def add_root(self):
        self.add_node(0, 0.0)

    def add_node(self, parent, time):
        return self.nodes.add_node(parent, time)

    def print(self):
        print("Hello from jitclass")
        print(self.nodes.n_nodes_reserved)
        print(self.nodes.n_nodes)
