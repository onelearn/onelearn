# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

import numpy as np
from numba import jitclass
from numba.types import float32, boolean, uint32

from .node_collection import NodeCollection
from .sample import SamplesCollection

spec_tree_classifier = [
    ("samples", SamplesCollection.class_type.instance_type),
    ("nodes", NodeCollection.class_type.instance_type),
    ("n_features", uint32),
    ("n_classes", uint32),
    ("iteration", uint32),
    ("use_aggregation", boolean),
    ("step", float32),
    ("split_pure", boolean),
    ("intensities", float32[::1]),
    ("dirichlet", float32),
]


# TODO: write all the docstrings


@jitclass(spec_tree_classifier)
class TreeClassifier(object):
    def __init__(self, n_features, n_classes, samples):
        self.samples = samples

        # TODO: something more clever than this ?
        default_reserve = 512

        self.nodes = NodeCollection(n_features, n_classes, default_reserve)

        self.n_features = n_features
        self.n_classes = n_classes
        self.iteration = 0

        self.use_aggregation = True
        self.step = 1.0
        self.split_pure = False
        self.dirichlet = 0.5
        self.intensities = np.empty(n_features, dtype=float32)

        # We add the root
        self.add_root()

    def add_root(self):
        self.add_node(0, 0.0)

    def add_node(self, parent, time):
        return self.nodes.add_node(parent, time)

    def print(self):
        print("Hello from jitclass")
        print(self.nodes.n_nodes_reserved)
        print(self.nodes.n_nodes)
