# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

import numpy as np
from numba import jitclass
from .types import float32, boolean, uint32, string

from .node_collection import (
    NodesClassifier,
    NodesRegressor,
    add_node_classifier,
    add_node_regressor,
)
from .sample import SamplesCollection
from .utils import get_type

spec_tree = [
    ("n_features", uint32),
    ("step", float32),
    ("loss", string),
    ("use_aggregation", boolean),
    ("split_pure", boolean),
    ("samples", get_type(SamplesCollection)),
    ("intensities", float32[::1]),
    ("iteration", uint32),
]

spec_tree_classifier = spec_tree + [
    ("n_classes", uint32),
    ("dirichlet", float32),
    ("nodes", get_type(NodesClassifier)),
]

spec_tree_regressor = spec_tree + [
    ("nodes", get_type(NodesRegressor)),
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
        n_samples_increment = self.samples.n_samples_increment
        self.nodes = NodesClassifier(n_features, n_classes, n_samples_increment)
        self.intensities = np.empty(n_features, dtype=float32)
        self.iteration = 0
        add_node_classifier(self.nodes, 0, 0.0)


@jitclass(spec_tree_regressor)
class TreeRegressor(object):
    def __init__(
        self, n_features, step, loss, use_aggregation, split_pure, samples,
    ):
        self.n_features = n_features
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.split_pure = split_pure
        self.samples = samples
        n_samples_increment = self.samples.n_samples_increment
        self.nodes = NodesRegressor(n_features, n_samples_increment)
        self.intensities = np.empty(n_features, dtype=float32)
        self.iteration = 0
        add_node_regressor(self.nodes, 0, 0.0)
