# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

import numpy as np
from numba.experimental import jitclass
from .types import float32, boolean, uint32, string

from .node_collection import (
    NodesClassifier,
    NodesRegressor,
    add_node_classifier,
    add_node_regressor,
    nodes_classifier_to_dict,
    nodes_regressor_to_dict,
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
        iteration,
        n_nodes,
        n_nodes_capacity,
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
        self.intensities = np.empty(n_features, dtype=float32)
        if n_nodes == 0:
            self.iteration = 0
            n_nodes = 0
            n_nodes_capacity = 0
            self.nodes = NodesClassifier(
                n_features, n_classes, n_samples_increment, n_nodes, n_nodes_capacity
            )
            add_node_classifier(self.nodes, 0, 0.0)
        else:
            self.iteration = iteration
            self.nodes = NodesClassifier(
                n_features, n_classes, n_samples_increment, n_nodes, n_nodes_capacity
            )


def tree_classifier_to_dict(tree):
    d = {}
    for key, dtype in spec_tree_classifier:
        if key == "nodes":
            nodes = nodes_classifier_to_dict(tree.nodes)
            d["nodes"] = nodes
        elif key == "samples":
            # We do not save the samples here. There are saved in the forest
            # otherwise a copy is made for each tree in the pickle file
            pass
        else:
            d[key] = getattr(tree, key)
    return d


@jitclass(spec_tree_regressor)
class TreeRegressor(object):
    def __init__(
        self,
        n_features,
        step,
        loss,
        use_aggregation,
        split_pure,
        samples,
        iteration,
        n_nodes,
        n_nodes_capacity,
    ):
        self.n_features = n_features
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.split_pure = split_pure
        self.samples = samples
        n_samples_increment = self.samples.n_samples_increment
        self.intensities = np.empty(n_features, dtype=float32)
        if n_nodes == 0:
            self.iteration = 0
            n_nodes = 0
            n_nodes_capacity = 0
            self.nodes = NodesRegressor(
                n_features, n_samples_increment, n_nodes, n_nodes_capacity
            )
            add_node_regressor(self.nodes, 0, 0.0)
        else:
            self.iteration = iteration
            self.nodes = NodesRegressor(
                n_features, n_samples_increment, n_nodes, n_nodes_capacity
            )


def tree_regressor_to_dict(tree):
    d = {}
    for key, dtype in spec_tree_regressor:
        if key == "nodes":
            nodes = nodes_regressor_to_dict(tree.nodes)
            d["nodes"] = nodes
        elif key == "samples":
            # We do not save the samples here. There are saved in the forest
            # otherwise a copy is made for each tree in the pickle file
            pass
        else:
            d[key] = getattr(tree, key)
    return d


# def dict_to_tree_regressor(d):
#     n_features = d["n_features"]
#     step = d["step"]
#     loss = d["loss"]
#     use_aggregation = d["use_aggregation"]
#     split_pure = d["split_pure"]
#     samples = dict_to_samples_collection(d["samples"])
#     tree = TreeRegressor(
#         n_features, step, loss, use_aggregation, split_pure, samples, False,
#     )
#     tree.iteration = d["iteration"]
#     nodes = dict_to_nodes_regressor(d["nodes"])
#     tree.nodes = nodes
#     intensities = np.empty(n_features, dtype=float32)
#     intensities[:] = d["intensities"]
#     tree.intensities[:] = intensities
#     return tree
