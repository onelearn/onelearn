# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

from math import exp

import numpy as np
from numba import njit
from numba.types import float32, boolean, uint32, void
from numpy.random import uniform

from .node_methods import (
    node_get_child,
    node_compute_split_time,
    node_predict,
    node_range,
    node_update_downwards,
    node_split,
    node_update_depth,
    node_update_weight_tree,
)
from .tree import TreeClassifier
from .utils import sample_discrete


# TODO: an overall task is to minimize the O(#n_features) complexity: pass few
#  times over the features
# TODO: write all the docstrings


@njit(uint32(TreeClassifier.class_type.instance_type, uint32))
def tree_go_downwards(tree, idx_sample):
    # We update the nodes along the path which leads to the leaf containing
    # x_t. For each node on the path, we consider the possibility of
    # splitting it, following the Mondrian process definition.
    # Index of the root is 0
    idx_current_node = 0
    x_t = tree.samples.features[idx_sample]

    if tree.iteration == 0:
        # If it's the first iteration, we just put x_t in the range of root
        node_update_downwards(tree, idx_current_node, idx_sample, False)
        return idx_current_node
    else:
        while True:
            # If it's not the first iteration (otherwise the current node
            # is root with no range), we consider the possibility of a split
            split_time = node_compute_split_time(tree, idx_current_node, idx_sample)
            if split_time > 0:
                # We split the current node: because the current node is a
                # leaf, or because we add a new node along the path
                # We normalize the range extensions to get probabilities
                # TODO: faster than this ?
                tree.intensities /= tree.intensities.sum()
                # Sample the feature at random with with a probability
                # proportional to the range extensions
                feature = sample_discrete(tree.intensities)
                x_tf = x_t[feature]
                # Is it a right extension of the node ?
                range_min, range_max = node_range(tree, idx_current_node, feature)
                is_right_extension = x_tf > range_max
                if is_right_extension:
                    threshold = uniform(range_max, x_tf)
                else:
                    threshold = uniform(x_tf, range_min)

                node_split(
                    tree,
                    idx_current_node,
                    split_time,
                    threshold,
                    feature,
                    is_right_extension,
                )

                # Update the current node
                node_update_downwards(tree, idx_current_node, idx_sample, True)

                left = tree.nodes.left[idx_current_node]
                right = tree.nodes.right[idx_current_node]
                depth = tree.nodes.depth[idx_current_node]

                # Now, get the next node
                if is_right_extension:
                    idx_current_node = right
                else:
                    idx_current_node = left

                node_update_depth(tree, left, depth)
                node_update_depth(tree, right, depth)

                # This is the leaf containing the sample point (we've just
                # splitted the current node with the data point)
                leaf = idx_current_node
                node_update_downwards(tree, leaf, idx_sample, False)
                return leaf
            else:
                # There is no split, so we just update the node and go to
                # the next one
                node_update_downwards(tree, idx_current_node, idx_sample, True)
                is_leaf = tree.nodes.is_leaf[idx_current_node]
                if is_leaf:
                    return idx_current_node
                else:
                    idx_current_node = node_get_child(tree, idx_current_node, x_t)


@njit(void(TreeClassifier.class_type.instance_type, uint32))
def tree_go_upwards(tree, leaf):
    idx_current_node = leaf
    if tree.iteration >= 1:
        while True:
            node_update_weight_tree(tree, idx_current_node)
            if idx_current_node == 0:
                # We arrived at the root
                break
            # Note that the root node is updated as well
            # We go up to the root in the tree
            idx_current_node = tree.nodes.parent[idx_current_node]


@njit(void(TreeClassifier.class_type.instance_type, uint32))
def tree_partial_fit(tree, idx_sample):
    leaf = tree_go_downwards(tree, idx_sample)
    if tree.use_aggregation:
        tree_go_upwards(tree, leaf)
    tree.iteration += 1


@njit(uint32(TreeClassifier.class_type.instance_type, float32[::1]))
def tree_get_leaf(tree, x_t):
    # Find the index of the leaf that contains the sample. Start at the root.
    # Index of the root is 0
    node = 0
    is_leaf = False
    while not is_leaf:
        is_leaf = tree.nodes.is_leaf[node]
        if not is_leaf:
            feature = tree.nodes.feature[node]
            threshold = tree.nodes.threshold[node]
            if x_t[feature] <= threshold:
                node = tree.nodes.left[node]
            else:
                node = tree.nodes.right[node]
    return node


@njit(
    void(TreeClassifier.class_type.instance_type, float32[::1], float32[::1], boolean)
)
def tree_predict(tree, x_t, scores, use_aggregation):
    leaf = tree_get_leaf(tree, x_t)
    if not use_aggregation:
        node_predict(tree, leaf, scores)
        return
    current = leaf
    pred_new = np.empty(tree.n_classes, float32)
    while True:
        if tree.nodes.is_leaf[current]:
            node_predict(tree, current, scores)
        else:
            weight = tree.nodes.weight[current]
            log_weight_tree = tree.nodes.log_weight_tree[current]
            w = exp(weight - log_weight_tree)
            # Get the predictions of the current node
            node_predict(tree, current, pred_new)
            for c in range(tree.n_classes):
                scores[c] = 0.5 * w * pred_new[c] + (1 - 0.5 * w) * scores[c]
        # Root must be update as well
        if current == 0:
            break
        # And now we go up
        current = tree.nodes.parent[current]
