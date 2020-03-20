# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
from math import log
from numpy.random import exponential
from numba import njit
from numba import types
from .types import float32, boolean, uint32, uint8, void, Tuple
from .node_collection import add_node, copy_node
from .tree import TreeClassifier
from .utils import log_sum_2_exp, get_type


# TODO: write all the docstrings


@njit(float32(get_type(TreeClassifier), uint32, uint32))
def node_score(tree, node, idx_class):
    """Computes the score of the node

    Parameters
    ----------
    tree : `TreeClassifier`
        The tree containing the node

    node : `uint32`
        The index of the node in the tree

    idx_class : `uint32`
        Class index for which we want the score

    Returns
    -------
    output : `float32`
        The log-loss of the node

    Notes
    -----
    This uses Jeffreys prior with dirichlet parameter for smoothing
    """
    nodes = tree.nodes
    count = nodes.counts[node, idx_class]
    n_samples = nodes.n_samples[node]
    n_classes = tree.n_classes
    dirichlet = tree.dirichlet
    # We use the Jeffreys prior with dirichlet parameter
    return (count + dirichlet) / (n_samples + dirichlet * n_classes)


@njit(float32(get_type(TreeClassifier), uint32, uint32))
def node_loss(tree, node, idx_sample):
    c = types.uint8(tree.samples.labels[idx_sample])
    sc = node_score(tree, node, c)
    # TODO: benchmark different logarithms
    return -log(sc)


@njit(float32(get_type(TreeClassifier), uint32, uint32))
def node_update_weight(tree, idx_node, idx_sample):
    loss_t = node_loss(tree, idx_node, idx_sample)
    if tree.use_aggregation:
        tree.nodes.weight[idx_node] -= tree.step * loss_t
    return loss_t


@njit(void(get_type(TreeClassifier), uint32, uint32))
def node_update_count(tree, idx_node, idx_sample):
    # TODO: Don't do it twice...
    c = types.uint32(tree.samples.labels[idx_sample])
    tree.nodes.counts[idx_node, c] += 1


@njit(void(get_type(TreeClassifier), uint32, uint32, boolean))
def node_update_downwards(tree, idx_node, idx_sample, do_update_weight):
    x_t = tree.samples.features[idx_sample]
    nodes = tree.nodes
    n_features = tree.n_features
    memory_range_min = nodes.memory_range_min[idx_node]
    memory_range_max = nodes.memory_range_max[idx_node]
    # If it is the first sample, we copy the features vector into the min and
    # max range
    if nodes.n_samples[idx_node] == 0:
        for j in range(n_features):
            x_tj = x_t[j]
            memory_range_min[j] = x_tj
            memory_range_max[j] = x_tj
    # Otherwise, we update the range
    else:
        for j in range(n_features):
            x_tj = x_t[j]
            if x_tj < memory_range_min[j]:
                memory_range_min[j] = x_tj
            if x_tj > memory_range_max[j]:
                memory_range_max[j] = x_tj

    # TODO: we should save the sample here and do a bunch of stuff about
    #  memorization
    # One more sample in the node
    nodes.n_samples[idx_node] += 1

    if do_update_weight:
        # TODO: Using x_t and y_t should be better...
        node_update_weight(tree, idx_node, idx_sample)

    node_update_count(tree, idx_node, idx_sample)


@njit(void(get_type(TreeClassifier), uint32))
def node_update_weight_tree(tree, idx_node):
    nodes = tree.nodes
    if nodes.is_leaf[idx_node]:
        nodes.log_weight_tree[idx_node] = nodes.weight[idx_node]
    else:
        left = nodes.left[idx_node]
        right = nodes.right[idx_node]
        weight = nodes.weight[idx_node]
        log_weight_tree = nodes.log_weight_tree
        log_weight_tree[idx_node] = log_sum_2_exp(
            weight, log_weight_tree[left] + log_weight_tree[right]
        )


@njit(void(get_type(TreeClassifier), uint32, uint8))
def node_update_depth(tree, idx_node, depth):
    depth += 1
    nodes = tree.nodes
    nodes.depth[idx_node] = depth
    if nodes.is_leaf[idx_node]:
        return
    else:
        left = nodes.left[idx_node]
        right = nodes.right[idx_node]
        node_update_depth(tree, left, depth)
        node_update_depth(tree, right, depth)


@njit(boolean(get_type(TreeClassifier), uint32, float32))
def node_is_dirac(tree, idx_node, y_t):
    c = types.uint8(y_t)
    nodes = tree.nodes
    n_samples = nodes.n_samples[idx_node]
    count = nodes.counts[idx_node, c]
    return n_samples == count


@njit(uint32(get_type(TreeClassifier), uint32, float32[::1]))
def node_get_child(tree, idx_node, x_t):
    nodes = tree.nodes
    feature = nodes.feature[idx_node]
    threshold = nodes.threshold[idx_node]
    if x_t[feature] <= threshold:
        return nodes.left[idx_node]
    else:
        return nodes.right[idx_node]


@njit(Tuple((float32, float32))(get_type(TreeClassifier), uint32, uint32))
def node_range(tree, idx_node, j):
    # TODO: do the version without memory...
    nodes = tree.nodes
    return (
        nodes.memory_range_min[idx_node, j],
        nodes.memory_range_max[idx_node, j],
    )


@njit(float32(get_type(TreeClassifier), uint32, float32[::1], float32[::1]))
def node_compute_range_extension(tree, idx_node, x_t, extensions):
    extensions_sum = 0
    for j in range(tree.n_features):
        x_tj = x_t[j]
        feature_min_j, feature_max_j = node_range(tree, idx_node, j)
        if x_tj < feature_min_j:
            diff = feature_min_j - x_tj
        elif x_tj > feature_max_j:
            diff = x_tj - feature_max_j
        else:
            diff = 0
        extensions[j] = diff
        extensions_sum += diff
    return extensions_sum


@njit(void(get_type(TreeClassifier), uint32, float32[::1]))
def node_predict(tree, idx_node, scores):
    # TODO: this is a bit silly ?... do everything at once
    for c in range(tree.n_classes):
        scores[c] = node_score(tree, idx_node, c)


@njit(float32(get_type(TreeClassifier), uint32, uint32))
def node_compute_split_time(tree, idx_node, idx_sample):
    samples = tree.samples
    nodes = tree.nodes
    y_t = samples.labels[idx_sample]
    #  Don't split if the node is pure: all labels are equal to the one of y_t
    if not tree.split_pure and node_is_dirac(tree, idx_node, y_t):
        return 0.0

    x_t = samples.features[idx_sample]
    extensions_sum = node_compute_range_extension(tree, idx_node, x_t, tree.intensities)
    # If x_t extends the current range of the node
    if extensions_sum > 0:
        # Sample an exponential with intensity = extensions_sum
        T = exponential(1 / extensions_sum)
        time = nodes.time[idx_node]
        # Splitting time of the node (if splitting occurs)
        split_time = time + T
        # If the node is a leaf we must split it
        if nodes.is_leaf[idx_node]:
            return split_time
        # Otherwise we apply Mondrian process dark magic :)
        # 1. We get the creation time of the childs (left and right is the same)
        left = nodes.left[idx_node]
        child_time = nodes.time[left]
        # 2. We check if splitting time occurs before child creation time
        if split_time < child_time:
            return split_time

    return 0


@njit(void(get_type(TreeClassifier), uint32, float32, float32, uint32, boolean,))
def node_split(tree, idx_node, split_time, threshold, feature, is_right_extension):
    # Create the two splits
    nodes = tree.nodes
    left_new = add_node(nodes, idx_node, split_time)
    right_new = add_node(nodes, idx_node, split_time)
    if is_right_extension:
        # left_new is the same as idx_node, excepted for the parent, time and the
        #  fact that it's a leaf
        copy_node(nodes, idx_node, left_new)
        # so we need to put back the correct parent and time
        nodes.parent[left_new] = idx_node
        nodes.time[left_new] = split_time
        # right_new must have idx_node has parent
        nodes.parent[right_new] = idx_node
        nodes.time[right_new] = split_time
        # We must tell the old childs that they have a new parent, if the
        # current node is not a leaf
        if not nodes.is_leaf[idx_node]:
            left = nodes.left[idx_node]
            right = nodes.right[idx_node]
            nodes.parent[left] = left_new
            nodes.parent[right] = left_new
    else:
        copy_node(nodes, idx_node, right_new)
        nodes.parent[right_new] = idx_node
        nodes.time[right_new] = split_time
        nodes.parent[left_new] = idx_node
        nodes.time[left_new] = split_time
        if not nodes.is_leaf[idx_node]:
            left = nodes.left[idx_node]
            right = nodes.right[idx_node]
            nodes.parent[left] = right_new
            nodes.parent[right] = right_new

    nodes.feature[idx_node] = feature
    nodes.threshold[idx_node] = threshold
    nodes.left[idx_node] = left_new
    nodes.right[idx_node] = right_new
    nodes.is_leaf[idx_node] = False
