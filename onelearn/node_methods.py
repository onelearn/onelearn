# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

from math import log
from numpy.random import exponential
from numba import njit
from numba.types import float32, boolean, uint32, uint8, void, Tuple

from .tree import TreeClassifier
from .utils import log_sum_2_exp


# TODO: write all the docstrings


@njit(float32(TreeClassifier.class_type.instance_type, uint32, uint32))
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
    count = tree.nodes.counts[node, idx_class]
    n_samples = tree.nodes.n_samples[node]
    n_classes = tree.n_classes
    dirichlet = tree.dirichlet
    # We use the Jeffreys prior with dirichlet parameter
    return (count + dirichlet) / (n_samples + dirichlet * n_classes)


@njit(float32(TreeClassifier.class_type.instance_type, uint32, uint32))
def node_loss(tree, node, idx_sample):

    c = uint8(tree.samples.labels[idx_sample])
    sc = node_score(tree, node, c)
    # TODO: benchmark different logarithms
    return -log(sc)


@njit(float32(TreeClassifier.class_type.instance_type, uint32, uint32))
def node_update_weight(tree, idx_node, idx_sample):
    loss_t = node_loss(tree, idx_node, idx_sample)
    if tree.use_aggregation:
        tree.nodes.weight[idx_node] -= tree.step * loss_t
    return loss_t


@njit(void(TreeClassifier.class_type.instance_type, uint32, uint32))
def node_update_count(tree, idx_node, idx_sample):
    # TODO: Don't do it twice...
    c = uint32(tree.samples.labels[idx_sample])
    tree.nodes.counts[idx_node, c] += 1


@njit
def node_print(tree, idx_node):
    print("----------------")
    print(
        "index:",
        tree.nodes.index[idx_node],
        "depth:",
        tree.nodes.depth[idx_node],
        "parent:",
        tree.nodes.parent[idx_node],
        "left:",
        tree.nodes.left[idx_node],
        "right:",
        tree.nodes.right[idx_node],
        "is_leaf:",
        tree.nodes.is_leaf[idx_node],
        "time:",
        tree.nodes.time[idx_node],
    )

    print(
        "feature:",
        tree.nodes.feature[idx_node],
        "threshold:",
        tree.nodes.threshold[idx_node],
        "weight:",
        tree.nodes.weight[idx_node],
        "log_tree_weight:",
        tree.nodes.log_weight_tree[idx_node],
    )

    print(
        "n_samples:",
        tree.nodes.n_samples[idx_node],
        "counts: [",
        tree.nodes.counts[idx_node, 0],
        ",",
        tree.nodes.counts[idx_node, 1],
        "]",
        "memorized:",
        tree.nodes.memorized[idx_node],
        "memory_range_min: [",
        tree.nodes.memory_range_min[idx_node, 0],
        ",",
        tree.nodes.memory_range_min[idx_node, 1],
        "]",
        "memory_range_max: [",
        tree.nodes.memory_range_max[idx_node, 0],
        ",",
        tree.nodes.memory_range_max[idx_node, 1],
        "]",
    )


@njit(void(TreeClassifier.class_type.instance_type, uint32, uint32, boolean))
def node_update_downwards(tree, idx_node, idx_sample, do_update_weight):
    x_t = tree.samples.features[idx_sample]
    memory_range_min = tree.nodes.memory_range_min[idx_node]
    memory_range_max = tree.nodes.memory_range_max[idx_node]
    # If it is the first sample, we copy the features vector into the min and
    # max range
    if tree.nodes.n_samples[idx_node] == 0:
        for j in range(tree.n_features):
            x_tj = x_t[j]
            memory_range_min[j] = x_tj
            memory_range_max[j] = x_tj
    # Otherwise, we update the range
    else:
        for j in range(tree.n_features):
            x_tj = x_t[j]
            if x_tj < memory_range_min[j]:
                memory_range_min[j] = x_tj
            if x_tj > memory_range_max[j]:
                memory_range_max[j] = x_tj

    # TODO: we should save the sample here and do a bunch of stuff about
    #  memorization
    # One more sample in the node
    tree.nodes.n_samples[idx_node] += 1

    if do_update_weight:
        # TODO: Using x_t and y_t should be better...
        node_update_weight(tree, idx_node, idx_sample)

    node_update_count(tree, idx_node, idx_sample)


@njit(void(TreeClassifier.class_type.instance_type, uint32))
def node_update_weight_tree(tree, idx_node):
    if tree.nodes.is_leaf[idx_node]:
        tree.nodes.log_weight_tree[idx_node] = tree.nodes.weight[idx_node]
    else:
        left = tree.nodes.left[idx_node]
        right = tree.nodes.right[idx_node]
        weight = tree.nodes.weight[idx_node]
        log_weight_tree = tree.nodes.log_weight_tree
        tree.nodes.weight[idx_node] = log_sum_2_exp(
            weight, log_weight_tree[left] + log_weight_tree[right]
        )


@njit(void(TreeClassifier.class_type.instance_type, uint32, uint8))
def node_update_depth(tree, idx_node, depth):
    depth += 1
    tree.nodes.depth[idx_node] = depth
    if tree.nodes.is_leaf[idx_node]:
        return
    else:
        left = tree.nodes.left[idx_node]
        right = tree.nodes.right[idx_node]
        node_update_depth(tree, left, depth)
        node_update_depth(tree, right, depth)


@njit(boolean(TreeClassifier.class_type.instance_type, uint32, float32))
def node_is_dirac(tree, idx_node, y_t):
    c = uint8(y_t)
    n_samples = tree.nodes.n_samples[idx_node]
    count = tree.nodes.counts[idx_node, c]
    return n_samples == count


@njit(uint32(TreeClassifier.class_type.instance_type, uint32, float32[::1]))
def node_get_child(tree, idx_node, x_t):
    feature = tree.nodes.feature[idx_node]
    threshold = tree.nodes.threshold[idx_node]
    if x_t[feature] <= threshold:
        return tree.nodes.left[idx_node]
    else:
        return tree.nodes.right[idx_node]


@njit(
    Tuple((float32, float32))(TreeClassifier.class_type.instance_type, uint32, uint32)
)
def node_range(tree, idx_node, j):
    # TODO: do the version without memory...
    if tree.nodes.n_samples[idx_node] == 0:
        raise RuntimeError("Node has no range since it has no samples")
    else:
        return (
            tree.nodes.memory_range_min[idx_node, j],
            tree.nodes.memory_range_max[idx_node, j],
        )


@njit(
    float32(TreeClassifier.class_type.instance_type, uint32, float32[::1], float32[::1])
)
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


@njit(void(TreeClassifier.class_type.instance_type, uint32, float32[::1]))
def node_predict(tree, idx_node, scores):
    # TODO: this is a bit silly ?... do everything at once
    for c in range(tree.n_classes):
        scores[c] = node_score(tree, idx_node, c)


@njit(float32(TreeClassifier.class_type.instance_type, uint32, uint32))
def node_compute_split_time(tree, idx_node, idx_sample):
    y_t = tree.samples.labels[idx_sample]
    #  Don't split if the node is pure: all labels are equal to the one of y_t
    # TODO: mais si idx_node est root on renvoie 0 forcement
    if tree.split_pure and node_is_dirac(tree, idx_node, y_t):
        return 0

    x_t = tree.samples.features[idx_sample]
    extensions_sum = node_compute_range_extension(tree, idx_node, x_t, tree.intensities)

    # If x_t extends the current range of the node
    if extensions_sum > 0:
        # Sample an exponential with intensity = extensions_sum
        T = exponential(1 / extensions_sum)
        time = tree.nodes.time[idx_node]
        # Splitting time of the node (if splitting occurs)
        split_time = time + T
        # If the node is a leaf we must split it
        if tree.nodes.is_leaf[idx_node]:
            # print("if tree.nodes.is_leaf[idx_node]:")
            return split_time
        # Otherwise we apply Mondrian process dark magic :)
        # 1. We get the creation time of the childs (left and right is the
        #    same)
        left = tree.nodes.left[idx_node]
        child_time = tree.nodes.time[left]
        # 2. We check if splitting time occurs before child creation time
        # print("split_time < child_time:", split_time, "<", child_time)
        if split_time < child_time:
            return split_time

    return 0


@njit(
    void(
        TreeClassifier.class_type.instance_type,
        uint32,
        float32,
        float32,
        uint32,
        boolean,
    )
)
def node_split(tree, idx_node, split_time, threshold, feature, is_right_extension):
    # Create the two splits
    left_new = tree.nodes.add_node(idx_node, split_time)
    right_new = tree.nodes.add_node(idx_node, split_time)
    if is_right_extension:
        # left_new is the same as idx_node, excepted for the parent, time
        # and the fact that it's a leaf
        tree.nodes.copy_node(idx_node, left_new)
        # so we need to put back the correct parent and time
        tree.nodes.parent[left_new] = idx_node
        tree.nodes.time[left_new] = split_time
        # right_new must have idx_node has parent
        tree.nodes.parent[right_new] = idx_node
        tree.nodes.time[right_new] = split_time
        # We must tell the old childs that they have a new parent, if the
        # current node is not a leaf
        if not tree.nodes.is_leaf[idx_node]:
            left = tree.nodes.left[idx_node]
            right = tree.nodes.right[idx_node]
            tree.nodes.parent[left] = left_new
            tree.nodes.parent[right] = left_new
    else:
        tree.nodes.copy_node(idx_node, right_new)
        tree.nodes.parent[right_new] = idx_node
        tree.nodes.time[right_new] = split_time
        tree.nodes.parent[left_new] = idx_node
        tree.nodes.time[left_new] = split_time
        if not tree.nodes.is_leaf[idx_node]:
            left = tree.nodes.left[idx_node]
            right = tree.nodes.right[idx_node]
            tree.nodes.parent[left] = right_new
            tree.nodes.parent[right] = right_new

    tree.nodes.feature[idx_node] = feature
    tree.nodes.threshold[idx_node] = threshold
    tree.nodes.left[idx_node] = left_new
    tree.nodes.right[idx_node] = right_new
    tree.nodes.is_leaf[idx_node] = False
