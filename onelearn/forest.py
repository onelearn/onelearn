# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import os
import pickle as pkl
import numpy as np
from numba import njit
from numba.experimental import jitclass
from numba import types, _helperlib
from .types import float32, boolean, uint32, string, void, get_array_2d_type
from .checks import check_X_y, check_array
from .sample import (
    SamplesCollection,
    add_samples,
    samples_collection_to_dict,
    dict_to_samples_collection,
)
from .tree import (
    TreeClassifier,
    TreeRegressor,
    tree_classifier_to_dict,
    tree_regressor_to_dict,
)
from .tree_methods import (
    tree_classifier_partial_fit,
    tree_regressor_partial_fit,
    tree_classifier_predict,
    tree_regressor_predict,
    tree_regressor_weighted_depth,
)
from .node_collection import dict_to_nodes_classifier, dict_to_nodes_regressor
from .utils import get_type

spec_amf_learner = [
    ("n_features", uint32),
    ("n_estimators", uint32),
    ("step", float32),
    ("loss", string),
    ("use_aggregation", boolean),
    ("split_pure", boolean),
    ("n_jobs", uint32),
    ("n_samples_increment", uint32),
    ("verbose", boolean),
    ("samples", get_type(SamplesCollection)),
    ("iteration", uint32),
]

spec_amf_classifier = spec_amf_learner + [
    ("n_classes", uint32),
    ("dirichlet", float32),
    ("trees", types.List(get_type(TreeClassifier), reflected=True)),
]


# TODO: we can force pre-compilation when creating the nopython forest


@jitclass(spec_amf_classifier)
class AMFClassifierNoPython(object):
    def __init__(
        self,
        n_classes,
        n_features,
        n_estimators,
        step,
        loss,
        use_aggregation,
        dirichlet,
        split_pure,
        n_jobs,
        n_samples_increment,
        verbose,
        samples,
        trees_iteration,
        trees_n_nodes,
        trees_n_nodes_capacity,
    ):
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.dirichlet = dirichlet
        self.split_pure = split_pure
        self.n_jobs = n_jobs
        self.n_samples_increment = n_samples_increment
        self.verbose = verbose
        self.samples = samples
        if trees_iteration.size == 0:
            self.iteration = 0
            # TODO: reflected lists will be replaced by typed list soon...
            iteration = 0
            n_nodes = 0
            n_nodes_capacity = 0
            trees = [
                TreeClassifier(
                    self.n_classes,
                    self.n_features,
                    self.step,
                    self.loss,
                    self.use_aggregation,
                    self.dirichlet,
                    self.split_pure,
                    self.samples,
                    iteration,
                    n_nodes,
                    n_nodes_capacity,
                )
                for _ in range(n_estimators)
            ]
            self.trees = trees
        else:
            trees = [
                TreeClassifier(
                    self.n_classes,
                    self.n_features,
                    self.step,
                    self.loss,
                    self.use_aggregation,
                    self.dirichlet,
                    self.split_pure,
                    self.samples,
                    trees_iteration[n_estimator],
                    trees_n_nodes[n_estimator],
                    trees_n_nodes_capacity[n_estimator],
                )
                for n_estimator in range(n_estimators)
            ]
            self.trees = trees


@njit(void(get_type(AMFClassifierNoPython), get_array_2d_type(float32), float32[::1],))
def forest_classifier_partial_fit(forest, X, y):
    n_samples_batch, n_features = X.shape
    # First, we save the new batch of data
    n_samples_before = forest.samples.n_samples
    # Add the samples in the forest
    add_samples(forest.samples, X, y)
    for i in range(n_samples_before, n_samples_before + n_samples_batch):
        # Then we fit all the trees using all new samples
        for tree in forest.trees:
            tree_classifier_partial_fit(tree, i)
        forest.iteration += 1


# TODO: code predict
# def predict(self, X, scores):
#     scores.fill(0.0)
#     n_samples_batch, _ = X.shape
#     if self.iteration > 0:
#         scores_tree = np.empty(self.n_classes, float32)
#         for i in range(n_samples_batch):
#             # print('i:', i)
#             scores_i = scores[i]
#             x_i = X[i]
#             # print('x_i:', x_i)
#             # The prediction is simply the average of the predictions
#             for tree in self.trees:
#                 tree_predict(tree, x_i, scores_tree, self.use_aggregation)
#                 # print('scores_tree:', scores_tree)
#                 scores_i += scores_tree
#             scores_i /= self.n_estimators
#             # print('scores_i:', scores_i)
#     else:
#         raise RuntimeError("You must call ``partial_fit`` before ``predict``.")


@njit(
    void(
        get_type(AMFClassifierNoPython),
        get_array_2d_type(float32),
        get_array_2d_type(float32),
    )
)
def forest_classifier_predict_proba(forest, X, scores):
    # TODO: use predict_proba_tree from below ? Or put it in the tree ?
    scores.fill(0.0)
    n_samples_batch, _ = X.shape

    scores_tree = np.empty(forest.n_classes, float32)
    for i in range(n_samples_batch):
        scores_i = scores[i]
        x_i = X[i]
        # The prediction is simply the average of the predictions
        for tree in forest.trees:
            tree_classifier_predict(tree, x_i, scores_tree, forest.use_aggregation)
            scores_i += scores_tree
        scores_i /= forest.n_estimators


@njit(
    get_array_2d_type(float32)(
        get_type(AMFClassifierNoPython), uint32, get_array_2d_type(float32)
    )
)
def forest_classifier_predict_proba_tree(forest, idx_tree, X):
    n_samples_batch, _ = X.shape
    scores = np.empty((n_samples_batch, forest.n_classes), dtype=float32)
    tree = forest.trees[idx_tree]
    for i in range(n_samples_batch):
        scores_i = scores[i]
        x_i = X[i]
        tree_classifier_predict(tree, x_i, scores_i, forest.use_aggregation)
    return scores


def amf_classifier_nopython_to_dict(forest):
    d = {}
    for key, _ in spec_amf_classifier:
        if key == "samples":
            d["samples"] = samples_collection_to_dict(forest.samples)
        elif key == "trees":
            d["trees"] = [tree_classifier_to_dict(tree) for tree in forest.trees]
        else:
            d[key] = getattr(forest, key)
    return d


def dict_to_amf_classifier_nopython(d):
    n_classes = d["n_classes"]
    n_features = d["n_features"]
    n_estimators = d["n_estimators"]
    step = d["step"]
    loss = d["loss"]
    use_aggregation = d["use_aggregation"]
    dirichlet = d["dirichlet"]
    split_pure = d["split_pure"]
    n_jobs = d["n_jobs"]
    n_samples_increment = d["n_samples_increment"]
    verbose = d["verbose"]
    # Create the samples jitclass from a dict
    samples = dict_to_samples_collection(d["samples"])

    trees_dict = d["trees"]
    trees_iteration = np.array(
        [tree_dict["iteration"] for tree_dict in trees_dict], dtype=np.uint32
    )
    trees_n_nodes = np.array(
        [tree_dict["nodes"]["n_nodes"] for tree_dict in trees_dict], dtype=np.uint32
    )
    trees_n_nodes_capacity = np.array(
        [tree_dict["nodes"]["n_nodes_capacity"] for tree_dict in trees_dict],
        dtype=np.uint32,
    )
    no_python = AMFClassifierNoPython(
        n_classes,
        n_features,
        n_estimators,
        step,
        loss,
        use_aggregation,
        dirichlet,
        split_pure,
        n_jobs,
        n_samples_increment,
        verbose,
        samples,
        trees_iteration,
        trees_n_nodes,
        trees_n_nodes_capacity,
    )
    no_python.iteration = d["iteration"]
    no_python.samples = samples
    trees = no_python.trees

    # no_python is initialized, it remains to initialize the nodes
    for n_estimator in range(n_estimators):
        tree_dict = trees_dict[n_estimator]
        nodes_dict = tree_dict["nodes"]
        tree = trees[n_estimator]
        nodes = tree.nodes
        # Copy node information
        dict_to_nodes_classifier(nodes, nodes_dict)
        # Copy intensities
        tree.intensities[:] = tree_dict["intensities"]

    return no_python


spec_amf_regressor = spec_amf_learner + [
    ("trees", types.List(get_type(TreeRegressor), reflected=True)),
]
# TODO: we can force pre-compilation when creating the nopython forest


@jitclass(spec_amf_regressor)
class AMFRegressorNoPython(object):
    def __init__(
        self,
        n_features,
        n_estimators,
        step,
        loss,
        use_aggregation,
        split_pure,
        n_jobs,
        n_samples_increment,
        verbose,
        samples,
        trees_iteration,
        trees_n_nodes,
        trees_n_nodes_capacity,
    ):
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.split_pure = split_pure
        self.n_jobs = n_jobs
        self.n_samples_increment = n_samples_increment
        self.verbose = verbose
        self.samples = samples
        if trees_iteration.size == 0:
            self.iteration = 0
            iteration = 0
            n_nodes = 0
            n_nodes_capacity = 0
            trees = [
                TreeRegressor(
                    self.n_features,
                    self.step,
                    self.loss,
                    self.use_aggregation,
                    self.split_pure,
                    self.samples,
                    iteration,
                    n_nodes,
                    n_nodes_capacity,
                )
                for _ in range(n_estimators)
            ]
            self.trees = trees
        else:
            trees = [
                TreeRegressor(
                    self.n_features,
                    self.step,
                    self.loss,
                    self.use_aggregation,
                    self.split_pure,
                    self.samples,
                    trees_iteration[n_estimator],
                    trees_n_nodes[n_estimator],
                    trees_n_nodes_capacity[n_estimator],
                )
                for n_estimator in range(n_estimators)
            ]
            self.trees = trees


@njit(void(get_type(AMFRegressorNoPython), get_array_2d_type(float32), float32[::1]))
def forest_regressor_partial_fit(forest, X, y):
    n_samples_batch, n_features = X.shape
    # First, we save the new batch of data
    n_samples_before = forest.samples.n_samples
    # Add the samples in the forest
    add_samples(forest.samples, X, y)
    for i in range(n_samples_before, n_samples_before + n_samples_batch):
        # Then we fit all the trees using all new samples
        for tree in forest.trees:
            tree_regressor_partial_fit(tree, i)
        forest.iteration += 1


# TODO: code predict
# def predict(self, X, scores):
#     scores.fill(0.0)
#     n_samples_batch, _ = X.shape
#     if self.iteration > 0:
#         scores_tree = np.empty(self.n_classes, float32)
#         for i in range(n_samples_batch):
#             # print('i:', i)
#             scores_i = scores[i]
#             x_i = X[i]
#             # print('x_i:', x_i)
#             # The prediction is simply the average of the predictions
#             for tree in self.trees:
#                 tree_predict(tree, x_i, scores_tree, self.use_aggregation)
#                 # print('scores_tree:', scores_tree)
#                 scores_i += scores_tree
#             scores_i /= self.n_estimators
#             # print('scores_i:', scores_i)
#     else:
#         raise RuntimeError("You must call ``partial_fit`` before ``predict``.")


@njit(void(get_type(AMFRegressorNoPython), get_array_2d_type(float32), float32[::1]))
def forest_regressor_predict(forest, X, predictions):
    # TODO: Useless ?
    predictions.fill(0.0)
    n_samples_batch, _ = X.shape
    for i in range(n_samples_batch):
        x_i = X[i]
        prediction = 0
        # The prediction is simply the average of the predictions
        for tree in forest.trees:
            prediction += tree_regressor_predict(tree, x_i, forest.use_aggregation)
        predictions[i] = prediction / forest.n_estimators


def amf_regressor_nopython_to_dict(forest):
    d = {}
    for key, _ in spec_amf_regressor:
        if key == "samples":
            d["samples"] = samples_collection_to_dict(forest.samples)
        elif key == "trees":
            d["trees"] = [tree_regressor_to_dict(tree) for tree in forest.trees]
        else:
            d[key] = getattr(forest, key)
    return d


def dict_to_amf_regressor_nopython(d):
    n_features = d["n_features"]
    n_estimators = d["n_estimators"]
    step = d["step"]
    loss = d["loss"]
    use_aggregation = d["use_aggregation"]
    split_pure = d["split_pure"]
    n_jobs = d["n_jobs"]
    n_samples_increment = d["n_samples_increment"]
    verbose = d["verbose"]
    # Create the samples jitclass from a dict
    samples = dict_to_samples_collection(d["samples"])

    trees_dict = d["trees"]
    trees_iteration = np.array(
        [tree_dict["iteration"] for tree_dict in trees_dict], dtype=np.uint32
    )
    trees_n_nodes = np.array(
        [tree_dict["nodes"]["n_nodes"] for tree_dict in trees_dict], dtype=np.uint32
    )
    trees_n_nodes_capacity = np.array(
        [tree_dict["nodes"]["n_nodes_capacity"] for tree_dict in trees_dict],
        dtype=np.uint32,
    )
    no_python = AMFRegressorNoPython(
        n_features,
        n_estimators,
        step,
        loss,
        use_aggregation,
        split_pure,
        n_jobs,
        n_samples_increment,
        verbose,
        samples,
        trees_iteration,
        trees_n_nodes,
        trees_n_nodes_capacity,
    )
    no_python.iteration = d["iteration"]
    no_python.samples = samples
    trees = no_python.trees

    # no_python is initialized, it remains to initialize the nodes
    for n_estimator in range(n_estimators):
        tree_dict = trees_dict[n_estimator]
        nodes_dict = tree_dict["nodes"]
        tree = trees[n_estimator]
        nodes = tree.nodes
        # Copy node information
        dict_to_nodes_regressor(nodes, nodes_dict)
        # Copy intensities
        tree.intensities[:] = tree_dict["intensities"]

    return no_python


@njit(
    void(
        get_type(AMFRegressorNoPython),
        get_array_2d_type(float32),
        get_array_2d_type(float32),
    )
)
def forest_regressor_weighted_depths(forest, X, weighted_depths):
    n_samples_batch, _ = X.shape
    for i in range(n_samples_batch):
        x_i = X[i]
        n_tree = 0
        for tree in forest.trees:
            weighted_depth = tree_regressor_weighted_depth(
                tree, x_i, forest.use_aggregation
            )
            weighted_depths[i, n_tree] = weighted_depth
            n_tree += 1


@njit(
    get_array_2d_type(float32)(
        get_type(AMFClassifierNoPython), uint32, get_array_2d_type(float32)
    )
)
def forest_classifier_predict_proba_tree(forest, idx_tree, X):
    n_samples_batch, _ = X.shape
    scores = np.empty((n_samples_batch, forest.n_classes), dtype=float32)
    tree = forest.trees[idx_tree]
    for i in range(n_samples_batch):
        scores_i = scores[i]
        x_i = X[i]
        tree_classifier_predict(tree, x_i, scores_i, forest.use_aggregation)
    return scores


# TODO: make amf.nopython.partial_fit work in a jitted function, test it and document it


class AMFLearner(object):
    """Base class for Aggregated Mondrian Forest classifier and regressors for online
    learning.

    Note
    ----
    This class is not intended for end users but for development only.

    """

    def __init__(
        self,
        n_estimators,
        step,
        loss,
        use_aggregation,
        split_pure,
        n_jobs,
        n_samples_increment,
        random_state,
        verbose,
    ):
        """Instantiates a `AMFLearner` instance.

        Parameters
        ----------
        n_estimators : :obj:`int`
            The number of trees in the forest.

        step : :obj:`float`
            Step-size for the aggregation weights.

        loss : :obj:`str`
            The loss used for the computation of the aggregation weights.

        use_aggregation : :obj:`bool`
            Controls if aggregation is used in the trees. It is highly recommended to
            leave it as `True`.

        split_pure : :obj:`bool`
            Controls if nodes that contains only sample of the same class should be
            split ("pure" nodes). Default is `False`, namely pure nodes are not split,
            but `True` can be sometimes better.

        n_jobs : :obj:`int`
            Sets the number of threads used to grow the tree in parallel. The default is
            n_jobs=1, namely single-threaded. Fow now, this parameter has no effect and
            only a single thread can be used.

        n_samples_increment : :obj:`int`
            Sets the minimum amount of memory which is pre-allocated each time extra
            memory is required for new samples and new nodes. Decreasing it can slow
            down training. If you know that each ``partial_fit`` will be called with
            approximately `n` samples, you can set n_samples_increment = `n` if `n` is
            larger than the default.

        random_state : :obj:`int` or :obj:`None`
            Controls the randomness involved in the trees.

        verbose : :obj:`bool`, default = `False`
            Controls the verbosity when fitting and predicting.
        """
        # We will instantiate the numba class when data is passed to
        # `partial_fit`, since we need to know about `n_features` among others things
        self.no_python = None
        self._n_features = None
        self.n_estimators = n_estimators
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.split_pure = split_pure
        self.n_jobs = n_jobs
        self.n_samples_increment = n_samples_increment
        self.random_state = random_state
        self.verbose = verbose

        if os.getenv("NUMBA_DISABLE_JIT", None) == "1":
            self._using_numba = False
        else:
            self._using_numba = True

    def partial_fit_helper(self, X, y):
        """Updates the classifier with the given batch of samples.

        Parameters
        ----------
        X : :obj:`np.ndarray`, shape=(n_samples, n_features)
            Input features matrix.

        y : :obj:`np.ndarray`
            Input labels vector.

        classes : :obj:`None`
            Must not be used, only here for backwards compatibility

        Returns
        -------
        output : :obj:`AMFClassifier`
            Updated instance of :obj:`AMFClassifier`

        """
        # First,ensure that X and y are C-contiguous and with float32 dtype
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="float32",
            order="C",
            copy=False,
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            multi_output=False,
            ensure_min_samples=1,
            ensure_min_features=1,
            y_numeric=True,
            estimator=self.__class__.__name__,
        )
        n_samples, n_features = X.shape

        self._extra_y_test(y)
        # This is the first call to `partial_fit`, so we need to instantiate
        # the no python class
        if self.no_python is None:
            self._n_features = n_features
            self._instantiate_nopython_class()
        else:
            _, n_features = X.shape
            if n_features != self.n_features:
                raise ValueError(
                    "`partial_fit` was first called with n_features=%d while "
                    "n_features=%d in this call" % (self.n_features, n_features)
                )
        self._set_random_state()
        self._partial_fit(X, y)
        self._put_back_random_state()
        return self

    # TODO: such methods should be private
    def predict_helper(self, X):
        """Helper method for the predictions of the given features vectors. This is used
        in the ``predict`` and ``predict_proba`` methods of ``AMFRegressor`` and
        ``AMFClassifier``.

        Parameters
        ----------
        X : :obj:`np.ndarray`, shape=(n_samples, n_features)
            Input features matrix to predict for.

        Returns
        -------
        output : :obj:`np.ndarray`
            Returns the predictions for the input features

        """
        X = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype=["float32"],
            order="C",
            copy=False,
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            ensure_min_samples=1,
            ensure_min_features=1,
            estimator=self.__class__.__name__,
        )
        n_samples, n_features = X.shape
        if not self.no_python:
            raise RuntimeError(
                "You must call `partial_fit` before asking for predictions"
            )
        else:
            if n_features != self.n_features:
                raise ValueError(
                    "`partial_fit` was called with n_features=%d while predictions are "
                    "asked with n_features=%d" % (self.n_features, n_features)
                )
        # TODO: this is useless for predictions ?!?
        self._set_random_state()
        predictions = self._compute_predictions(X)
        self._put_back_random_state()
        return predictions

    def weighted_depth_helper(self, X):
        X = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype=["float32"],
            order="C",
            copy=False,
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            ensure_min_samples=1,
            ensure_min_features=1,
            estimator=self.__class__.__name__,
        )
        n_samples, n_features = X.shape
        if not self.no_python:
            raise RuntimeError(
                "You must call `partial_fit` before asking for weighted depths"
            )
        else:
            if n_features != self.n_features:
                raise ValueError(
                    "`partial_fit` was called with n_features=%d while depths are "
                    "asked with n_features=%d" % (self.n_features, n_features)
                )
        weighted_depths = self._compute_weighted_depths(X)
        return weighted_depths

    @classmethod
    def load(cls, filename):
        """Loads a AMF object from file (created with :meth:`save`)

        Parameters
        ----------
        filename : :obj:`str`
            Filename containing the serialized AMF object

        Returns
        -------
        output : object
            Either AMFClassifier or AMFRegressor contained in the file
        """
        with open(filename, "rb") as f:
            d = pkl.load(f)
            return cls._from_dict(d)

    def save(self, filename):
        """Saves a AMF object to file using pickle

        Parameters
        ----------
        filename : :obj:`str`
            Filename containing the serialized AMF object
        """

        with open(filename, "wb") as f:
            d = self._to_dict()
            pkl.dump(d, f)

    def _compute_predictions(self, X):
        pass

    def _extra_y_test(self, y):
        pass

    def _instantiate_nopython_class(self):
        pass

    def _set_random_state(self):
        # This uses a trick by Alexandre Gramfort,
        #   see https://github.com/numba/numba/issues/3249
        if self._random_state >= 0:
            if self._using_numba:
                r = np.random.RandomState(self._random_state)
                ptr = _helperlib.rnd_get_np_state_ptr()
                ints, index = r.get_state()[1:3]
                _helperlib.rnd_set_state(ptr, (index, [int(x) for x in ints]))
                self._ptr = ptr
                self._r = r
            else:
                np.random.seed(self._random_state)

    def _put_back_random_state(self):
        # This uses a trick by Alexandre Gramfort,
        #   see https://github.com/numba/numba/issues/3249
        if self._random_state >= 0:
            if self._using_numba:
                ptr = self._ptr
                r = self._r
                index, ints = _helperlib.rnd_get_state(ptr)
                r.set_state(("MT19937", ints, index, 0, 0.0))

    def get_nodes_df(self, idx_tree):
        import pandas as pd

        tree = self.no_python.trees[idx_tree]
        nodes = tree.nodes
        n_nodes = nodes.n_nodes
        index = nodes.index[:n_nodes]
        parent = nodes.parent[:n_nodes]
        left = nodes.left[:n_nodes]
        right = nodes.right[:n_nodes]
        feature = nodes.feature[:n_nodes]
        threshold = nodes.threshold[:n_nodes]
        time = nodes.time[:n_nodes]
        depth = nodes.depth[:n_nodes]
        memory_range_min = nodes.memory_range_min[:n_nodes]
        memory_range_max = nodes.memory_range_max[:n_nodes]
        n_samples = nodes.n_samples[:n_nodes]
        weight = nodes.weight[:n_nodes]
        log_weight_tree = nodes.log_weight_tree[:n_nodes]
        is_leaf = nodes.is_leaf[:n_nodes]
        # is_memorized = nodes.is_memorized[:n_nodes]
        counts = nodes.counts[:n_nodes]

        columns = [
            "id",
            "parent",
            "left",
            "right",
            "depth",
            "is_leaf",
            "feature",
            "threshold",
            "time",
            "n_samples",
            "weight",
            "log_weight_tree",
            "memory_range_min",
            "memory_range_max",
            "counts",
        ]

        data = {
            "id": index,
            "parent": parent,
            "left": left,
            "right": right,
            "depth": depth,
            "feature": feature,
            "threshold": threshold,
            "is_leaf": is_leaf,
            "time": time,
            "n_samples": n_samples,
            "weight": weight,
            "log_weight_tree": log_weight_tree,
            "memory_range_min": [tuple(t) for t in memory_range_min],
            "memory_range_max": [tuple(t) for t in memory_range_max],
            "counts": [tuple(t) for t in counts],
        }
        df = pd.DataFrame(data, columns=columns)
        return df

    @property
    def n_features(self):
        """:obj:`int`: Number of features used during training."""
        return self._n_features

    @n_features.setter
    def n_features(self, val):
        raise ValueError("`n_features` is a readonly attribute")

    @property
    def n_estimators(self):
        """:obj:`int`: Number of trees in the forest."""
        return self._n_estimators

    @n_estimators.setter
    def n_estimators(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `n_estimators` after calling `partial_fit`"
            )
        else:
            if not isinstance(val, int):
                raise ValueError("`n_estimators` must be of type `int`")
            elif val < 1:
                raise ValueError("`n_estimators` must be >= 1")
            else:
                self._n_estimators = val

    @property
    def n_jobs(self):
        """:obj:`int`: Number of expected classes in the labels."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, val):
        if self.no_python:
            raise ValueError("You cannot modify `n_jobs` after calling `partial_fit`")
        else:
            if not isinstance(val, int):
                raise ValueError("`n_jobs` must be of type `int`")
            elif val < 1:
                raise ValueError("`n_jobs` must be >= 1")
            else:
                self._n_jobs = val

    @property
    def n_samples_increment(self):
        """:obj:`int`: Amount of memory pre-allocated each time extra memory is
        required."""
        return self._n_samples_increment

    @n_samples_increment.setter
    def n_samples_increment(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `n_samples_increment` after calling `partial_fit`"
            )
        else:
            if not isinstance(val, int):
                raise ValueError("`n_samples_increment` must be of type `int`")
            elif val < 1:
                raise ValueError("`n_samples_increment` must be >= 1")
            else:
                self._n_samples_increment = val

    @property
    def step(self):
        """:obj:`float`: Step-size for the aggregation weights."""
        return self._step

    @step.setter
    def step(self, val):
        if self.no_python:
            raise ValueError("You cannot modify `step` after calling `partial_fit`")
        else:
            if not isinstance(val, float):
                raise ValueError("`step` must be of type `float`")
            elif val <= 0:
                raise ValueError("`step` must be > 0")
            else:
                self._step = val

    @property
    def use_aggregation(self):
        """:obj:`bool`: Controls if aggregation is used in the trees."""
        return self._use_aggregation

    @use_aggregation.setter
    def use_aggregation(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `use_aggregation` after calling `partial_fit`"
            )
        else:
            if not isinstance(val, bool):
                raise ValueError("`use_aggregation` must be of type `bool`")
            else:
                self._use_aggregation = val

    @property
    def split_pure(self):
        """:obj:`bool`: Controls if nodes that contains only sample of the same class
        should be split."""
        return self._split_pure

    @split_pure.setter
    def split_pure(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `split_pure` after calling `partial_fit`"
            )
        else:
            if not isinstance(val, bool):
                raise ValueError("`split_pure` must be of type `bool`")
            else:
                self._split_pure = val

    @property
    def verbose(self):
        """:obj:`bool`: Controls the verbosity when fitting and predicting."""
        return self._verbose

    @verbose.setter
    def verbose(self, val):
        if self.no_python:
            raise ValueError("You cannot modify `verbose` after calling `partial_fit`")
        else:
            if not isinstance(val, bool):
                raise ValueError("`verbose` must be of type `bool`")
            else:
                self._verbose = val

    @property
    def loss(self):
        """:obj:`str`: The loss used for the computation of the aggregation weights."""
        return "log"

    @loss.setter
    def loss(self, val):
        pass

    @property
    def random_state(self):
        """:obj:`int` or :obj:`None`: Controls the randomness involved in the trees."""
        if self._random_state == -1:
            return None
        else:
            return self._random_state

    @random_state.setter
    def random_state(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `random_state` after calling `partial_fit`"
            )
        else:
            if val is None:
                self._random_state = -1
            elif not isinstance(val, int):
                raise ValueError("`random_state` must be of type `int`")
            elif val < 0:
                raise ValueError("`random_state` must be >= 0")
            else:
                self._random_state = val

    def __repr__(self):
        r = self.__class__.__name__
        r += "n_estimators={n_estimators}, ".format(n_estimators=self.n_estimators)
        r += "step={step}, ".format(step=self.step)
        r += "loss={loss}, ".format(loss=self.loss)
        r += "use_aggregation={use_aggregation}, ".format(
            use_aggregation=self.use_aggregation
        )
        r += "split_pure={split_pure}, ".format(split_pure=self.split_pure)
        r += "n_jobs={n_jobs}, ".format(n_jobs=self.n_jobs)
        r += "random_state={random_state}, ".format(random_state=self.random_state)
        r += "verbose={verbose})".format(verbose=self.verbose)
        return r


# TODO: add attributes in docstring


class AMFClassifier(AMFLearner):
    """Aggregated Mondrian Forest classifier for online learning. This algorithm
    is truly online, in the sense that a single pass is performed, and that predictions
    can be produced anytime.

    Each node in a tree predicts according to the distribution of the labels
    it contains. This distribution is regularized using a "Jeffreys" prior
    with parameter ``dirichlet``. For each class with `count` labels in the
    node and `n_samples` samples in it, the prediction of a node is given by

        (count + dirichlet) / (n_samples + dirichlet * n_classes)

    The prediction for a sample is computed as the aggregated predictions of all the
    subtrees along the path leading to the leaf node containing the sample. The
    aggregation weights are exponential weights with learning rate ``step`` and loss
    ``loss`` when ``use_aggregation`` is ``True``.

    This computation is performed exactly thanks to a context tree weighting algorithm.
    More details can be found in the paper cited in references below.

    The final predictions are the average class probabilities predicted by each of the
    ``n_estimators`` trees in the forest.

    Note
    ----
    All the parameters of ``AMFClassifier`` become **read-only** after the first call
    to ``partial_fit``

    References
    ----------
    J. Mourtada, S. Gaiffas and E. Scornet, *AMF: Aggregated Mondrian Forests for Online Learning*, arXiv:1906.10529, 2019

    """

    def __init__(
        self,
        n_classes,
        n_estimators=10,
        step=1.0,
        loss="log",
        use_aggregation=True,
        dirichlet=None,
        split_pure=False,
        n_jobs=1,
        n_samples_increment=1024,
        random_state=None,
        verbose=False,
    ):
        """Instantiates a `AMFClassifier` instance.

        Parameters
        ----------
        n_classes : :obj:`int`
            Number of expected classes in the labels. This is required since we
            don't know the number of classes in advance in a online setting.

        n_estimators : :obj:`int`, default = 10
            The number of trees in the forest.

        step : :obj:`float`, default = 1
            Step-size for the aggregation weights. Default is 1 for classification with
            the log-loss, which is usually the best choice.

        loss : {"log"}, default = "log"
            The loss used for the computation of the aggregation weights. Only "log"
            is supported for now, namely the log-loss for multi-class
            classification.

        use_aggregation : :obj:`bool`, default = `True`
            Controls if aggregation is used in the trees. It is highly recommended to
            leave it as `True`.

        dirichlet : :obj:`float` or :obj:`None`, default = `None`
            Regularization level of the class frequencies used for predictions in each
            node. Default is dirichlet=0.5 for n_classes=2 and dirichlet=0.01 otherwise.

        split_pure : :obj:`bool`, default = `False`
            Controls if nodes that contains only sample of the same class should be
            split ("pure" nodes). Default is `False`, namely pure nodes are not split,
            but `True` can be sometimes better.

        n_jobs : :obj:`int`, default = 1
            Sets the number of threads used to grow the tree in parallel. The default is
            n_jobs=1, namely single-threaded. Fow now, this parameter has no effect and
            only a single thread can be used.

        n_samples_increment : :obj:`int`, default = 1024
            Sets the minimum amount of memory which is pre-allocated each time extra
            memory is required for new samples and new nodes. Decreasing it can slow
            down training. If you know that each ``partial_fit`` will be called with
            approximately `n` samples, you can set n_samples_increment = `n` if `n` is
            larger than the default.

        random_state : :obj:`int` or :obj:`None`, default = `None`
            Controls the randomness involved in the trees.

        verbose : :obj:`bool`, default = `False`
            Controls the verbosity when fitting and predicting.
        """
        AMFLearner.__init__(
            self,
            n_estimators=n_estimators,
            step=step,
            loss=loss,
            use_aggregation=use_aggregation,
            split_pure=split_pure,
            n_jobs=n_jobs,
            n_samples_increment=n_samples_increment,
            random_state=random_state,
            verbose=verbose,
        )

        self.n_classes = n_classes
        if dirichlet is None:
            if self.n_classes == 2:
                self.dirichlet = 0.5
            else:
                self.dirichlet = 0.01
        else:
            self.dirichlet = dirichlet

        self._classes = set(range(n_classes))

    def _extra_y_test(self, y):
        if y.min() < 0:
            raise ValueError("All the values in `y` must be non-negative")
        y_max = y.max()
        if y_max not in self._classes:
            raise ValueError("n_classes=%d while y.max()=%d" % (self.n_classes, y_max))

    def _to_dict(self):
        attrs = [
            "_n_features",
            "n_classes",
            "n_estimators",
            "step",
            "loss",
            "use_aggregation",
            "dirichlet",
            "split_pure",
            "n_jobs",
            "n_samples_increment",
            "random_state",
            "verbose",
            "_classes",
        ]
        d = {}
        for key in attrs:
            d[key] = getattr(self, key)
        d["no_python"] = amf_classifier_nopython_to_dict(self.no_python)
        return d

    @classmethod
    def _from_dict(cls, d):
        amf = AMFClassifier(
            n_classes=d["n_classes"],
            n_estimators=d["n_estimators"],
            step=d["step"],
            loss=d["loss"],
            use_aggregation=d["use_aggregation"],
            dirichlet=d["dirichlet"],
            split_pure=d["split_pure"],
            n_jobs=d["n_jobs"],
            n_samples_increment=d["n_samples_increment"],
            random_state=d["random_state"],
            verbose=d["verbose"],
        )
        amf._n_features = d["_n_features"]
        amf.no_python = dict_to_amf_classifier_nopython(d["no_python"])
        return amf

    def _instantiate_nopython_class(self):
        trees_iteration = np.empty(0, dtype=np.uint32)
        trees_n_nodes = np.empty(0, dtype=np.uint32)
        trees_n_nodes_capacity = np.empty(0, dtype=np.uint32)
        n_samples = 0
        n_samples_capacity = 0
        samples = SamplesCollection(
            self.n_samples_increment, self.n_features, n_samples, n_samples_capacity
        )
        self.no_python = AMFClassifierNoPython(
            self.n_classes,
            self.n_features,
            self.n_estimators,
            self.step,
            self.loss,
            self.use_aggregation,
            self.dirichlet,
            self.split_pure,
            self.n_jobs,
            self.n_samples_increment,
            self.verbose,
            samples,
            trees_iteration,
            trees_n_nodes,
            trees_n_nodes_capacity,
        )

    def _partial_fit(self, X, y):
        forest_classifier_partial_fit(self.no_python, X, y)

    def partial_fit(self, X, y, classes=None):
        """Updates the classifier with the given batch of samples.

        Parameters
        ----------
        X : :obj:`np.ndarray`, shape=(n_samples, n_features)
            Input features matrix.

        y : :obj:`np.ndarray`
            Input labels vector.

        classes : :obj:`None`
            Must not be used, only here for backwards compatibility

        Returns
        -------
        output : :obj:`AMFClassifier`
            Updated instance of :obj:`AMFClassifier`

        """
        return AMFLearner.partial_fit_helper(self, X, y)

    def _compute_predictions(self, X):
        n_samples, n_features = X.shape
        scores = np.zeros((n_samples, self.n_classes), dtype="float32")
        forest_classifier_predict_proba(self.no_python, X, scores)
        return scores

    def predict_proba(self, X):
        """Predicts the class probabilities for the given features vectors.

        Parameters
        ----------
        X : :obj:`np.ndarray`, shape=(n_samples, n_features)
            Input features matrix to predict for.

        Returns
        -------
        output : :obj:`np.ndarray`, shape=(n_samples, n_classes)
            Returns the predicted class probabilities for the input features

        """
        return AMFLearner.predict_helper(self, X)

    # TODO: put in AMFLearner and reorganize
    def predict_proba_tree(self, X, tree):
        """Predicts the class probabilities for the given features vectors using a
        single tree at given index ``tree``. Should be used only for debugging or
        visualisation purposes.
        
        Parameters
        ----------
        X : :obj:`np.ndarray`, shape=(n_samples, n_features)
            Input features matrix to predict for.

        tree : :obj:`int`
            Index of the tree, must be between 0 and ``n_estimators`` - 1

        Returns
        -------
        output : :obj:`np.ndarray`, shape=(n_samples, n_classes)
            Returns the predicted class probabilities for the input features

        """
        # TODO: unittests for this method
        if not self.no_python:
            raise RuntimeError(
                "You must call `partial_fit` before calling `predict_proba`"
            )
        else:
            X = check_array(
                X,
                accept_sparse=False,
                accept_large_sparse=False,
                dtype=["float32"],
                order="C",
                copy=False,
                force_all_finite=True,
                ensure_2d=True,
                allow_nd=False,
                ensure_min_samples=1,
                ensure_min_features=1,
                estimator="AMFClassifier",
            )
            n_samples, n_features = X.shape
            if n_features != self.n_features:
                raise ValueError(
                    "`partial_fit` was called with n_features=%d while `predict_proba` "
                    "received n_features=%d" % (self.n_features, n_features)
                )
            if not isinstance(tree, int):
                raise ValueError("`tree` must be of integer type")
            if tree < 0 or tree >= self.n_estimators:
                raise ValueError("`tree` must be between 0 and `n_estimators` - 1")

            self._set_random_state()
            scores = forest_classifier_predict_proba_tree(self.no_python, tree, X)
            self._put_back_random_state()
            return scores

    # def predict(self, X):
    #     if not self._fitted:
    #         raise RuntimeError("You must call ``fit`` before")
    #     else:
    #         X = safe_array(X, dtype='float32')
    #         scores = self.predict_proba(X)
    #         return scores.argmax(axis=1)

    def get_nodes_df(self, idx_tree):
        import pandas as pd

        tree = self.no_python.trees[idx_tree]
        nodes = tree.nodes
        n_nodes = nodes.n_nodes
        index = nodes.index[:n_nodes]
        parent = nodes.parent[:n_nodes]
        left = nodes.left[:n_nodes]
        right = nodes.right[:n_nodes]
        feature = nodes.feature[:n_nodes]
        threshold = nodes.threshold[:n_nodes]
        time = nodes.time[:n_nodes]
        depth = nodes.depth[:n_nodes]
        memory_range_min = nodes.memory_range_min[:n_nodes]
        memory_range_max = nodes.memory_range_max[:n_nodes]
        n_samples = nodes.n_samples[:n_nodes]
        weight = nodes.weight[:n_nodes]
        log_weight_tree = nodes.log_weight_tree[:n_nodes]
        is_leaf = nodes.is_leaf[:n_nodes]
        # is_memorized = nodes.is_memorized[:n_nodes]
        counts = nodes.counts[:n_nodes]

        columns = [
            "id",
            "parent",
            "left",
            "right",
            "depth",
            "is_leaf",
            "feature",
            "threshold",
            "time",
            "n_samples",
            "weight",
            "log_weight_tree",
            "memory_range_min",
            "memory_range_max",
            "counts",
        ]

        data = {
            "id": index,
            "parent": parent,
            "left": left,
            "right": right,
            "depth": depth,
            "feature": feature,
            "threshold": threshold,
            "is_leaf": is_leaf,
            "time": time,
            "n_samples": n_samples,
            "weight": weight,
            "log_weight_tree": log_weight_tree,
            "memory_range_min": [tuple(t) for t in memory_range_min],
            "memory_range_max": [tuple(t) for t in memory_range_max],
            "counts": [tuple(t) for t in counts],
        }
        df = pd.DataFrame(data, columns=columns)
        return df

    @property
    def n_classes(self):
        """:obj:`int`: Number of expected classes in the labels."""
        return self._n_classes

    @n_classes.setter
    def n_classes(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `n_classes` after calling `partial_fit`"
            )
        else:
            if not isinstance(val, int):
                raise ValueError("`n_classes` must be of type `int`")
            elif val < 2:
                raise ValueError("`n_classes` must be >= 2")
            else:
                self._n_classes = val

    @property
    def dirichlet(self):
        """:obj:`float` or :obj:`None`: Regularization level of the class
        frequencies."""
        return self._dirichlet

    @dirichlet.setter
    def dirichlet(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `dirichlet` after calling `partial_fit`"
            )
        else:
            if not isinstance(val, float):
                raise ValueError("`dirichlet` must be of type `float`")
            elif val <= 0:
                raise ValueError("`dirichlet` must be > 0")
            else:
                self._dirichlet = val

    @property
    def loss(self):
        """:obj:`str`: The loss used for the computation of the aggregation weights."""
        return "log"

    @loss.setter
    def loss(self, val):
        pass

    def __repr__(self):
        r = "AMFClassifier"
        r += "(n_classes={n_classes}, ".format(n_classes=repr(self.n_classes))
        r += "n_estimators={n_estimators}, ".format(
            n_estimators=repr(self.n_estimators)
        )
        r += "step={step}, ".format(step=repr(self.step))
        r += "loss={loss}, ".format(loss=repr(self.loss))
        r += "use_aggregation={use_aggregation}, ".format(
            use_aggregation=repr(self.use_aggregation)
        )
        r += "dirichlet={dirichlet}, ".format(dirichlet=repr(self.dirichlet))
        r += "split_pure={split_pure}, ".format(split_pure=repr(self.split_pure))
        r += "n_jobs={n_jobs}, ".format(n_jobs=repr(self.n_jobs))
        r += "random_state={random_state}, ".format(
            random_state=repr(self.random_state)
        )
        r += "verbose={verbose})".format(verbose=repr(self.verbose))
        return r


class AMFRegressor(AMFLearner):
    """Aggregated Mondrian Forest regressor for online learning. This algorithm
    is truly online, in the sense that a single pass is performed, and that predictions
    can be produced anytime.

    Each node in a tree predicts according to the average of the labels
    it contains. The prediction for a sample is computed as the aggregated predictions
    of all the subtrees along the path leading to the leaf node containing the sample.
    The aggregation weights are exponential weights with learning rate ``step`` and loss
    ``loss`` when ``use_aggregation`` is ``True``.

    This computation is performed exactly thanks to a context tree weighting algorithm.
    More details can be found in the paper cited in references below.

    The final predictions are the average of the predictions of each of the
    ``n_estimators`` trees in the forest.

    Note
    ----
    All the parameters of ``AMFRegressor`` become **read-only** after the first call
    to ``partial_fit``

    References
    ----------
    J. Mourtada, S. Gaiffas and E. Scornet, *AMF: Aggregated Mondrian Forests for Online Learning*, arXiv:1906.10529, 2019

    """

    def __init__(
        self,
        n_estimators=10,
        step=1.0,
        loss="least-squares",
        use_aggregation=True,
        split_pure=False,
        n_jobs=1,
        n_samples_increment=1024,
        random_state=None,
        verbose=False,
    ):
        """Instantiates a `AMFRegressor` instance.

        Parameters
        ----------
        n_estimators : :obj:`int`, default = 10
            The number of trees in the forest.

        step : :obj:`float`, default = 1
            Step-size for the aggregation weights. Default is ??? for regression with
            the least-squares loss.

        loss : {"least-squares"}, default = "least-squares"
            The loss used for the computation of the aggregation weights. Only
            "least-squares" is supported for now.

        use_aggregation : :obj:`bool`, default = `True`
            Controls if aggregation is used in the trees. It is highly recommended to
            leave it as `True`.

        split_pure : :obj:`bool`, default = `False`
            Controls if nodes that contains only sample of the same class should be
            split ("pure" nodes). Default is `False`, namely pure nodes are not split,
            but `True` can be sometimes better.

        n_jobs : :obj:`int`, default = 1
            Sets the number of threads used to grow the tree in parallel. The default is
            n_jobs=1, namely single-threaded. Fow now, this parameter has no effect and
            only a single thread can be used.

        n_samples_increment : :obj:`int`, default = 1024
            Sets the minimum amount of memory which is pre-allocated each time extra
            memory is required for new samples and new nodes. Decreasing it can slow
            down training. If you know that each ``partial_fit`` will be called with
            approximately `n` samples, you can set n_samples_increment = `n` if `n` is
            larger than the default.

        random_state : :obj:`int` or :obj:`None`, default = `None`
            Controls the randomness involved in the trees.

        verbose : :obj:`bool`, default = `False`
            Controls the verbosity when fitting and predicting.
        """
        AMFLearner.__init__(
            self,
            n_estimators=n_estimators,
            step=step,
            loss=loss,
            use_aggregation=use_aggregation,
            split_pure=split_pure,
            n_jobs=n_jobs,
            n_samples_increment=n_samples_increment,
            random_state=random_state,
            verbose=verbose,
        )

    def _instantiate_nopython_class(self):
        trees_iteration = np.empty(0, dtype=np.uint32)
        trees_n_nodes = np.empty(0, dtype=np.uint32)
        trees_n_nodes_capacity = np.empty(0, dtype=np.uint32)
        n_samples = 0
        n_samples_capacity = 0
        samples = SamplesCollection(
            self.n_samples_increment, self.n_features, n_samples, n_samples_capacity
        )
        self.no_python = AMFRegressorNoPython(
            self.n_features,
            self.n_estimators,
            self.step,
            self.loss,
            self.use_aggregation,
            self.split_pure,
            self.n_jobs,
            self.n_samples_increment,
            self.verbose,
            samples,
            trees_iteration,
            trees_n_nodes,
            trees_n_nodes_capacity,
        )

    def _to_dict(self):
        attrs = [
            "_n_features",
            "n_estimators",
            "step",
            "loss",
            "use_aggregation",
            "split_pure",
            "n_jobs",
            "n_samples_increment",
            "random_state",
            "verbose",
        ]
        d = {}
        for key in attrs:
            d[key] = getattr(self, key)
        d["no_python"] = amf_regressor_nopython_to_dict(self.no_python)
        return d

    @classmethod
    def _from_dict(cls, d):
        amf = AMFRegressor(
            n_estimators=d["n_estimators"],
            step=d["step"],
            loss=d["loss"],
            use_aggregation=d["use_aggregation"],
            split_pure=d["split_pure"],
            n_jobs=d["n_jobs"],
            n_samples_increment=d["n_samples_increment"],
            random_state=d["random_state"],
            verbose=d["verbose"],
        )
        amf._n_features = d["_n_features"]
        amf.no_python = dict_to_amf_regressor_nopython(d["no_python"])
        return amf

    def _partial_fit(self, X, y):
        forest_regressor_partial_fit(self.no_python, X, y)

    def partial_fit(self, X, y, classes=None):
        """Updates the classifier with the given batch of samples.

        Parameters
        ----------
        X : :obj:`np.ndarray`, shape=(n_samples, n_features)
            Input features matrix.

        y : :obj:`np.ndarray`
            Input labels vector.

        classes : :obj:`None`
            Must not be used, only here for backwards compatibility

        Returns
        -------
        output : :obj:`AMFRegressor`
            Updated instance of :obj:`AMFRegressor`

        """
        AMFLearner.partial_fit_helper(self, X, y)

    def _compute_predictions(self, X):
        n_samples, n_features = X.shape
        predictions = np.zeros(n_samples, dtype="float32")
        forest_regressor_predict(self.no_python, X, predictions)
        return predictions

    def _compute_weighted_depths(self, X):
        n_samples, n_features = X.shape
        weighted_depths = np.zeros((n_samples, self.n_estimators), dtype="float32")
        forest_regressor_weighted_depths(self.no_python, X, weighted_depths)
        return weighted_depths

    def predict(self, X):
        """Predicts the labels for the given features vectors.

        Parameters
        ----------
        X : :obj:`np.ndarray`, shape=(n_samples, n_features)
            Input features matrix to predict for.

        Returns
        -------
        output : :obj:`np.ndarray`, shape=(n_samples,)
            Returns the predicted labels for the input features

        """
        return AMFLearner.predict_helper(self, X)

    def weighted_depth(self, X):
        return AMFLearner.weighted_depth_helper(self, X)

    def get_nodes_df(self, idx_tree):
        import pandas as pd

        tree = self.no_python.trees[idx_tree]
        nodes = tree.nodes
        n_nodes = nodes.n_nodes
        index = nodes.index[:n_nodes]
        parent = nodes.parent[:n_nodes]
        left = nodes.left[:n_nodes]
        right = nodes.right[:n_nodes]
        feature = nodes.feature[:n_nodes]
        threshold = nodes.threshold[:n_nodes]
        time = nodes.time[:n_nodes]
        depth = nodes.depth[:n_nodes]
        memory_range_min = nodes.memory_range_min[:n_nodes]
        memory_range_max = nodes.memory_range_max[:n_nodes]
        n_samples = nodes.n_samples[:n_nodes]
        weight = nodes.weight[:n_nodes]
        log_weight_tree = nodes.log_weight_tree[:n_nodes]
        is_leaf = nodes.is_leaf[:n_nodes]
        # is_memorized = nodes.is_memorized[:n_nodes]
        counts = nodes.counts[:n_nodes]

        columns = [
            "id",
            "parent",
            "left",
            "right",
            "depth",
            "is_leaf",
            "feature",
            "threshold",
            "time",
            "n_samples",
            "weight",
            "log_weight_tree",
            "memory_range_min",
            "memory_range_max",
            "counts",
        ]

        data = {
            "id": index,
            "parent": parent,
            "left": left,
            "right": right,
            "depth": depth,
            "feature": feature,
            "threshold": threshold,
            "is_leaf": is_leaf,
            "time": time,
            "n_samples": n_samples,
            "weight": weight,
            "log_weight_tree": log_weight_tree,
            "memory_range_min": [tuple(t) for t in memory_range_min],
            "memory_range_max": [tuple(t) for t in memory_range_max],
            "counts": [tuple(t) for t in counts],
        }
        df = pd.DataFrame(data, columns=columns)
        return df

    @property
    def loss(self):
        """:obj:`str`: The loss used for the computation of the aggregation weights."""
        return "least-squares"

    @loss.setter
    def loss(self, val):
        pass

    def __repr__(self):
        r = "AMFRegressor("
        r += "n_estimators={n_estimators}, ".format(
            n_estimators=repr(self.n_estimators)
        )
        r += "step={step}, ".format(step=repr(self.step))
        r += "loss={loss}, ".format(loss=repr(self.loss))
        r += "use_aggregation={use_aggregation}, ".format(
            use_aggregation=repr(self.use_aggregation)
        )
        r += "split_pure={split_pure}, ".format(split_pure=repr(self.split_pure))
        r += "n_jobs={n_jobs}, ".format(n_jobs=repr(self.n_jobs))
        r += "random_state={random_state}, ".format(
            random_state=repr(self.random_state)
        )
        r += "verbose={verbose})".format(verbose=repr(self.verbose))
        return r
