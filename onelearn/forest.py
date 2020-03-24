# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import os
import numpy as np
from numba import jitclass, njit
from numba import types, _helperlib
from .types import float32, boolean, uint32, string, void, get_array_2d_type
from .checks import check_X_y, check_array
from .sample import SamplesCollection, add_samples
from .tree import TreeClassifier
from .tree_methods import tree_partial_fit, tree_predict
from .utils import get_type

spec = [
    ("n_classes", uint32),
    ("n_features", uint32),
    ("n_estimators", uint32),
    ("step", float32),
    ("loss", string),
    ("use_aggregation", boolean),
    ("dirichlet", float32),
    ("split_pure", boolean),
    ("n_jobs", uint32),
    ("reserve_samples", uint32),
    ("verbose", boolean),
    ("trees", types.List(get_type(TreeClassifier), reflected=True)),
    ("samples", get_type(SamplesCollection)),
    ("iteration", uint32),
]


# TODO: we can force pre-compilation when creating the nopython forest


@jitclass(spec)
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
        reserve_samples,
        verbose,
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
        self.reserve_samples = reserve_samples
        self.verbose = verbose
        self.iteration = 0

        samples = SamplesCollection(self.reserve_samples)
        self.samples = samples

        # TODO: reflected lists will be replaced by typed list soon...
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
            )
            for _ in range(n_estimators)
        ]
        self.trees = trees


@njit(void(get_type(AMFClassifierNoPython), get_array_2d_type(float32), float32[::1],))
def partial_fit(forest, X, y):
    n_samples_batch, n_features = X.shape
    # First, we save the new batch of data
    n_samples_before = forest.samples.n_samples
    # Add the samples in the forest
    add_samples(forest.samples, X, y)
    for i in range(n_samples_before, n_samples_before + n_samples_batch):
        # Then we fit all the trees using all new samples
        for tree in forest.trees:
            tree_partial_fit(tree, i)
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
def predict_proba(forest, X, scores):
    # TODO: use predict_proba_tree from below ? Or put it in the tree ?
    scores.fill(0.0)
    n_samples_batch, _ = X.shape

    scores_tree = np.empty(forest.n_classes, float32)
    for i in range(n_samples_batch):
        # print('i:', i)
        scores_i = scores[i]
        x_i = X[i]
        # print('x_i:', x_i)
        # The prediction is simply the average of the predictions
        for tree in forest.trees:
            tree_predict(tree, x_i, scores_tree, forest.use_aggregation)
            # print('scores_tree:', scores_tree)
            scores_i += scores_tree
        scores_i /= forest.n_estimators
        # print('scores_i:', scores_i)


@njit(
    get_array_2d_type(float32)(
        get_type(AMFClassifierNoPython), uint32, get_array_2d_type(float32)
    )
)
def predict_proba_tree(forest, idx_tree, X):
    n_samples_batch, _ = X.shape
    scores = np.empty((n_samples_batch, forest.n_classes), dtype=float32)
    tree = forest.trees[idx_tree]
    for i in range(n_samples_batch):
        scores_i = scores[i]
        x_i = X[i]
        tree_predict(tree, x_i, scores_i, forest.use_aggregation)
    return scores


class AMFClassifier(object):
    """Aggregated Mondrian Forest classifier for online learning. This algorithm
    is truly online, in the sense that a single pass is performed, and one can
    ask for predictions anytime.

    Parameters
    ----------
    n_classes : `int`
        Number of excepted classes in the labels. This is required since we
        don't know the number of classes in advance in a online setting.

    n_estimators : `int`, optional (default=10)
        Number of trees to grow in the forest.

    step : `float`, optional (default=1)
        Step-size for the aggregation weights. Default is 1 for classification,
        which is typically the best choice.

    loss : 'str', optional (default='log')
        The loss used for the computation of the aggregation weights. Only `log`
        is supported for now, namely the log-loss for multi-class
        classification.

    use_aggregation : `bool`, optional (default=True)
        Whether to use aggregation in each tree. It is highly recommended to
        leave it as `True`.

    dirichlet : `float` or `None`, optional (default=None)
        Each node in a tree predicts according to the distribution of the labels
        it contains. This distribution is regularized using a "Jeffreys" prior
        with parameter `dirichlet`. For each class with `count` labels in the
        node and `n_samples` samples in it, the prediction of a node is given by
        (count + dirichlet) / (n_samples + dirichlet * n_classes).

        Default is dirichlet=0.5 for n_classes=2 and dirichlet=0.01 otherwise.

    split_pure : `bool`, optional (default=False)
        Whether we split nodes that contain only one class of labels. Default is
        False.

    n_jobs : `int`, optional (default=1)
        The number of threads used to grow the tree in parallel. This has no
        effect for now, the default is n_jobs=1, namely single-threaded.

    reserve_samples : `int`, optional (default=1024)
        Each time extra memory is required for new samples and new nodes, pre-allocate
        what is required for `reserve_samples` in advance.

    random_state : `int` or `None`, optional (default=None)
        Controls the randomness involved in the trees growing, which is a highly
        randomized process.

    verbose : `bool`, optional (default=True)
        Whether a bar showing progress for `partial_fit`, `predict_proba` and
        `predict` should be used.

    Attributes
    ----------
    n_classes = `int`
        Number of excepted classes in the labels.

    n_features : `int`
        The number of features from the training dataset (passed to ``fit``)

    # TODO: add missing attributes

    Notes
    -----
    Note that all the parameters of AMFClassifier become read-only after the
    first call to `partial_fit`

    References
    ----------
    TODO : add the reference of the paper
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
        reserve_samples=1024,
        random_state=None,
        verbose: bool = True,
    ):
        # We will instantiate the numba class when data is passed to
        # `partial_fit`, since we need to know about `n_features` among others things
        self.no_python = None
        self.n_classes = n_classes
        self._n_features = None
        self.n_estimators = n_estimators
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation

        if dirichlet is None:
            if self.n_classes == 2:
                self.dirichlet = 0.5
            else:
                self.dirichlet = 0.01
        else:
            self.dirichlet = dirichlet

        self.split_pure = split_pure
        self.n_jobs = n_jobs
        self.reserve_samples = reserve_samples
        self.random_state = random_state
        self.verbose = verbose
        self._classes = set(range(n_classes))

        if os.getenv("NUMBA_DISABLE_JIT", None) == "1":
            self._using_numba = False
        else:
            self._using_numba = True

    def partial_fit(self, X, y, classes=None):
        # TODO: write the docstring
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
            estimator="AMFClassifier",
        )
        n_samples, n_features = X.shape

        if y.min() < 0:
            raise ValueError("All the values in `y` must be non-negative")
        y_max = y.max()
        if y_max not in self._classes:
            raise ValueError("n_classes=%d while y.max()=%d" % (self.n_classes, y_max))

        # This is the first call to `partial_fit`, so we need to instantiate
        # the no python class
        if self.no_python is None:
            self._n_features = n_features
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
                self.reserve_samples,
                self.verbose,
            )
        else:
            _, n_features = X.shape
            if n_features != self.n_features:
                raise ValueError(
                    "`partial_fit` was first called with n_features=%d while "
                    "n_features=%d in this call" % (self.n_features, n_features)
                )
        self._set_random_state()
        partial_fit(self.no_python, X, y)
        self._put_back_random_state()
        return self

    def predict_proba(self, X):
        """Predict class for given samples

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_features)
            Features matrix to predict for

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples, n_classes)
            Returns the predicted class probabilities
        """
        import numpy as np

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
        scores = np.zeros((n_samples, self.n_classes), dtype="float32")
        if not self.no_python:
            raise RuntimeError(
                "You must call `partial_fit` before calling `predict_proba`"
            )
        else:
            if n_features != self.n_features:
                raise ValueError(
                    "`partial_fit` was called with n_features=%d while `predict_proba` "
                    "received n_features=%d" % (self.n_features, n_features)
                )
        self._set_random_state()
        predict_proba(self.no_python, X, scores)
        self._put_back_random_state()
        return scores

    def predict_proba_tree(self, X, tree):
        """Predict class for given samples using a single tree. Mainly useful  for
        debugging or visualisation purposes

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_features)
            Features matrix to predict for

        tree : `int`
            Number of the tree, must be between 0 and `n_estimators` - 1

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples, n_classes)
            Returns the predicted class probabilities
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
            scores = predict_proba_tree(self.no_python, tree, X)
            self._put_back_random_state()
            return scores

    # def predict(self, X):
    #     if not self._fitted:
    #         raise RuntimeError("You must call ``fit`` before")
    #     else:
    #         X = safe_array(X, dtype='float32')
    #         scores = self.predict_proba(X)
    #         return scores.argmax(axis=1)

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
    def n_classes(self):
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
    def n_features(self):
        return self._n_features

    @n_features.setter
    def n_features(self, val):
        raise ValueError("`n_features` is a readonly attribute")

    @property
    def n_estimators(self):
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
    def reserve_samples(self):
        return self._reserve_samples

    @reserve_samples.setter
    def reserve_samples(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `reserve_samples` after calling `partial_fit`"
            )
        else:
            if not isinstance(val, int):
                raise ValueError("`reserve_samples` must be of type `int`")
            elif val < 1:
                raise ValueError("`reserve_samples` must be >= 1")
            else:
                self._reserve_samples = val

    @property
    def step(self):
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
    def dirichlet(self):
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
    def split_pure(self):
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
        return "log"

    @loss.setter
    def loss(self, val):
        pass

    @property
    def random_state(self):
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
        r = "AMFClassifier"
        r += "(n_classes={n_classes}, ".format(n_classes=self.n_classes)
        r += "n_estimators={n_estimators}, ".format(n_estimators=self.n_estimators)
        r += "step={step}, ".format(step=self.step)
        r += "loss={loss}, ".format(loss=self.loss)
        r += "use_aggregation={use_aggregation}, ".format(
            use_aggregation=self.use_aggregation
        )
        r += "dirichlet={dirichlet}, ".format(dirichlet=self.dirichlet)
        r += "split_pure={split_pure}, ".format(split_pure=self.split_pure)
        r += "n_jobs={n_jobs}, ".format(n_jobs=self.n_jobs)
        r += "random_state={random_state}, ".format(random_state=self.random_state)
        r += "verbose={verbose})".format(verbose=self.verbose)
        return r
