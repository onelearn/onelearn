# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import numpy as np


class OnlineDummyClassifier(object):
    """A dummy online classifier only using past frequencies of the labels.
    Namely, predictions don't use the features and simply compute

        (count + dirichlet) / (n_samples + dirichlet * n_classes)
    
    for each class, where count is the count for the class, and where ``dirichlet`` is a
    "smoothing" parameter. This is simply a regularized class frequency with a
    dirichlet prior with ``dirichlet`` parameter

    Note
    ----
    This class cannot produce serious predictions, and must only be used as a dummy
    baseline.

    """

    def __init__(self, n_classes, dirichlet=None):
        """Instantiates a `OnlineDummyClassifier` instance.

        Parameters
        ----------
        n_classes : :obj:`int`
            Number of expected classes in the labels. This is required since we
            don't know the number of classes in advance in a online setting.

        dirichlet : :obj:`float` or :obj:`None`, default = `None`
            Regularization level of the class frequencies used for predictions in each
            node. Default is dirichlet=0.5 for n_classes=2 and dirichlet=0.01 otherwise.

        """
        self.iteration = 0
        self.n_classes = n_classes
        self._classes = set(range(n_classes))
        if dirichlet is None:
            if self.n_classes == 2:
                self.dirichlet = 0.5
            else:
                self.dirichlet = 0.01
        else:
            self.dirichlet = dirichlet
        self.counts = np.zeros((n_classes,), dtype=np.uint32)

    def partial_fit(self, X, y, classes=None):
        """Updates the classifier with the given batch of samples.

        Parameters
        ----------
        X : :obj:`np.ndarray` or :obj:`scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
            Input features matrix.

        y : :obj:`np.ndarray`
            Input labels vector.

        classes : :obj:`None`
            Must not be used, only here for backwards compatibility

        Returns
        -------
        output : :obj:`OnlineDummyClassifier`
            Updated instance of `OnlineDummyClassifier`

        """
        if y.min() < 0:
            raise ValueError("All the values in `y` must be non-negative")
        y_max = y.max()
        if y_max not in self._classes:
            raise ValueError("n_classes=%d while y.max()=%d" % (self.n_classes, y_max))

        for yi in y:
            self.counts[int(yi)] += 1
            self.iteration += 1
        return self

    def predict_proba(self, X):
        """Predicts the class probabilities for the given features vectors

        Parameters
        ----------
        X : :obj:`np.ndarray` or :obj:`scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
            Input features matrix to predict for.

        Returns
        -------
        output : :obj:`np.ndarray`, shape=(n_samples, n_classes)
            Returns the predicted class probabilities for the input features

        """
        if self.iteration == 0:
            raise RuntimeError(
                "You must call `partial_fit` before calling `predict_proba`"
            )

        n_samples = X.shape[0]
        dirichet = self.dirichlet
        n_classes = self.n_classes
        scores = (self.counts + dirichet) / (self.iteration + n_classes * dirichet)
        # scores = self.counts / self.n_samples
        probas = np.tile(scores, reps=(n_samples, 1))
        return probas

    def predict(self, X):
        """Predicts the labels for the given features vectors

        Parameters
        ----------
        X : :obj:`np.ndarray` or :obj:`scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
            Input features matrix to predict for.

        Returns
        -------
        output : :obj:`np.ndarray`, shape=(n_samples,)
            Returns the predicted labels for the input features

        """
        scores = self.predict_proba(X)
        return scores.argmax(axis=1)

    @property
    def n_classes(self):
        """:obj:`int`: Number of expected classes in the labels."""
        return self._n_classes

    @n_classes.setter
    def n_classes(self, val):
        if self.iteration > 0:
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
        if self.iteration > 0:
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

    def __repr__(self):
        r = "OnlineDummyClassifier"
        r += "(n_classes={n_classes}, ".format(n_classes=self.n_classes)
        r += "dirichlet={dirichlet})".format(dirichlet=self.dirichlet)
        return r
