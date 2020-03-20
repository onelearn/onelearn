# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import numpy as np


class OnlineDummyClassifier(object):
    """A dummy online classifier only using past frequencies of the labels.

    Parameters
    ----------
    n_classes : `int`
        Number of excepted classes in the labels. This is required since we
        don't know the number of classes in advance in a online setting.

    dirichlet : `float` or `None`, optional (default=None)
        Each node in a tree predicts according to the distribution of the labels
        it contains. This distribution is regularized using a "Jeffreys" prior
        with parameter `dirichlet`. For each class with `count` labels in the
        node and `n_samples` samples in it, the prediction of a node is given by
        (count + dirichlet) / (n_samples + dirichlet * n_classes).

        Default is dirichlet=0.5 for n_classes=2 and dirichlet=0.01 otherwise.

    Attributes
    ----------
    iteration : `int`
        Number of iterations performed
    """

    def __init__(self, n_classes, dirichlet=0.5):
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

    def partial_fit(self, _, y, classes=None):
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
        """Predict the class probabilities class for the given features

        Parameters
        ----------
        X : `np.ndarray` or `scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
            Features matrix to predict for.

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples, n_classes)
            Returns the predicted class probabilities
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
        scores = self.predict_proba(X)
        return scores.argmax(axis=1)

    @property
    def n_classes(self):
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
