# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

import numpy as np
import pytest
from onelearn import OnlineDummyClassifier
from . import parameter_test_with_min


class TestOnlineDummyClassifier(object):
    def test_n_classes(self):
        parameter_test_with_min(
            OnlineDummyClassifier,
            parameter="n_classes",
            valid_val=3,
            invalid_type_val=2.0,
            invalid_val=1,
            min_value=2,
            min_value_str="2",
            mandatory=True,
            fixed_type=int,
        )

    def test_dirichlet(self):
        parameter_test_with_min(
            OnlineDummyClassifier,
            parameter="dirichlet",
            valid_val=0.1,
            invalid_type_val=0,
            invalid_val=0.0,
            min_value_strict=0.0,
            min_value_str="0",
            mandatory=False,
            fixed_type=float,
        )

    def test_repr(self):
        dummy = OnlineDummyClassifier(n_classes=3)
        print(repr(dummy))
        assert repr(dummy) == "OnlineDummyClassifier(n_classes=3, dirichlet=0.5)"

    def test_partial_fit(self):
        clf = OnlineDummyClassifier(n_classes=2)
        n_features = 4
        X = np.random.randn(2, n_features)
        y = np.array([0.0, 1.0])
        clf.partial_fit(X, y)
        assert clf.iteration == 2

        with pytest.raises(
            ValueError, match="All the values in `y` must be non-negative",
        ):
            clf = OnlineDummyClassifier(n_classes=2)
            X = np.random.randn(2, n_features)
            y = np.array([0.0, -1.0])
            clf.partial_fit(X, y)

        with pytest.raises(ValueError) as exc_info:
            clf = OnlineDummyClassifier(n_classes=2)
            X = np.random.randn(2, 3)
            y = np.array([0.0, 2.0])
            clf.partial_fit(X, y)
        assert exc_info.type is ValueError
        assert exc_info.value.args[0] == "n_classes=2 while y.max()=2"

    def test_predict_proba(self):
        clf = OnlineDummyClassifier(n_classes=2)
        with pytest.raises(
            RuntimeError,
            match="You must call `partial_fit` before calling `predict_proba`",
        ):
            X_test = np.random.randn(2, 3)
            clf.predict_proba(X_test)
