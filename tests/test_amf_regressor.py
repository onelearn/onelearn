# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

# py.test -rA

import numpy as np
import pytest
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from onelearn import AMFRegressor
from . import parameter_test_with_min, parameter_test_with_type, approx
from onelearn.datasets import make_regression


class TestAMFRegressor(object):
    def test_n_features(self):
        clf = AMFRegressor()
        X = np.random.randn(2, 2)
        y = np.array([0.0, 1.0])
        clf.partial_fit(X, y)
        assert clf.n_features == 2
        with pytest.raises(ValueError, match="`n_features` is a readonly attribute"):
            clf.n_features = 3

    def test_n_estimators(self):
        parameter_test_with_min(
            AMFRegressor,
            parameter="n_estimators",
            valid_val=3,
            invalid_type_val=2.0,
            invalid_val=0,
            min_value=1,
            min_value_str="1",
            mandatory=False,
            fixed_type=int,
        )

    def test_step(self):
        parameter_test_with_min(
            AMFRegressor,
            parameter="step",
            valid_val=2.0,
            invalid_type_val=0,
            invalid_val=0.0,
            min_value_strict=0.0,
            min_value_str="0",
            mandatory=False,
            fixed_type=float,
        )

    def test_loss(self):
        amf = AMFRegressor()
        assert amf.loss == "least-squares"
        amf.loss = "other loss"
        assert amf.loss == "least-squares"

    def test_use_aggregation(self):
        parameter_test_with_type(
            AMFRegressor,
            parameter="step",
            valid_val=False,
            invalid_type_val=0,
            mandatory=False,
            fixed_type=bool,
        )

    def test_split_pure(self):
        parameter_test_with_type(
            AMFRegressor,
            parameter="split_pure",
            valid_val=False,
            invalid_type_val=0,
            mandatory=False,
            fixed_type=bool,
        )

    def test_random_state(self):
        parameter_test_with_min(
            AMFRegressor,
            parameter="random_state",
            valid_val=4,
            invalid_type_val=2.0,
            invalid_val=-1,
            min_value=0,
            min_value_str="0",
            mandatory=False,
            fixed_type=int,
        )
        amf = AMFRegressor()
        assert amf.random_state is None
        assert amf._random_state == -1
        amf.random_state = 1
        amf.random_state = None
        assert amf._random_state == -1

    def test_n_jobs(self):
        parameter_test_with_min(
            AMFRegressor,
            parameter="n_jobs",
            valid_val=4,
            invalid_type_val=2.0,
            invalid_val=0,
            min_value=1,
            min_value_str="1",
            mandatory=False,
            fixed_type=int,
        )

    def test_n_samples_increment(self):
        parameter_test_with_min(
            AMFRegressor,
            parameter="n_samples_increment",
            valid_val=128,
            invalid_type_val=2.0,
            invalid_val=0,
            min_value=1,
            min_value_str="1",
            mandatory=False,
            fixed_type=int,
        )

    def test_verbose(self):
        parameter_test_with_type(
            AMFRegressor,
            parameter="verbose",
            valid_val=False,
            invalid_type_val=0,
            mandatory=False,
            fixed_type=bool,
        )

    def test_repr(self):
        amf = AMFRegressor()
        assert (
            repr(amf) == "AMFRegressor(n_estimators=10, step=1.0, "
            "loss='least-squares', use_aggregation=True, split_pure=False, n_jobs=1, "
            "random_state=None, verbose=False)"
        )

        amf.n_estimators = 42
        assert (
            repr(amf) == "AMFRegressor(n_estimators=42, step=1.0, "
            "loss='least-squares', use_aggregation=True, "
            "split_pure=False, n_jobs=1, random_state=None, verbose=False)"
        )

        amf.verbose = False
        assert (
            repr(amf) == "AMFRegressor(n_estimators=42, "
            "step=1.0, loss='least-squares', use_aggregation=True, "
            "split_pure=False, n_jobs=1, random_state=None, verbose=False)"
        )

    def test_partial_fit(self):
        clf = AMFRegressor()
        n_features = 4
        X = np.random.randn(2, n_features)
        y = np.array([0.0, 1.0])
        clf.partial_fit(X, y)
        assert clf.n_features == n_features
        assert clf.no_python.iteration == 2
        assert clf.no_python.samples.n_samples == 2
        assert clf.no_python.n_features == n_features

        with pytest.raises(ValueError) as exc_info:
            X = np.random.randn(2, 3)
            y = np.array([0.0, 1.0])
            clf.partial_fit(X, y)
        assert exc_info.type is ValueError
        assert (
            exc_info.value.args[0] == "`partial_fit` was first called with "
            "n_features=4 while n_features=3 in this call"
        )

    def test_predict(self):
        clf = AMFRegressor()
        with pytest.raises(
            RuntimeError,
            match="You must call `partial_fit` before asking for predictions",
        ):
            X_test = np.random.randn(2, 3)
            clf.predict(X_test)

        with pytest.raises(ValueError) as exc_info:
            X = np.random.randn(2, 2)
            y = np.array([0.0, 1.0])
            clf.partial_fit(X, y)
            X_test = np.random.randn(2, 3)
            clf.predict(X_test)
        assert exc_info.type is ValueError
        assert exc_info.value.args[
            0
        ] == "`partial_fit` was called with n_features=%d while predictions are asked with n_features=%d" % (
            clf.n_features,
            3,
        )

    def test_performance_on_blocks(self):
        n_samples = 2000
        random_state = 42
        X, y = make_regression(
            n_samples=n_samples, signal="blocks", random_state=random_state
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=random_state
        )
        clf = AMFRegressor(random_state=random_state)
        clf.partial_fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        err = np.abs(y_test - y_pred).mean()
        # With this random_state, err should be exactly 0.07848953956518727
        assert err < 0.08

    def test_random_state_is_consistant(self):
        n_samples = 300
        random_state = 42
        X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=random_state
        )

        clf = AMFRegressor(random_state=random_state)
        clf.partial_fit(X_train, y_train)
        y_pred_1 = clf.predict(X_test)

        clf = AMFRegressor(random_state=random_state)
        clf.partial_fit(X_train, y_train)
        y_pred_2 = clf.predict(X_test)

        assert y_pred_1 == approx(y_pred_2)

    def test_weighted_depth(self):
        n_samples = 2000
        random_state = 42
        noise = 0.03
        use_aggregation = True
        split_pure = True
        n_estimators = 100
        step = 10.0

        X, y = make_regression(
            n_samples=n_samples,
            signal="blocks",
            noise=noise,
            random_state=random_state,
        )

        amf = AMFRegressor(
            random_state=random_state,
            use_aggregation=use_aggregation,
            n_estimators=n_estimators,
            split_pure=split_pure,
            step=step,
        )

        amf.partial_fit(X.reshape(n_samples, 1), y)
        X_test = np.array(
            [
                0.5,
                0.1,
                0.13,
                0.15,
                0.2,
                0.23,
                0.25,
                0.3,
                0.4,
                0.44,
                0.55,
                0.65,
                0.7,
                0.76,
                0.78,
                0.81,
                0.9,
            ]
        ).reshape(-1, 1)
        weighted_depth = amf.weighted_depth(X_test).mean(axis=1)
        weighted_depth_expected = np.array(
            [
                6.8916187,
                11.966855,
                12.637342,
                10.947144,
                7.8675694,
                12.680364,
                12.640211,
                7.3447876,
                12.585537,
                10.4313545,
                6.458374,
                13.874192,
                7.446446,
                11.255474,
                10.093488,
                12.541043,
                4.807294,
            ]
        )
        assert weighted_depth == approx(weighted_depth_expected, 1e-5)
