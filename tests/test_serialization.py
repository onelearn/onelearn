# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

# py.test -rA

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os

from onelearn import AMFClassifier, AMFRegressor


def test_amf_classifier_serialization():
    """Trains a AMFClassifier on iris, saves and loads it again. Check that
    everything is the same between the original and loaded forest
    """
    random_state = 42
    n_estimators = 1
    n_classes = 3

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )
    clf1 = AMFClassifier(
        n_estimators=n_estimators, n_classes=n_classes, random_state=random_state
    )
    clf1.partial_fit(X_train_1, y_train_1)

    filename = "amf_on_iris.pkl"
    clf1.save(filename)
    clf2 = AMFClassifier.load(filename)
    os.remove(filename)

    def test_forests_are_equal(clf1, clf2):
        # Test samples
        samples1 = clf1.no_python.samples
        samples2 = clf2.no_python.samples
        assert samples1.n_samples_increment == samples2.n_samples_increment
        n_samples1 = samples1.n_samples
        n_samples2 = samples2.n_samples
        assert n_samples1 == n_samples2
        assert samples1.n_samples_capacity == samples2.n_samples_capacity
        assert np.all(samples1.labels[:n_samples1] == samples2.labels[:n_samples2])
        assert np.all(samples1.features[:n_samples1] == samples2.features[:n_samples2])

        # Test nopython.trees
        for n_estimator in range(n_estimators):
            tree1 = clf1.no_python.trees[n_estimator]
            tree2 = clf2.no_python.trees[n_estimator]
            # Test tree attributes
            assert tree1.n_features == tree2.n_features
            assert tree1.step == tree2.step
            assert tree1.loss == tree2.loss
            assert tree1.use_aggregation == tree2.use_aggregation
            assert tree1.iteration == tree2.iteration
            assert tree1.n_classes == tree2.n_classes
            assert tree1.dirichlet == tree2.dirichlet
            assert np.all(tree1.intensities == tree2.intensities)
            # Test tree.nodes
            nodes1 = tree1.nodes
            nodes2 = tree2.nodes
            assert np.all(nodes1.index == nodes2.index)
            assert np.all(nodes1.is_leaf == nodes2.is_leaf)
            assert np.all(nodes1.depth == nodes2.depth)
            assert np.all(nodes1.n_samples == nodes2.n_samples)
            assert np.all(nodes1.parent == nodes2.parent)
            assert np.all(nodes1.left == nodes2.left)
            assert np.all(nodes1.right == nodes2.right)
            assert np.all(nodes1.feature == nodes2.feature)
            assert np.all(nodes1.weight == nodes2.weight)
            assert np.all(nodes1.log_weight_tree == nodes2.log_weight_tree)
            assert np.all(nodes1.threshold == nodes2.threshold)
            assert np.all(nodes1.time == nodes2.time)
            assert np.all(nodes1.memory_range_min == nodes2.memory_range_min)
            assert np.all(nodes1.memory_range_max == nodes2.memory_range_max)
            assert np.all(nodes1.n_features == nodes2.n_features)
            assert nodes1.n_nodes == nodes2.n_nodes
            assert nodes1.n_samples_increment == nodes2.n_samples_increment
            assert nodes1.n_nodes_capacity == nodes2.n_nodes_capacity
            assert np.all(nodes1.counts == nodes2.counts)
            assert nodes1.n_classes == nodes2.n_classes

    test_forests_are_equal(clf1, clf2)

    # Test predict proba
    y_pred = clf1.predict_proba(X_test)
    y_pred_pkl = clf2.predict_proba(X_test)
    assert np.all(y_pred == y_pred_pkl)

    clf1.partial_fit(X_train_2, y_train_2)
    clf2.partial_fit(X_train_2, y_train_2)
    test_forests_are_equal(clf1, clf2)

    y_pred = clf1.predict_proba(X_test)
    y_pred_pkl = clf2.predict_proba(X_test)
    assert np.all(y_pred == y_pred_pkl)


def test_amf_regressor_serialization():
    """Trains a AMFRegressor on diabetes, saves and loads it again. Check that
    everything is the same between the original and loaded forest
    """
    random_state = 42
    n_estimators = 1
    iris = datasets.load_diabetes()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )
    clf1 = AMFRegressor(n_estimators=n_estimators, random_state=random_state)
    clf1.partial_fit(X_train_1, y_train_1)

    filename = "amf_on_diabetes.pkl"
    clf1.save(filename)
    clf2 = AMFRegressor.load(filename)
    os.remove(filename)

    def test_forests_are_equal(clf1, clf2):
        # Test samples
        samples1 = clf1.no_python.samples
        samples2 = clf2.no_python.samples
        assert samples1.n_samples_increment == samples2.n_samples_increment
        n_samples1 = samples1.n_samples
        n_samples2 = samples2.n_samples
        assert n_samples1 == n_samples2
        assert samples1.n_samples_capacity == samples2.n_samples_capacity
        assert np.all(samples1.labels[:n_samples1] == samples2.labels[:n_samples2])
        assert np.all(samples1.features[:n_samples1] == samples2.features[:n_samples2])

        # Test nopython.trees
        for n_estimator in range(n_estimators):
            tree1 = clf1.no_python.trees[n_estimator]
            tree2 = clf2.no_python.trees[n_estimator]
            # Test tree attributes
            assert tree1.n_features == tree2.n_features
            assert tree1.step == tree2.step
            assert tree1.loss == tree2.loss
            assert tree1.use_aggregation == tree2.use_aggregation
            assert tree1.iteration == tree2.iteration
            assert np.all(tree1.intensities == tree2.intensities)
            # Test tree.nodes
            nodes1 = tree1.nodes
            nodes2 = tree2.nodes
            assert np.all(nodes1.index == nodes2.index)
            assert np.all(nodes1.is_leaf == nodes2.is_leaf)
            assert np.all(nodes1.depth == nodes2.depth)
            assert np.all(nodes1.n_samples == nodes2.n_samples)
            assert np.all(nodes1.parent == nodes2.parent)
            assert np.all(nodes1.left == nodes2.left)
            assert np.all(nodes1.right == nodes2.right)
            assert np.all(nodes1.feature == nodes2.feature)
            assert np.all(nodes1.weight == nodes2.weight)
            assert np.all(nodes1.log_weight_tree == nodes2.log_weight_tree)
            assert np.all(nodes1.threshold == nodes2.threshold)
            assert np.all(nodes1.time == nodes2.time)
            assert np.all(nodes1.memory_range_min == nodes2.memory_range_min)
            assert np.all(nodes1.memory_range_max == nodes2.memory_range_max)
            assert np.all(nodes1.n_features == nodes2.n_features)
            assert nodes1.n_nodes == nodes2.n_nodes
            assert nodes1.n_samples_increment == nodes2.n_samples_increment
            assert nodes1.n_nodes_capacity == nodes2.n_nodes_capacity
            assert np.all(nodes1.mean == nodes2.mean)

    test_forests_are_equal(clf1, clf2)

    # Test predict
    y_pred = clf1.predict(X_test)
    y_pred_pkl = clf2.predict(X_test)
    assert np.all(y_pred == y_pred_pkl)

    clf1.partial_fit(X_train_2, y_train_2)
    clf2.partial_fit(X_train_2, y_train_2)
    test_forests_are_equal(clf1, clf2)

    y_pred = clf1.predict(X_test)
    y_pred_pkl = clf2.predict(X_test)
    assert np.all(y_pred == y_pred_pkl)
