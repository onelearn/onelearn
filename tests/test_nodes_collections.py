# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
from . import approx


class TestNodesCollection(object):
    def test_nodes_classifier(self):
        from onelearn.node_collection import (
            NodesClassifier,
            add_node_classifier,
            copy_node_classifier,
        )

        n_features = 3
        n_classes = 2
        n_samples_increment = 2

        nodes = NodesClassifier(n_features, n_classes, n_samples_increment)
        assert nodes.n_nodes == 0
        assert nodes.n_samples_increment == 2
        assert nodes.n_nodes_capacity == 5

        add_node_classifier(nodes, 0, 3.14)
        add_node_classifier(nodes, 0, 3.14)
        add_node_classifier(nodes, 0, 3.14)
        assert nodes.n_nodes == 3
        assert nodes.n_nodes_capacity == 5

        add_node_classifier(nodes, 0, 3.14)
        add_node_classifier(nodes, 0, 3.14)
        add_node_classifier(nodes, 13, 2.78)
        assert nodes.n_nodes == 6
        assert nodes.n_nodes_capacity == 10
        assert nodes.parent[4] == 0
        assert nodes.time[4] == approx(3.14, 1e-7)

        copy_node_classifier(nodes, 5, 4)
        assert nodes.parent[4] == 13
        assert nodes.time[4] == approx(2.78, 1e-7)

    def test_nodes_regressor(self):
        from onelearn.node_collection import (
            NodesRegressor,
            add_node_regressor,
            copy_node_regressor,
        )

        n_features = 3
        n_samples_increment = 2

        nodes = NodesRegressor(n_features, n_samples_increment)
        assert nodes.n_nodes == 0
        assert nodes.n_samples_increment == 2
        assert nodes.n_nodes_capacity == 5

        add_node_regressor(nodes, 0, 3.14)
        add_node_regressor(nodes, 0, 3.14)
        add_node_regressor(nodes, 0, 3.14)
        assert nodes.n_nodes == 3
        assert nodes.n_nodes_capacity == 5

        add_node_regressor(nodes, 0, 3.14)
        add_node_regressor(nodes, 0, 3.14)
        add_node_regressor(nodes, 13, 2.78)
        assert nodes.n_nodes == 6
        assert nodes.n_nodes_capacity == 10
        assert nodes.parent[4] == 0
        assert nodes.time[4] == approx(3.14, 1e-7)

        copy_node_regressor(nodes, 5, 4)
        assert nodes.parent[4] == 13
        assert nodes.time[4] == approx(2.78, 1e-7)
