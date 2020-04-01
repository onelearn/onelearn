# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
"""
Illustration of the forest effect
=================================

In this example we show that the decision function of a forest is the average of
independent trees, and that averaging allows to produce smooth decision functions.
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import logging
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


sys.path.extend([".", ".."])
from onelearn import AMFClassifier
from experiments.plot import (
    plot_contour_binary_classif,
    plot_scatter_binary_classif,
    get_mesh,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def plot_forest_effect(forest, dataset):
    n_estimators = forest.n_estimators
    _ = plt.figure(figsize=(2 * (n_estimators / 2 + 1), 4))

    X, y = dataset
    xx, yy, X_mesh = get_mesh(X)

    # Plot the training points
    ax = plt.subplot(2, n_estimators / 2 + 1, 1)
    plot_scatter_binary_classif(ax, xx, yy, X, y, title="Input data")

    forest.partial_fit(X, y)

    for idx_tree in range(n_estimators):
        ax = plt.subplot(2, n_estimators / 2 + 1, idx_tree + 2)
        Z = forest.predict_proba_tree(X_mesh, idx_tree)[:, 1].reshape(xx.shape)
        plot_contour_binary_classif(
            ax, xx, yy, Z, title="Tree #%d" % (idx_tree + 1),
        )

    ax = plt.subplot(2, n_estimators / 2 + 1, n_estimators + 2)
    Z = forest.predict_proba(X_mesh)[:, 1].reshape(xx.shape)
    plot_contour_binary_classif(ax, xx, yy, Z, title="Forest")
    plt.tight_layout()


n_samples = 200
n_features = 2
n_classes = 2
random_state = 42
dataset = make_moons(n_samples=n_samples, noise=0.3, random_state=random_state)

n_estimators = 10
amf = AMFClassifier(
    n_classes=n_classes,
    n_estimators=n_estimators,
    random_state=random_state,
    use_aggregation=True,
    split_pure=True,
)

logging.info("Building the graph...")
plot_forest_effect(amf, dataset)

plt.savefig("forest_effect.pdf")
logging.info("Saved the forest effect plot in forest_effect.pdf")
