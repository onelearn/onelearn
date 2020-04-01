# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
"""
Logo example
============

This is a small example that produces the logo of the ``onelearn`` library.
"""
import sys
import logging
import matplotlib.pyplot as plt
from tqdm import trange
from sklearn.datasets import make_moons
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.extend([".", ".."])
from onelearn import AMFClassifier
from experiments.plot import (
    get_mesh,
    plot_contour_binary_classif,
    plot_scatter_binary_classif,
)

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO
)

n_samples = 100
n_features = 2
n_classes = 2
random_state = 123
save_iterations = [5, 10, 30, 70]

logging.info("Simulation of the data")
X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=random_state)

logging.info("Train/Test splitting")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=random_state
)

logging.info("Computation of the meshgrid")
xx, yy, X_mesh = get_mesh(X)


clf = AMFClassifier(
    n_classes=n_classes,
    n_estimators=100,
    random_state=random_state,
    split_pure=True,
    use_aggregation=True,
)

n_plots = len(save_iterations)
n_fig = 0
save_iterations = [0, *save_iterations]

plt.figure(figsize=(3, 3))

logging.info("Launching iterations")
bar = trange(n_plots, desc="Plotting iterations", leave=True)

norm = plt.Normalize(vmin=0.0, vmax=1.0)

for start, end in zip(save_iterations[:-1], save_iterations[1:]):
    X_iter = X_train[start:end]
    y_iter = y_train[start:end]
    clf.partial_fit(X_iter, y_iter)
    n_fig += 1
    Z = clf.predict_proba(X_mesh)[:, 1].reshape(xx.shape)
    ax = plt.subplot(2, 2, n_fig)
    score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    plot_contour_binary_classif(ax, xx, yy, Z, levels=5, norm=norm)

    plot_scatter_binary_classif(
        ax, xx, yy, X_train[:end], y_train[:end], s=15, norm=norm
    )
    bar.update(1)

bar.close()
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("logo.png", transparent=True)
logging.info("Saved logo in file logo.png")
