# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import trange

from onelearn import AMFClassifier

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO
)

# Simulation settings
n_samples = 1000
n_features = 2
n_classes = 2
seed = 123
random_state_sim = 123

# Classifier settings
n_estimators = 10
random_state_clf = 123
step = 1.0
split_pure = False
use_aggregation = True

# Experiment settings
save_iterations = [5, 10, 30, 50, 100, 200]
output_filename = "iterations.pdf"

logging.info("Simulation of the data")
X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=random_state_sim)

logging.info("Train/Test splitting")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=random_state_sim
)

logging.info("Computation of the meshgrid")
h = 0.1
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = np.zeros(xx.shape)

cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])

clf = AMFClassifier(
    n_classes=n_classes,
    n_estimators=n_estimators,
    random_state=random_state_clf,
    step=step,
    split_pure=split_pure,
    use_aggregation=use_aggregation,
)

n_plots = len(save_iterations)
n_fig = 0
save_iterations = [0, *save_iterations]

fig = plt.figure(figsize=(3 * n_plots, 3.2))

logging.info("Launching iterations")
bar = trange(n_plots, desc="Plotting iterations", leave=True)

for start, end in zip(save_iterations[:-1], save_iterations[1:]):
    X_iter = X_train[start:end]
    y_iter = y_train[start:end]
    clf.partial_fit(X_iter, y_iter)

    n_fig += 1
    Z = clf.predict_proba(np.array([xx.ravel(), yy.ravel()]).T)[:, 1]
    Z = Z.reshape(xx.shape)
    ax = plt.subplot(1, n_plots, n_fig)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
    ax.scatter(X_train[:end, 0], X_train[:end, 1], c=y_train[:end], s=25, cmap=cm)
    score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    ax.set_title("t = %d" % end, fontsize=20)

    ax.text(
        xx.max() - 0.3,
        yy.min() + 0.3,
        ("%.2f" % score).lstrip("0"),
        size=19,
        horizontalalignment="right",
    )

    bar.update(1)

bar.close()

plt.tight_layout()
plt.savefig(output_filename)

logging.info("Saved result in file %s" % output_filename)

