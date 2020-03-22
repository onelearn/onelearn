# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import logging
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tqdm import trange

from onelearn import AMFClassifier
from onelearn.plot import (
    get_mesh,
    plot_contour_binary_classif,
    plot_scatter_binary_classif,
)

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO
)

n_samples = 400
n_features = 2
n_classes = 2
seed = 123
random_state = 42
save_iterations = [5, 10, 30, 50, 100, 200]
output_filename = "iterations.pdf"

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
    n_estimators=10,
    random_state=random_state,
    split_pure=True,
    use_aggregation=True,
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
    Z = clf.predict_proba(X_mesh)[:, 1].reshape(xx.shape)
    ax = plt.subplot(1, n_plots, n_fig)

    score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    plot_contour_binary_classif(ax, xx, yy, Z, score=score, levels=20)

    plot_scatter_binary_classif(
        ax,
        xx,
        yy,
        X_train[:end],
        y_train[:end],
        s=25,
        title="t = %d" % end,
        fontsize=20,
    )
    bar.update(1)

bar.close()

plt.tight_layout()
plt.savefig(output_filename)
logging.info("Saved result in file %s" % output_filename)
