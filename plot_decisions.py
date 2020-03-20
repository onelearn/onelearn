import numpy as np
import matplotlib.pyplot as plt
import logging

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.datasets import make_moons, make_classification, make_circles
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from skgarden import MondrianForestClassifier

from onelearn import AMFClassifier
from onelearn.plot import (
    get_mesh,
    plot_contour_binary_classif,
    plot_scatter_binary_classif,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

np.set_printoptions(precision=2)


def plot_decision_classification(classifiers, datasets, h, levels):
    n_classifiers = len(classifiers)
    n_datasets = len(datasets)
    _ = plt.figure(figsize=(2 * (n_classifiers + 1), 2 * n_datasets))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        X, y = ds
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        xx, yy, X_mesh = get_mesh(X, h=h)
        ax = plt.subplot(n_datasets, n_classifiers + 1, i)

        if ds_cnt == 0:
            title = "Input data"
        else:
            title = None

        plot_scatter_binary_classif(ax, xx, yy, X_train, y_train, s=10, title=title)
        plot_scatter_binary_classif(ax, xx, yy, X_train, y_train, s=10, alpha=0.6)

        i += 1
        for name, clf in classifiers:
            ax = plt.subplot(n_datasets, n_classifiers + 1, i)
            if hasattr(clf, "clear"):
                clf.clear()
            if hasattr(clf, "partial_fit"):
                clf.partial_fit(X_train, y_train)
            else:
                clf.fit(X_train, y_train)

            Z = clf.predict_proba(X_mesh)[:, 1].reshape(xx.shape)
            score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
            if ds_cnt == 0:
                plot_contour_binary_classif(
                    ax, xx, yy, Z, levels=levels, score=score, title=name
                )
            else:
                plot_contour_binary_classif(ax, xx, yy, Z, score=score, levels=levels)
            i += 1

    plt.tight_layout()


# Simulation of datasets
n_samples = 300
n_features = 2
n_classes = 2
random_state = 12

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_redundant=0,
    n_informative=2,
    random_state=random_state,
    n_clusters_per_class=1,
    flip_y=0.001,
    class_sep=2.0,
)
rng = np.random.RandomState(random_state)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    make_moons(n_samples=n_samples, noise=0.2, random_state=random_state),
    make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=random_state),
    linearly_separable,
]

n_estimators = 10

classifiers = [
    (
        "AMF",
        AMFClassifier(
            n_classes=n_classes,
            n_estimators=n_estimators,
            random_state=random_state,
            use_aggregation=True,
            split_pure=True,
        ),
    ),
    (
        "MF",
        MondrianForestClassifier(n_estimators=n_estimators, random_state=random_state),
    ),
    (
        "RF",
        RandomForestClassifier(n_estimators=n_estimators, random_state=random_state),
    ),
    ("ET", ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state)),
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=random_state
)

h = 0.02
levels = 20
plot_decision_classification(classifiers, datasets, h, levels)


plt.savefig("decisions.pdf")
logging.info("Saved the decision functions in 'decision.pdf")
