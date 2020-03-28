# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import sys
import logging
from itertools import product
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from skgarden import MondrianForestClassifier

sys.path.append("..")
from onelearn import AMFClassifier, OnlineDummyClassifier


classifier_colors = {
    "AMF": "#2ca02c",
    "AMF(sp=False)": "#2ca02c",
    "AMF(sp=True)": "#2ca0a0",
    "MF": "#1f77b4",
    "SGD": "#ff7f0e",
    "Dummy": "#d62728",
    "ET": "#9467bd",
    "RF": "#e377c2",
}


def print_datasets(loaders, datasets_path):
    col_n_samples = []
    col_n_features = []
    col_n_classes = []
    col_names = []
    col_sizes = []

    for loader in loaders:
        X, y, dataset_name = loader(datasets_path)
        n_samples, n_features = X.shape
        n_classes = int(y.max() + 1)
        size = n_samples * n_features
        col_n_samples.append(n_samples)
        col_n_features.append(n_features)
        col_n_classes.append(n_classes)
        col_names.append(dataset_name)
        col_sizes.append(size)

    datasets_description = pd.DataFrame(
        {
            "dataset": col_names,
            "n_samples": col_n_samples,
            "n_features": col_n_features,
            "n_classes": col_n_classes,
        },
        columns=["dataset", "n_samples", "n_features", "n_classes"],
    )
    logging.info("Running experiments on the following datasets:")
    print(datasets_description)


def log_loss(y_test, y_pred, n_classes, y_train=None, eps=1e-15, normalize=False):
    y_train_unique = np.unique(y_train)
    n_samples = y_test.shape[0]
    # We need to correct y_pred since it's likely to be wrong...
    if y_train is not None:
        y_pred_correct = eps * np.ones((n_samples, n_classes))
        # y_pred is obtained using samples with y_train labels
        # We replace it by y_pred_correct with score prediction in the right
        # places and eps elsewhere
        for j, col in enumerate(y_train_unique):
            y_pred_correct[:, col] = y_pred[:, j]
        probs = np.array([y_pred_correct[i, y] for (i, y) in enumerate(y_test)])
    else:
        probs = np.array([y_pred[i, y] for (i, y) in enumerate(y_test)])

    np.clip(probs, eps, 1 - eps, out=probs)
    if normalize:
        return (-np.log(probs)).mean()
    else:
        return (-np.log(probs)).sum()


def log_loss_single(y_test, y_pred, eps=1e-15):
    score = y_pred[0, int(y_test[0])]
    score_clipped = np.clip(score, eps, 1)
    return -np.log(score_clipped)


def get_classifiers_online(n_classes, random_state=42):
    use_aggregations = [True]
    n_estimatorss = [10]
    split_pures = [False]
    dirichlets = [None]
    learning_rates = [0.1]

    for (n_estimators, use_aggregation, split_pure, dirichlet) in product(
        n_estimatorss, use_aggregations, split_pures, dirichlets
    ):
        yield (
            # "AMF(nt=%s, ag=%s, sp=%s, di=%s)"
            # % (
            #     str(n_estimators),
            #     str(use_aggregation),
            #     str(split_pure),
            #     str(dirichlet),
            # ),
            "AMF",
            AMFClassifier(
                n_classes=n_classes,
                random_state=random_state,
                use_aggregation=use_aggregation,
                n_estimators=n_estimators,
                split_pure=split_pure,
                dirichlet=dirichlet,
                verbose=False,
            ),
        )

    yield "Dummy", OnlineDummyClassifier(n_classes=n_classes)

    for n_estimators in n_estimatorss:
        yield (
            "MF",
            MondrianForestClassifier(
                n_estimators=n_estimators, random_state=random_state
            ),
        )

    for learning_rate in learning_rates:
        yield (
            # "SGD(%s)" % str(learning_rate),
            "SGD",
            SGDClassifier(
                loss="log",
                learning_rate="constant",
                eta0=learning_rate,
                random_state=random_state,
            ),
        )


def get_classifiers_batch(n_classes, random_state=42):
    use_aggregations = [True]
    n_estimatorss = [10]
    split_pures = [False]
    dirichlets = [None]
    learning_rates = [1e-1]

    for (n_estimators, use_aggregation, split_pure, dirichlet) in product(
        n_estimatorss, use_aggregations, split_pures, dirichlets
    ):
        yield (
            # "AMF(nt=%s, ag=%s, sp=%s, di=%s)"
            #           % (
            #           str(n_estimators),
            #       str(use_aggregation),
            #       str(split_pure),
            #       str(dirichlet),
            # ),
            "AMF",
            AMFClassifier(
                n_classes=n_classes,
                random_state=random_state,
                use_aggregation=use_aggregation,
                n_estimators=n_estimators,
                split_pure=split_pure,
                dirichlet=dirichlet,
                verbose=False,
            ),
        )

    for n_estimators in n_estimatorss:
        yield (
            # "MF(nt=%s)" % str(n_estimators),
            "MF",
            MondrianForestClassifier(
                n_estimators=n_estimators, random_state=random_state
            ),
        )

    for n_estimators in n_estimatorss:
        yield (
            # "RF(nt=%s)" % str(n_estimators),
            "RF",
            RandomForestClassifier(
                n_estimators=n_estimators,
                class_weight=None,
                random_state=random_state,
                n_jobs=1,
            ),
        )

    for n_estimators in n_estimatorss:
        yield (
            # "ET(nt=%s)" % str(n_estimators),
            "ET",
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                class_weight=None,
                random_state=random_state,
                n_jobs=1,
            ),
        )

    for learning_rate in learning_rates:
        yield (
            # "SGD(%s)" % str(learning_rate),
            "SGD",
            SGDClassifier(
                loss="log",
                learning_rate="constant",
                eta0=learning_rate,
                random_state=random_state,
            ),
        )


def get_classifiers_n_trees_comparison(n_classes, random_state=42):
    use_aggregations = [True]
    n_estimatorss = [1, 2, 5, 10, 20, 50]
    split_pures = [False]
    dirichlets = [None]
    for (n_estimators, use_aggregation, split_pure, dirichlet) in product(
        n_estimatorss, use_aggregations, split_pures, dirichlets
    ):
        yield (
            "AMF(nt=%s)" % str(n_estimators),
            AMFClassifier(
                n_classes=n_classes,
                random_state=random_state,
                use_aggregation=use_aggregation,
                n_estimators=n_estimators,
                split_pure=split_pure,
                dirichlet=dirichlet,
                verbose=False,
            ),
        )

    for n_estimators in n_estimatorss:
        yield (
            "MF(nt=%s)" % str(n_estimators),
            MondrianForestClassifier(
                n_estimators=n_estimators, random_state=random_state
            ),
        )

    for n_estimators in n_estimatorss:
        yield (
            "RF(nt=%s)" % str(n_estimators),
            RandomForestClassifier(
                n_estimators=n_estimators,
                class_weight=None,
                random_state=random_state,
                n_jobs=1,
            ),
        )

    for n_estimators in n_estimatorss:
        yield (
            "ET(nt=%s)" % str(n_estimators),
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                class_weight=None,
                random_state=random_state,
                n_jobs=1,
            ),
        )
