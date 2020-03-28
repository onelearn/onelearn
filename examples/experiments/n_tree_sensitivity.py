# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import os
import logging
from collections import defaultdict
from itertools import product
import pickle as pkl
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from .utils import classifier_colors, get_classifiers_n_trees_comparison


def compute_aucs_n_trees(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=random_state, shuffle=True, stratify=y
    )
    n_classes = int(y.max() + 1)
    classes = np.arange(n_classes)
    classifiers = get_classifiers_n_trees_comparison(n_classes)
    test_aucs = defaultdict(list)
    for clf_name, clf in classifiers:
        logging.info("  using %s" % clf_name)
        if hasattr(clf, "partial_fit"):
            clf.partial_fit(X_train, y_train, classes=classes)
        else:
            clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_pred[:, 1])
        test_aucs[clf_name].append(auc)
        if hasattr(clf, "clear"):
            clf.clear()
        del clf
    return test_aucs


def read_data_n_trees(filenames):
    n_treess = [1, 2, 5, 10, 20, 50]
    col_clf = []
    col_n_trees = []
    col_auc = []
    col_dataset = []
    col_x_pos = []
    for filename in filenames:
        filename_pkl = os.path.join(filename)
        p = Path(filename_pkl)
        with open(p, "rb") as f:
            results = pkl.load(f)
            dataset = results["dataset"]
            for clf_name, (x_pos, n_trees) in product(
                ["AMF", "MF", "RF", "ET"], enumerate(n_treess)
            ):
                key = clf_name + "(nt=%d)" % n_trees
                auc = max(results["test_aucs"][key])
                col_dataset.append(dataset)
                col_clf.append(clf_name)
                col_n_trees.append(n_trees)
                col_auc.append(auc)
                col_x_pos.append(x_pos + 1)
    df = pd.DataFrame(
        {
            "dataset": col_dataset,
            "clf": col_clf,
            "n_trees": col_n_trees,
            "auc": col_auc,
            "x_pos": col_x_pos,
        }
    )
    return df


def plot_comparison_n_trees(df, filename=None, legend=True):
    df["dataset"].unique()
    g = sns.FacetGrid(
        df, col="dataset", col_wrap=4, aspect=1, height=4, sharex=True, sharey=False
    )
    g.map(
        sns.lineplot,
        "x_pos",
        "auc",
        "clf",
        lw=4,
        marker="o",
        markersize=10,
        palette=classifier_colors,
    ).set(yscale="log", xlabel="", ylabel="")

    axes = g.axes.flatten()

    for i, dataset in enumerate(df["dataset"].unique()):
        axes[i].set_xticklabels([0, 1, 2, 5, 10, 20, 50], fontsize=14)
        axes[i].set_title(dataset, fontsize=18)

    if legend:
        plt.legend(
            ["AMF", "MF", "RF", "ET"],
            bbox_to_anchor=(0.3, 0.7, 1.0, 0.0),
            loc="upper right",
            ncol=1,
            borderaxespad=0.0,
            fontsize=14,
        )

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
        logging.info("Saved figure in " + filename)
    else:
        plt.show()
