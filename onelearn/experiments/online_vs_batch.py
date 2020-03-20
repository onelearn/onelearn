# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import logging
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from .utils import classifier_colors, get_classifiers_batch, log_loss


def compute_regrets_and_batch(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42, shuffle=True, stratify=y
    )
    test_losses = defaultdict(list)
    iterations = defaultdict(list)
    n_samples, n_features = X_train.shape
    n_iterations = n_samples - 1
    n_classes = int(y.max() + 1)
    classes = np.arange(n_classes)
    classifiers = get_classifiers_batch(n_classes)
    test_aucs = defaultdict(list)
    for clf_name, clf in classifiers:
        logging.info("  using %s" % clf_name)
        if hasattr(clf, "partial_fit"):
            for i in range(1, n_iterations):
                xi_train = X_train[i - 1].reshape(1, n_features)
                yi_train = np.array([y_train[i - 1]])
                clf.partial_fit(xi_train, yi_train, classes=classes)
                if i % 100 == 0:
                    y_pred = clf.predict_proba(X_test)
                    test_loss = log_loss(
                        y_test, y_pred, n_classes=n_classes, normalize=True
                    )
                    test_losses[clf_name].append(test_loss)
                    iterations[clf_name].append(i)
                    if n_classes == 2:
                        auc = roc_auc_score(y_test, y_pred[:, 1])
                        test_aucs[clf_name].append(auc)
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)
            test_loss = log_loss(y_test, y_pred, n_classes=n_classes, normalize=True)
            test_losses[clf_name] = test_loss
            iterations[clf_name] = n_iterations
            if n_classes == 2:
                auc = roc_auc_score(y_test, y_pred[:, 1])
                test_aucs[clf_name].append(auc)
        if hasattr(clf, "clear"):
            clf.clear()
        del clf

    return {
        "iterations": iterations,
        "test_losses": test_losses,
        "test_aucs": test_aucs,
    }


def plot_online_vs_batch(
    results,
    show_classifiers,
    savefig=None,
    remove_parameters=True,
    figsize=(4.5, 4),
    type="log",
):
    dataset_name = results["dataset"]
    plt.figure(figsize=figsize)
    if type == "auc":
        key = "test_aucs"
    else:
        key = "test_losses"
    offset = 0

    for clf_name in show_classifiers:
        if clf_name.startswith("RF") or clf_name.startswith("ET"):
            test_loss = results[key][clf_name]
            test_losses = [test_loss, test_loss]
            n_iter = results["iterations"][clf_name]
            iterations = [0, n_iter]
            if remove_parameters:
                label = clf_name.split("(")[0]
            else:
                label = clf_name
        else:
            test_losses = np.array(results[key][clf_name])
            iterations = results["iterations"][clf_name]
            if remove_parameters:
                label = clf_name.split("(")[0]
            else:
                label = clf_name
            offset = 0

        col = classifier_colors[label]
        plt.plot(iterations[offset:], test_losses[offset:], label=label, lw=6, c=col)

    plt.title(dataset_name, fontsize=20)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
