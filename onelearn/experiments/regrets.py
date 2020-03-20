# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import logging
from collections import defaultdict
from time import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle as skshuffle
from .utils import classifier_colors, get_classifiers_online, log_loss_single


def compute_regrets(X, y, shuffle=True, random_state=42):
    regrets = defaultdict(list)
    timings = defaultdict(list)
    if shuffle:
        X, y = skshuffle(X, y, random_state=random_state)

    n_samples, n_features = X.shape
    n_iterations = n_samples - 1
    iterations = np.arange(n_iterations)
    n_classes = int(y.max() + 1)
    classes = np.arange(n_classes)
    classifiers = get_classifiers_online(n_classes)

    for clf_name, clf in classifiers:
        assert hasattr(clf, "partial_fit")
        logging.info("  using %s" % clf_name)
        for i in tqdm(range(1, n_iterations)):
            x_train = X[i - 1].reshape(1, n_features)
            y_train = np.array([y[i - 1]])
            x_test = X[i].reshape(1, n_features)
            y_test = np.array([y[i]])
            t1 = time()
            clf.partial_fit(x_train, y_train, classes)
            t2 = time()
            y_pred = clf.predict_proba(x_test)
            test_loss = log_loss_single(y_test, y_pred)
            regrets[clf_name].append(test_loss)
            timings[clf_name].append(t2 - t1)
        if hasattr(clf, "clear"):
            clf.clear()
        del clf
    return iterations, regrets, timings


def plot_regrets(
    results,
    show_classifiers,
    savefig=None,
    log_scale=False,
    remove_parameters=True,
    offset=0,
    figsize=(4.5, 4),
    ylim=None,
):
    dataset_name = results["dataset"]
    plt.figure(figsize=figsize)

    ylim_min = 1e15
    ylim_max = -1e15

    for clf_name in show_classifiers:
        regret = np.array(results["regrets"][clf_name])
        n_minibatch = regret.shape[0]
        iterations = np.arange(1, n_minibatch + 1)

        if remove_parameters:
            label = clf_name.split("(")[0]
        else:
            label = clf_name

        avg_regret = np.array(regret).cumsum() / iterations
        col = classifier_colors[label]

        iterations = iterations[offset:]
        avg_regret = avg_regret[offset:]

        plt.plot(iterations, avg_regret, label=label, lw=6, c=col)
        start_regret = avg_regret.max()

        if start_regret > ylim_max:
            ylim_max = start_regret

        end_regret = avg_regret[-1]
        if end_regret < ylim_min:
            ylim_min = end_regret

    ylim_min *= 0.9
    ylim_max *= 1.1

    plt.title(dataset_name, fontsize=20)
    plt.legend(fontsize=16, loc="upper right")

    if log_scale:
        plt.yscale("log")

    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim((ylim_min, ylim_max))

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
