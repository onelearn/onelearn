# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
"""
Weighted depths of `AMFRegressor` on several 1D signals.
========================================================

The example below illustrates the weighted depth learned internally by the
AMF algorithm to estimate 1D regression functions. We observe that AMF automatically
adapts to the local regularity of signals, by putting more emphasis on deeper trees
where the regression function is not unsmooth.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import logging

sys.path.extend([".", ".."])

from onelearn import AMFRegressor
from onelearn.datasets import get_signal, make_regression

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

colormap = get_cmap("tab20")

n_samples_train = 5000
n_samples_test = 1000
random_state = 42


noise = 0.03
use_aggregation = True
split_pure = True
n_estimators = 100
step = 10.0


signals = ["heavisine", "bumps", "blocks", "doppler"]


def plot_weighted_depth(signal):
    X_train, y_train = make_regression(
        n_samples=n_samples_train, signal=signal, noise=noise, random_state=random_state
    )
    X_test = np.linspace(0, 1, num=n_samples_test)

    amf = AMFRegressor(
        random_state=random_state,
        use_aggregation=use_aggregation,
        n_estimators=n_estimators,
        split_pure=split_pure,
        step=step,
    )

    amf.partial_fit(X_train.reshape(n_samples_train, 1), y_train)
    y_pred = amf.predict(X_test.reshape(n_samples_test, 1))
    weighted_depths = amf.weighted_depth(X_test.reshape(n_samples_test, 1))

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 5))

    plot_samples = ax1.plot(
        X_train, y_train, color=colormap.colors[1], lw=2, label="Samples"
    )[0]
    plot_signal = ax1.plot(
        X_test,
        get_signal(X_test, signal),
        lw=2,
        color=colormap.colors[0],
        label="Signal",
    )[0]
    plot_prediction = ax2.plot(
        X_test.ravel(), y_pred, lw=2, color=colormap.colors[2], label="Prediction"
    )[0]
    ax3.plot(
        X_test,
        weighted_depths[:, 1:],
        lw=1,
        color=colormap.colors[5],
        alpha=0.2,
        label="Weighted depths",
    )
    plot_weighted_depths = ax3.plot(
        X_test, weighted_depths[:, 0], lw=1, color=colormap.colors[5], alpha=0.2
    )[0]

    plot_mean_weighted_depths = ax3.plot(
        X_test,
        weighted_depths.mean(axis=1),
        lw=2,
        color=colormap.colors[4],
        label="Mean weighted depth",
    )[0]
    filename = "weighted_depths_%s.pdf" % signal
    fig.subplots_adjust(hspace=0.1)
    fig.legend(
        (
            plot_signal,
            plot_samples,
            plot_mean_weighted_depths,
            plot_weighted_depths,
            plot_prediction,
        ),
        (
            "Signal",
            "Samples",
            "Average weighted depths",
            "Weighted depths",
            "Prediction",
        ),
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
    )
    plt.savefig(filename)
    logging.info("Saved the decision functions in '%s'" % filename)


for signal in signals:
    plot_weighted_depth(signal)
