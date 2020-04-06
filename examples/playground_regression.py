# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

# Run the following:
# streamlit run playground_regression.py

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import streamlit as st

sys.path.extend([".", ".."])

from onelearn import AMFRegressor
from onelearn.datasets import get_signal, make_regression

n_samples_train = 5000
n_samples_test = 1000
random_state = 42

colormap = get_cmap("tab20")

st.title("`AMFRegressor` playground")
st.sidebar.title("Dataset")
st.sidebar.markdown("Choose the signal and noise level below")
signal = st.sidebar.selectbox(
    "Signal", ["heavisine", "bumps", "blocks", "doppler"], index=2
)
noise = st.sidebar.slider(
    label="Noise", min_value=0.0, max_value=0.1, step=0.01, value=0.03
)
st.sidebar.title("Parameters")
st.sidebar.markdown("Hyperparameters of the AMFRegressor")
use_aggregation = st.sidebar.checkbox("Use aggregation", value=True)
split_pure = st.sidebar.checkbox("Split pure cells", value=True)
n_estimators = st.sidebar.selectbox(
    "Number of trees", [1, 5, 10, 50, 100, 200], index=4
)
step = st.sidebar.slider(
    label="Step", min_value=1.0, max_value=20.0, step=1.0, value=10.0
)


@st.cache
def get_test(num):
    return np.linspace(0, 1, num=num)


X_test = get_test(n_samples_test)


@st.cache
def simulate_data(n_samples, signal, noise):
    return make_regression(
        n_samples=n_samples, signal=signal, noise=noise, random_state=random_state
    )


@st.cache()
def compute_signal(x, dataset):
    return get_signal(x, signal=dataset)


X_train, y_train = simulate_data(n_samples=n_samples_train, signal=signal, noise=noise)


@st.cache
def get_amf_prediction(use_aggregation, n_estimators, split_pure, step):
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
    return y_pred, weighted_depths, weighted_depths.mean(axis=1)


y_pred, weighted_depths, weighted_depths_mean = get_amf_prediction(
    use_aggregation, n_estimators, split_pure, step
)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 5))
fig.subplots_adjust(hspace=0.1)
plot_samples = ax1.plot(
    X_train, y_train, color=colormap.colors[1], lw=2, label="Samples"
)[0]
plot_signal = ax1.plot(
    X_test, get_signal(X_test, signal), lw=2, color=colormap.colors[0], label="Signal"
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
    weighted_depths_mean,
    lw=2,
    color=colormap.colors[4],
    label="Mean weighted depth",
)[0]
fig.legend(
    (
        plot_signal,
        plot_samples,
        plot_mean_weighted_depths,
        plot_weighted_depths,
        plot_prediction,
    ),
    ("Signal", "Samples", "Average weighted depths", "Weighted depths", "Prediction",),
    fontsize=8,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=3,
)

st.pyplot()


"""This small demo illustrates the usage of the Aggregated Mondrian Forest for 
regression using `AMFRegressor`.

## Reference

> J. Mourtada, S. Gaïffas and E. Scornet, *AMF: Aggregated Mondrian Forests for Online Learning*, 
arXiv link: http://arxiv.org/abs/1906.10529
"""
