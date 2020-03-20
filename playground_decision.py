# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

# Run the following:
# streamlit run playground_decision.py

import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.datasets import (
    make_moons,
    make_circles,
    make_blobs,
    make_classification,
)
from sklearn.preprocessing import MinMaxScaler

from onelearn import AMFClassifier
from onelearn.plot import (
    get_mesh,
    plot_contour_binary_classif,
    plot_scatter_binary_classif,
)

# Some global parameters
n_samples = 101
random_state = 42
# Delta in the meshgrid
h = 0.01
# Number of color levels in the contour plot
levels = 100

st.title("`AMFClassifier` playground")

# The sidebar
st.sidebar.title("Dataset")
st.sidebar.markdown("Choose the dataset below")
dataset = st.sidebar.selectbox(
    "dataset", ["moons", "circles", "linear", "blobs"], index=0
)
show_data = st.sidebar.checkbox("Show data", value=True)
normalize = st.sidebar.checkbox("Normalize colors", value=True)

st.sidebar.title("Parameters")
st.sidebar.markdown("Hyperparameters of the AMFClassifier")
use_aggregation = st.sidebar.checkbox("Use aggregation", value=True)
split_pure = st.sidebar.checkbox("Split pure cells", value=True)
n_estimators = st.sidebar.selectbox(
    "Number of trees", [1, 5, 10, 50, 200, 500], index=2
)
step = st.sidebar.selectbox("step", [0.01, 0.1, 0.5, 1.0, 5.0, 100.0, 1000.0], index=3)
dirichlet = st.sidebar.selectbox(
    "dirichlet", [1e-7, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 50.0, 1000.0], index=5
)

if normalize:
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
else:
    norm = None


@st.cache
def simulate_data(dataset="moons"):
    if dataset == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=random_state)
    elif dataset == "circles":
        X, y = make_circles(
            n_samples=n_samples, noise=0.1, factor=0.5, random_state=random_state
        )
    elif dataset == "linear":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=random_state,
            n_clusters_per_class=1,
            flip_y=0.001,
            class_sep=2.0,
        )
        rng = np.random.RandomState(random_state)
        X += 2 * rng.uniform(size=X.shape)
    elif dataset == "blobs":
        X, y = make_blobs(n_samples=n_samples, centers=5, random_state=random_state)
        y[y == 2] = 0
        y[y == 3] = 1
        y[y == 4] = 0
    else:
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=random_state)
    X = MinMaxScaler().fit_transform(X)
    return X, y


X, y = simulate_data(dataset)
n_classes = int(y.max() + 1)
n_samples_train = X.shape[0]
st.sidebar.markdown("Show the iterations instead of the final decision function")
show_iterations = st.sidebar.checkbox("Show iterations", value=False)


@st.cache
def build_mesh(X):
    return get_mesh(X, h=h, padding=0.2)


xx, yy, X_mesh = build_mesh(X)


@st.cache
def get_amf_decision_batch(use_aggregation, n_estimators, split_pure, dirichlet, step):
    # TODO: add a progress bar
    amf = AMFClassifier(
        n_classes=n_classes,
        random_state=random_state,
        use_aggregation=use_aggregation,
        n_estimators=n_estimators,
        split_pure=split_pure,
        dirichlet=dirichlet,
        step=step,
    )
    amf.partial_fit(X, y)
    zz = amf.predict_proba(X_mesh)[:, 1].reshape(xx.shape)
    return zz


@st.cache(suppress_st_warning=True)
def get_amf_decisions(use_aggregation, n_estimators, split_pure, dirichlet, step):
    amf = AMFClassifier(
        n_classes=n_classes,
        random_state=random_state,
        use_aggregation=use_aggregation,
        n_estimators=n_estimators,
        split_pure=split_pure,
        dirichlet=dirichlet,
        step=step,
    )
    zzs = []
    progress_bar = st.sidebar.progress(0)
    for it in range(1, n_samples_train + 1):
        amf.partial_fit(X[it - 1].reshape(1, 2), np.array([y[it - 1]]))
        zz = amf.predict_proba(X_mesh)[:, 1].reshape(xx.shape)
        zzs.append(zz)
        progress = int(100 * it / n_samples_train)
        progress_bar.progress(progress)
    return zzs


if show_iterations:
    iteration = st.sidebar.number_input(
        label="iteration", min_value=1, max_value=n_samples_train - 1, value=1, step=1
    )
    zzs = get_amf_decisions(use_aggregation, n_estimators, split_pure, dirichlet, step)
    zz = zzs[iteration - 1]
    X_current = X[:iteration]
    y_current = y[:iteration]
else:
    zz = get_amf_decision_batch(
        use_aggregation, n_estimators, split_pure, dirichlet, step
    )
    X_current = X
    y_current = y


_ = plt.figure(figsize=(3, 3))
ax = plt.subplot(1, 1, 1)
plot_contour_binary_classif(ax, xx, yy, zz, levels=levels, norm=norm)
if show_data:
    plot_scatter_binary_classif(ax, xx, yy, X_current, y_current, s=5, lw=1, norm=norm)
plt.tight_layout()
st.pyplot()


"""This small demo illustrates the usage of the Aggregation Mondrian Forest for 
binary classification using `AMFClassifier`.

## Reference

> J. Mourtada, S. Gaïffas and E. Scornet, *AMF: Aggregated Mondrian Forests for Online Learning*, 
arXiv link: http://arxiv.org/abs/1906.10529
"""
