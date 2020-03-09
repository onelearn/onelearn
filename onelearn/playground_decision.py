# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

# streamlit run playground.py

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from onelearn import AMFClassifier

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
import colorcet as cc


n_samples = 100
grid_size = 200
random_state = 42

# TODO: at iteration=3 there is a blue region but only red points with
#  aggregation ?!?


st.title("`AMFClassifier` playground")

# The sidebar
st.sidebar.title("Dataset")
st.sidebar.markdown("Choose the dataset below")
dataset = st.sidebar.selectbox("dataset", ["moons"], index=0)
st.sidebar.title("Parameters")
st.sidebar.markdown(
    """You can tune below some 
hyperparameters of the AMFClassifier"""
)
use_aggregation = st.sidebar.checkbox("use_aggregation", value=True)
split_pure = st.sidebar.checkbox("split_pure", value=False)
n_estimators = st.sidebar.selectbox("n_estimators", [1, 5, 10, 50, 200], index=2)
step = st.sidebar.selectbox("step", [1.0, 0.1, 2.0, 3.0], index=0)
dirichlet = st.sidebar.selectbox("dirichlet", [0.01, 0.05, 0.1, 0.5, 1.0, 2.0], index=3)


@st.cache
def simulate_data():
    X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=random_state)
    X = MinMaxScaler().fit_transform(X)
    return X, y


X_train, y_train = simulate_data()
n_classes = int(y_train.max() + 1)
n_samples_train = X_train.shape[0]
show_iterations = st.sidebar.checkbox("show_iterations", value=False)


@st.cache
def get_data_df(X, y):
    y_color = {0: "blue", 1: "red"}
    df = pd.DataFrame(
        {"x1": X[:, 0], "x2": X[:, 1], "y": y}, columns=["x1", "x2", "y"],
    )
    df["y"] = df["y"].map(lambda y: y_color[y])
    return df


df_data = get_data_df(X_train, y_train)
df_data_current = df_data.copy()


@st.cache
def get_mesh(grid_size):
    xx, yy = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
    xy = np.array([xx.ravel(), yy.ravel()]).T
    xy = np.ascontiguousarray(xy, dtype="float32")
    return xy


xy = get_mesh(grid_size)


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
    amf.partial_fit(X_train, y_train)
    zz = amf.predict_proba(xy)[:, 1].reshape(grid_size, grid_size)
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
        amf.partial_fit(X_train[it - 1].reshape(1, 2), np.array([y_train[it - 1]]))
        zz = amf.predict_proba(xy)[:, 1].reshape(grid_size, grid_size)
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
    df_data_current = df_data[:iteration]
else:
    zz = get_amf_decision_batch(
        use_aggregation, n_estimators, split_pure, dirichlet, step
    )

source_data = ColumnDataSource(ColumnDataSource.from_df(df_data_current))
source_decision = ColumnDataSource(data={"image": [zz]})

plot_data = figure(plot_width=600, plot_height=600, x_range=[0, 1], y_range=[0, 1],)
plot_data.image(
    "image", source=source_decision, x=0, y=0, dw=1, dh=1, palette=cc.CET_D1A
)


circles_data = plot_data.circle(
    x="x1",
    y="x2",
    size=10,
    color="y",
    line_width=2,
    line_color="black",
    name="circles",
    alpha=0.7,
    source=source_data,
)

plot_data.outline_line_color = None
plot_data.grid.visible = False
plot_data.axis.visible = False

st.bokeh_chart(plot_data)


"""This small demo illustrates the usage of the Aggregation Mondrian Forest for 
binary classification using `AMFClassifier`.

## Reference

> J. Mourtada, S. Gaïffas and E. Scornet, *AMF: Aggregated Mondrian Forests for Online Learning*, 
arXiv link: http://arxiv.org/abs/1906.10529
"""
