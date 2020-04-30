# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

# streamlit run playground_tree.py

import sys
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
import colorcet as cc

sys.path.extend([".", ".."])
from onelearn import AMFClassifier


n_samples = 100
grid_size = 200
random_state = 42
n_estimators = 1


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
split_pure = st.sidebar.checkbox("split_pure", value=True)
step = st.sidebar.selectbox("step", [1.0, 0.1, 2.0, 3.0], index=0)
dirichlet = st.sidebar.selectbox(
    "dirichlet", [1e-8, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0], index=3
)


@st.cache
def simulate_data():
    X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=random_state)
    X = MinMaxScaler().fit_transform(X)
    return X, y


X_train, y_train = simulate_data()
n_classes = int(y_train.max() + 1)
n_samples_train = X_train.shape[0]


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


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_amf_trees_and_decisions(
    use_aggregation, n_estimators, split_pure, dirichlet, step
):
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
    df_trees = []
    df_datas = []
    progress_bar = st.sidebar.progress(0)
    for it in range(1, n_samples_train + 1):
        # Append the current data
        df_datas.append(df_data[:it])

        # Partial fit AMFClassifier
        amf.partial_fit(X_train[it - 1].reshape(1, 2), np.array([y_train[it - 1]]))

        # Get the tree
        df_tree = amf.get_nodes_df(0)
        df_tree["min_x"] = df_tree["memory_range_min"].apply(lambda t: t[0])
        df_tree["min_y"] = df_tree["memory_range_min"].apply(lambda t: t[1])
        df_tree["max_x"] = df_tree["memory_range_max"].apply(lambda t: t[0])
        df_tree["max_y"] = df_tree["memory_range_max"].apply(lambda t: t[1])
        df_tree["count_0"] = df_tree["counts"].apply(lambda t: t[0])
        df_tree["count_1"] = df_tree["counts"].apply(lambda t: t[1])
        df_tree.sort_values(by=["depth", "parent", "id"], inplace=True)
        # max_depth = df.depth.max()
        max_depth = 10
        n_nodes = df_tree.shape[0]
        x = np.zeros(n_nodes)
        x[0] = 0.5
        indexes = df_tree["id"].values
        df_tree["x"] = x
        df_tree["y"] = max_depth - df_tree["depth"]
        df_tree["x0"] = df_tree["x"]
        df_tree["y0"] = df_tree["y"]
        for node in range(1, n_nodes):
            index = indexes[node]
            parent = df_tree.at[index, "parent"]
            depth = df_tree.at[index, "depth"]
            left_parent = df_tree.at[parent, "left"]
            x_parent = df_tree.at[parent, "x"]
            if left_parent == index:
                # It's a left node
                df_tree.at[index, "x"] = x_parent - 0.5 ** (depth + 1)
            else:
                df_tree.at[index, "x"] = x_parent + 0.5 ** (depth + 1)
            df_tree.at[index, "x0"] = x_parent
            df_tree.at[index, "y0"] = df_tree.at[parent, "y"]

        df_tree["color"] = df_tree["is_leaf"].astype("str")
        df_tree.replace({"color": {"False": "blue", "True": "green"}}, inplace=True)
        df_trees.append(df_tree)

        # Compute the decision function
        zz = amf.predict_proba(xy)[:, 1].reshape(grid_size, grid_size)
        zzs.append(zz)
        progress = int(100 * it / n_samples_train)
        progress_bar.progress(progress)

    return zzs, df_datas, df_trees


iteration = st.sidebar.number_input(
    label="iteration", min_value=1, max_value=n_samples_train - 1, value=1, step=1
)

zzs, df_datas, df_trees = get_amf_trees_and_decisions(
    use_aggregation, n_estimators, split_pure, dirichlet, step
)


zz = zzs[iteration - 1]
df_data_current = df_datas[iteration - 1]
df_tree = df_trees[iteration - 1]

source_tree = ColumnDataSource(ColumnDataSource.from_df(df_tree))

source_data = ColumnDataSource(ColumnDataSource.from_df(df_data_current))
source_decision = ColumnDataSource(data={"image": [zz]})

plot_tree = figure(
    plot_width=1000, plot_height=500, x_range=[-0.1, 1.1], y_range=[0, 11],
)

plot_tree.outline_line_color = None
plot_tree.axis.visible = False
plot_tree.grid.visible = False


circles = plot_tree.circle(
    x="x",
    y="y",
    size=10,
    fill_color="color",
    name="circles",
    fill_alpha=0.4,
    source=source_tree,
)
plot_tree.segment(
    x0="x",
    y0="y",
    x1="x0",
    y1="y0",
    line_color="#151515",
    line_alpha=0.4,
    source=source_tree,
)

tree_hover = HoverTool(
    renderers=[circles],
    tooltips=[
        ("index", "@id"),
        ("depth", "@depth"),
        ("parent", "@parent"),
        ("left", "@left"),
        ("right", "@right"),
        ("is_leaf", "@is_leaf"),
        # ('time', '@time'),
        # ("feature", "@feature"),
        # ("threshold", "@threshold"),
        ("n_samples", "@n_samples"),
        ("weight", "@weight"),
        ("log_weight_tree", "@log_weight_tree"),
        # ("min_x", "@min_x{0.000}"),
        # ("min_y", "@min_y{0.000}"),
        # ("max_x", "@max_x{0.000}"),
        # ("max_y", "@max_y{0.000}"),
        ("count_0", "@count_0"),
        ("count_1", "@count_1"),
        # ("memorized", "@memorized"),
    ],
)


plot_tree.add_tools(tree_hover)

plot_tree.text(x="x", y="y", text="id", source=source_tree)

plot_data = figure(plot_width=500, plot_height=500, x_range=[0, 1], y_range=[0, 1])
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

st.bokeh_chart(plot_tree)

plot_data.outline_line_color = None
plot_data.grid.visible = False
plot_data.axis.visible = False

st.bokeh_chart(plot_data)


"""This small demo illustrates the internal tree construction performed by  
`AMFClassifier` for binary classification.

## Reference

> J. Mourtada, S. Gaïffas and E. Scornet, *AMF: Aggregated Mondrian Forests for Online Learning*, 
arXiv link: http://arxiv.org/abs/1906.10529
"""
