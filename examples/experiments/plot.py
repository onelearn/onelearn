# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import numpy as np
import matplotlib.pyplot as plt

# TODO: docstrings


def get_mesh(X, h=0.02, padding=0.5):
    """Build a regular meshgrid using the range of the features in X
    """
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X_mesh = np.array([xx.ravel(), yy.ravel()]).T
    return xx, yy, X_mesh


def plot_scatter_binary_classif(
    ax,
    xx,
    yy,
    X,
    y,
    s=10,
    alpha=None,
    cm=None,
    title=None,
    fontsize=None,
    lw=None,
    norm=None,
    noaxes=False,
):
    if cm is None:
        cm = plt.get_cmap("RdBu")

    ax.scatter(X[:, 0], X[:, 1], c=y, s=s, cmap=cm, alpha=alpha, lw=lw, norm=norm)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if noaxes:
        ax.axis("off")


def plot_contour_binary_classif(
    ax, xx, yy, Z, cm=None, alpha=0.8, levels=200, title=None, score=None, norm=None
):
    if cm is None:
        cm = plt.get_cmap("RdBu")
    ax.contourf(xx, yy, Z, cmap=cm, alpha=alpha, levels=levels, norm=norm)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if title is not None:
        ax.set_title(title)
    if score is not None:
        ax.text(
            xx.max() - 0.3,
            yy.min() + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=25,
            horizontalalignment="right",
        )
