# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import numpy as np
from numpy import sin, sign, pi, abs, sqrt
import matplotlib.pyplot as plt
from numba import vectorize
from numba.types import float32, float64


# Examples of signals originally made by David L. Donoho.


@vectorize([float32(float32), float64(float64)], nopython=True)
def heavisine(x):
    """Computes the "heavisine" signal.

    Parameters
    ----------
    x : `numpy.array`, shape=(n_samples,)
        Inputs values

    Returns
    -------
    output : `numpy.array`, shape=(n_samples,)
        The value of the signal at given inputs

    Notes
    -----
    Inputs are supposed to belong to [0, 1] and must have dtype `float32` or `float64`

    """
    return 4 * sin(4 * pi * x) - sign(x - 0.3) - sign(0.72 - x)


@vectorize([float32(float32), float64(float64)], nopython=True)
def bumps(x):
    """Computes the "bumps" signal.

    Parameters
    ----------
    x : `numpy.array`, shape=(n_samples,)
        Inputs values

    Returns
    -------
    output : `numpy.array`, shape=(n_samples,)
        The value of the signal at given inputs

    Notes
    -----
    Inputs are supposed to belong to [0, 1] and must have dtype `float32` or `float64`

    """
    pos = np.array([0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81])
    hgt = np.array([4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])
    wth = np.array(
        [0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005],
    )
    y = 0
    for j in range(pos.shape[0]):
        y += hgt[j] / ((1 + (abs(x - pos[j]) / wth[j])) ** 4)
    return y


@vectorize([float32(float32), float64(float64)], nopython=True)
def blocks(x):
    """Computes the "blocks" signal.

    Parameters
    ----------
    x : `numpy.array`, shape=(n_samples,)
        Inputs values

    Returns
    -------
    output : `numpy.array`, shape=(n_samples,)
        The value of the signal at given inputs

    Notes
    -----
    Inputs are supposed to belong to [0, 1] and must have dtype `float32` or `float64`

    """
    pos = np.array([0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81])
    hgt = np.array([4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2])
    y = 2.0
    for j in range(pos.shape[0]):
        y += (1 + sign(x - pos[j])) * (hgt[j] / 2)
    return y


@vectorize([float32(float32), float64(float64)], nopython=True)
def doppler(x):
    """Computes the "doppler" signal.

    Parameters
    ----------
    x : `numpy.array`, shape=(n_samples,)
        Inputs values

    Returns
    -------
    output : `numpy.array`, shape=(n_samples,)
        The value of the signal at given inputs

    Notes
    -----
    Inputs are supposed to belong to [0, 1] and must have dtype `float32` or `float64`

    """
    return sqrt(x * (1 - x)) * sin((2 * pi * 1.05) / (x + 0.05)) + 0.5


def get_signal(x, signal="heavisine"):
    """Computes a signal at the given inputs.

    Parameters
    ----------
    x : `numpy.array`, shape=(n_samples,)
        Inputs values

    signal : {"heavisine", "bumps", "blocks", "doppler"}, default="heavisine"
        Type of signal

    Returns
    -------
    output : `numpy.array`, shape=(n_samples,)
        The value of the signal at given inputs

    Notes
    -----
    Inputs are supposed to belong to [0, 1] and must have dtype `float32` or `float64`

    """
    if signal == "heavisine":
        y = heavisine(x)
    elif signal == "bumps":
        y = bumps(x)
    elif signal == "blocks":
        y = blocks(x)
    elif signal == "doppler":
        y = doppler(x)
    else:
        y = heavisine(x)
    y_min = y.min()
    y_max = y.max()
    return (y - y_min) / (y_max - y_min)


def make_regression(n_samples, signal="heavisine", noise=0.03, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    X = np.random.uniform(size=n_samples)
    X = np.sort(X)
    y = get_signal(X, signal)
    X = X.reshape(n_samples, 1)
    y += noise * np.random.randn(n_samples)
    return X, y
