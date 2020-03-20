# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import numpy as np
from numba import jitclass, njit
from .types import float32, uint32, get_array_2d_type, void
from .utils import get_type


spec_samples_collection = [
    ("features", get_array_2d_type(float32)),
    ("labels", float32[::1]),
    ("reserve_samples", uint32),
    ("n_samples", uint32),
]


# TODO: write all the docstrings


@jitclass(spec_samples_collection)
class SamplesCollection(object):
    def __init__(self, reserve_samples):
        self.reserve_samples = reserve_samples
        self.n_samples = 0


@njit(void(get_type(SamplesCollection), get_array_2d_type(float32), float32[::1]))
def add_samples(samples, X, y):
    n_samples_batch, n_features = X.shape
    if samples.n_samples == 0:
        samples.n_samples += n_samples_batch
        # TODO: use copy instead or something else instead ?
        features = np.empty((n_samples_batch, n_features), dtype=float32)
        features[:] = X
        samples.features = features
        labels = np.empty(n_samples_batch, dtype=float32)
        labels[:] = y
        samples.labels = labels
    else:
        samples.n_samples += n_samples_batch
        samples.features = np.concatenate((samples.features, X))
        samples.labels = np.concatenate((samples.labels, y))
