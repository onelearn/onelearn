# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

import numpy as np
from numba import jitclass
from numba.types import float32, uint32

spec_samples_collection = [
    ("features", float32[:, ::1]),
    ("labels", float32[::1]),
    ("n_samples", uint32),
]


# TODO: this requires serious improvements, such as pre-allocating things in
#  advance instead of just what is required each time

# TODO: write all the docstrings


@jitclass(spec_samples_collection)
class SamplesCollection(object):
    def __init__(self):
        self.n_samples = 0

    def append(self, X, y):
        n_samples_batch, n_features = X.shape
        if self.n_samples == 0:
            self.n_samples += n_samples_batch
            # TODO: use copy instead or something else instead ?
            features = np.empty((n_samples_batch, n_features), dtype=float32)
            features[:] = X
            self.features = features
            labels = np.empty(n_samples_batch, dtype=float32)
            labels[:] = y
            self.labels = labels
        else:
            self.n_samples += n_samples_batch
            self.features = np.concatenate((self.features, X))
            self.labels = np.concatenate((self.labels, y))
