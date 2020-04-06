# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import numpy as np
from numba import jitclass, njit
from .types import float32, uint32, get_array_2d_type, void
from .utils import get_type, resize_array


spec_samples_collection = [
    ("features", get_array_2d_type(float32)),
    ("labels", float32[::1]),
    ("n_samples_increment", uint32),
    ("n_samples", uint32),
    ("n_samples_capacity", uint32),
]


@jitclass(spec_samples_collection)
class SamplesCollection(object):
    """A class which simply keeps in memory the samples used for training when
        using repeated call to ``partial_fit``. A minimum increment is used when
        extending the capacity of the collection, in order to avoid repeated copies
        when ``add_samples`` is used on small batches.

    Attributes
    ----------
    n_samples_increment : :obj:`int`
        The minimum amount of memory which is pre-allocated each time extra memory is
        required for new samples.

    n_samples : :obj:`int`
        Number of samples currently saved in the collection.

    n_samples_capacity : :obj:`int`
        Number of samples that can be currently saved in the object.

    Note
    ----
    This is not intended for end-users. No sanity checks are performed here, we assume
    that such tests are performed beforehand in objects using this class.
    """

    def __init__(self, n_samples_increment, n_features):
        """Instantiates a `SamplesCollection` instance.

        Parameters
        ----------
        n_samples_increment : :obj:`int`
            Sets the amount of memory which is pre-allocated each time extra memory is
            required for new samples.

        n_features : :obj:`int`
            Number of features used during training.
        """

        self.n_samples_increment = n_samples_increment
        self.n_samples_capacity = n_samples_increment
        self.features = np.empty((n_samples_increment, n_features), dtype=float32)
        self.labels = np.empty(n_samples_increment, dtype=float32)
        self.n_samples = 0


@njit(void(get_type(SamplesCollection), get_array_2d_type(float32), float32[::1]))
def add_samples(samples, X, y):
    """Adds the features `X` and labels `y` to the collection of samples `samples`

    Parameters
    ----------
    samples : :obj:`SamplesCollection`
        The collection of samples where we want to append X and y

    X : :obj:`np.ndarray`, shape=(n_samples, n_features)
        Input features matrix to be appended

    y : :obj:`np.ndarray`
        Input labels vector to be appended

    """
    n_new_samples, n_features = X.shape
    n_current_samples = samples.n_samples
    n_samples_required = n_current_samples + n_new_samples
    capacity_missing = n_samples_required - samples.n_samples_capacity
    if capacity_missing >= 0:
        # We don't have enough room. Increase the memory reserved.
        if capacity_missing > samples.n_samples_increment:
            # If what's required is larger than the increment, we use what's missing
            # plus de minimum increment
            increment = capacity_missing + samples.n_samples_increment
        else:
            increment = samples.n_samples_increment

        n_samples_reserved = samples.n_samples_capacity + increment
        samples.features = resize_array(
            samples.features, n_current_samples, n_samples_reserved
        )
        samples.labels = resize_array(
            samples.labels, n_current_samples, n_samples_reserved
        )
        samples.n_samples_capacity += increment

    samples.features[n_current_samples:n_samples_required] = X
    samples.labels[n_current_samples:n_samples_required] = y
    samples.n_samples += n_new_samples
