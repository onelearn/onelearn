# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause


class TestSamplesCollection(object):
    def test_samples_collection(self):
        import numpy as np
        from onelearn.sample import SamplesCollection, add_samples

        n_samples_increment = 10
        n_features = 3

        samples = SamplesCollection(n_samples_increment, n_features)
        X = np.zeros((4, n_features), dtype="float32")
        y = np.zeros(4, dtype="float32")
        add_samples(samples, X, y)
        assert samples.n_samples_increment == 10
        assert samples.n_samples_capacity == 10
        assert samples.n_samples == 4

        add_samples(samples, X, y)
        assert samples.n_samples_capacity == 10
        assert samples.n_samples == 8

        add_samples(samples, X, y)
        assert samples.n_samples_capacity == 20
        assert samples.n_samples == 12

        add_samples(samples, X, y)
        assert samples.n_samples_capacity == 20
        assert samples.n_samples == 16

        X = np.zeros((8, n_features), dtype="float32")
        y = np.zeros(8, dtype="float32")
        add_samples(samples, X, y)
        assert samples.n_samples_capacity == 30
        assert samples.n_samples == 24

        X = np.zeros((12, n_features), dtype="float32")
        y = np.zeros(12, dtype="float32")
        add_samples(samples, X, y)
        assert samples.n_samples_capacity == 40
        assert samples.n_samples == 36

        X = np.zeros((4, n_features), dtype="float32")
        y = np.zeros(4, dtype="float32")
        add_samples(samples, X, y)
        assert samples.n_samples_capacity == 50
        assert samples.n_samples == 40

        add_samples(samples, X, y)
        assert samples.n_samples_capacity == 50
        assert samples.n_samples == 44

        add_samples(samples, X, y)
        assert samples.n_samples_capacity == 50
        assert samples.n_samples == 48
