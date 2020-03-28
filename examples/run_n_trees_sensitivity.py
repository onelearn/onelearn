# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import os
import sys
from datetime import datetime
import logging
import warnings
import pickle as pkl

sys.path.extend([".", ".."])
from experiments import (
    print_datasets,
    compute_aucs_n_trees,
    read_data_n_trees,
    plot_comparison_n_trees,
)
import onelearn
from onelearn.datasets import loaders_online_vs_batch


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    path = os.path.join(os.path.dirname(onelearn.__file__), "datasets/data")
    print_datasets(loaders_online_vs_batch, path)

    filenames = []
    for loader in loaders_online_vs_batch:
        X, y, dataset_name = loader(path)
        logging.info("-" * 64)
        logging.info("Working on dataset %s." % dataset_name)
        results = {}
        test_aucs = compute_aucs_n_trees(X, y)
        results["test_aucs"] = test_aucs
        results["dataset"] = dataset_name
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(path, dataset_name + "_" + now + ".pkl")
        with open(filename, "wb") as f:
            pkl.dump(results, f)
            logging.info("Saved results in %s" % filename)
        filenames.append(filename)

    df = read_data_n_trees(filenames)
    filename_pdf = os.path.join(path, "comparison_n_trees.pdf")
    plot_comparison_n_trees(df, filename=filename_pdf, legend=True)
