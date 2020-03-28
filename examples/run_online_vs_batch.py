# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import os
import sys
from datetime import datetime
import logging
import warnings
from pathlib import Path
import pickle as pkl

sys.path.extend([".", ".."])
from experiments import (
    compute_regrets_and_batch,
    print_datasets,
    plot_online_vs_batch,
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

    for loader in loaders_online_vs_batch:
        X, y, dataset_name = loader(path)
        logging.info("-" * 64)
        logging.info("Working on dataset %s." % dataset_name)
        results = compute_regrets_and_batch(X, y)
        results["dataset"] = dataset_name
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(path, dataset_name + "_online_vs_batch_" + now + ".pkl")
        with open(filename, "wb") as f:
            pkl.dump(results, f)
            logging.info("Saved results in %s" % filename)

        logging.info("Results are available for the following classifiers:")
        with open(filename, "rb") as f:
            results = pkl.load(f)
            available_clfs = list(sorted(list(results["test_losses"].keys())))
            for clf_name in available_clfs:
                logging.info(" - " + clf_name)

        show_classifiers = available_clfs

        logging.info("Plotting results for the following classifiers:")
        for clf_name in show_classifiers:
            logging.info(" - " + clf_name)

        filename_pkl = filename
        p = Path(filename_pkl)
        with open(filename_pkl, "rb") as f:
            results = pkl.load(f)
            dataset_name = results["dataset"]
            filename_pdf = os.path.join(path, dataset_name + "_online_vs_batch.pdf")
            plot_online_vs_batch(
                results,
                show_classifiers=show_classifiers,
                remove_parameters=True,
                figsize=(4.5, 4),
                savefig=filename_pdf,
                type="auc",
            )
            logging.info("Saved plot in %s" % filename_pdf)
