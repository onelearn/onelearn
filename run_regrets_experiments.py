# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import os
import logging
import warnings
import pickle as pkl
from pathlib import Path
from datetime import datetime
import onelearn
from onelearn.experiments import compute_regrets, plot_regrets, print_datasets
from onelearn.datasets import loaders_regrets


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    path = os.path.join(os.path.dirname(onelearn.__file__), "datasets/data")
    print_datasets(loaders_regrets, path)

    for loader in loaders_regrets:
        X, y, dataset_name = loader(path)
        logging.info("-" * 64)
        logging.info("Working on dataset %s." % dataset_name)

        iterations, regrets, timings = compute_regrets(X, y)
        regrets_results = {
            "iterations": iterations,
            "regrets": regrets,
            "timings": timings,
            "dataset": dataset_name,
        }

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(path, dataset_name + "_" + now + ".pkl")
        with open(filename, "wb") as f:
            pkl.dump(regrets_results, f)
            logging.info("Saved results in %s" % filename)

        logging.info("")
        logging.info("Results are available for the following classifiers:")
        with open(os.path.join(path, filename), "rb") as f:
            results = pkl.load(f)
            available_clfs = list(sorted(list(results["regrets"].keys())))
            for clf_name in available_clfs:
                logging.info(" - " + clf_name)

        show_classifiers = tuple(available_clfs)

        logging.info("Plotting results for the following classifiers:")
        for clf_name in show_classifiers:
            logging.info(" - " + clf_name)

        filename_pkl = os.path.join(path, filename)
        p = Path(filename_pkl)
        with open(filename_pkl, "rb") as f:
            results = pkl.load(f)
            dataset_name = results["dataset"]
            filename_pdf_no_time = os.path.join(path, dataset_name + "_regrets.pdf")
            filename_pdf = filename_pdf_no_time
            plot_regrets(
                results,
                show_classifiers=show_classifiers,
                log_scale=True,
                remove_parameters=False,
                figsize=(4.5, 4),
                offset=100,
                savefig=filename_pdf,
            )
            logging.info("Saved plot in %s" % filename_pdf)
