# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
from .regrets import compute_regrets, plot_regrets
from .online_vs_batch import compute_regrets_and_batch, plot_online_vs_batch
from .n_tree_sensitivity import (
    compute_aucs_n_trees,
    read_data_n_trees,
    plot_comparison_n_trees,
)
from .utils import (
    classifier_colors,
    print_datasets,
    get_classifiers_batch,
    get_classifiers_online,
    get_classifiers_n_trees_comparison,
    log_loss,
    log_loss_single,
)
