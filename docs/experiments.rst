
Experiments
===========

This page explains how you can reproduce all the experiments from the paper

.. code-block:: bibtex

    @article{mourtada2019amf,
      title={AMF: Aggregated Mondrian Forests for Online Learning},
      author={Mourtada, Jaouad and Ga{\"\i}ffas, St{\'e}phane and Scornet, Erwan},
      journal={arXiv preprint arXiv:1906.10529},
      year={2019}
    }

Running the experiments requires the installation of ``scikit-garden``, for a comparison
with the Mondrian forests algorithm. This can be done as follows:

.. code-block:: bash

    git clone https://github.com/scikit-garden/scikit-garden.git && \
        cd scikit-garden && \
        python setup.py build install

in order to get the last version. All the scripts used to produce the figures from the paper
are available in the ``examples`` folder of the ``onelearn`` repository.
Clone the repository using

.. code-block:: bash

    git clone https://github.com/onelearn/onelearn.git

and go to the ``onelearn`` folder. Now, running the following scripts allows to reproduce all the
figures from the paper :

* ``python examples/plot_iterations.py``
* ``python examples/plot_decisions.py``
* ``python examples/plot_forest_effect.py``
* ``python examples/run_regrets_experiments.py``
* ``python examples/run_online_vs_batch.py``
* ``python examples/run_n_trees_sensitivity.py``

Note that the ``run_*`` scripts can take a while to run, in particular ``run_regrets_experiments.py``.
