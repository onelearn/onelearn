
This is ``onelearn``'s documentation
====================================

.. image:: https://travis-ci.org/onelearn/onelearn.svg?branch=master
   :target: https://travis-ci.org/onelearn/onelearn
.. image:: https://readthedocs.org/projects/onelearn/badge/?version=latest
   :target: https://onelearn.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://img.shields.io/pypi/pyversions/onelearn
   :alt: PyPI - Python Version
.. image:: https://img.shields.io/pypi/wheel/onelearn
   :alt: PyPI - Wheel
.. image:: https://img.shields.io/github/stars/onelearn/onelearn
   :alt: GitHub stars
   :target: https://github.com/onelearn/onelearn/stargazers
.. image:: https://img.shields.io/github/issues/onelearn/onelearn
   :alt: GitHub issues
   :target: https://github.com/onelearn/onelearn/issues
.. image:: https://img.shields.io/github/license/onelearn/onelearn
   :alt: GitHub license
   :target: https://github.com/onelearn/onelearn/blob/master/LICENSE
.. image:: https://coveralls.io/repos/github/onelearn/onelearn/badge.svg?branch=master
   :target: https://coveralls.io/github/onelearn/onelearn?branch=master


onelearn stands for ONE-shot LEARNning. It is a small python package for **online learning** with Python.
It provides :

   * **online** (or **one-shot**) learning algorithms: each sample is processed **once**, only a
     single pass is performed on the data
   * including **multi-class classification** and regression algorithms
   * For now, only *ensemble* methods, namely **Random Forests**

Usage
-----

onelearn follows the scikit-learn API: you call fit instead of partial_fit each
time a new bunch of data is available and use predict_proba or predict whenever you
need predictions.

.. code-block:: python

   from onelearn import AMFClassifier

   amf = AMFClassifier(n_classes=2)
   clf.partial_fit(X_train, y_train)
   y_pred = clf.predict_proba(X_test)[:, 1]

Each time you call partial_fit the algorithm updates its decision function using the
new data as illustrated in the next figure.

.. image:: images/iterations.pdf

Installation
------------
The easiest way to install onelearn is using pip :

.. code-block:: bash

    pip install onelearn


But you can also use the latest development from github directly with ::

    pip install git+https://github.com/onelearn/onelearn.git


Where to go from here?
----------------------

To know more about onelearn, check out our :ref:`example gallery <sphx_glr_auto_examples>` or
browse through the module reference using the left navigation bar.


.. toctree::
   :maxdepth: 2
   :hidden:

   classification
   regression
   experiments
   playground
   auto_examples/index
