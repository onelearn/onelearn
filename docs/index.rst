.. onelearn documentation master file, created by
   sphinx-quickstart on Sat Mar 21 22:37:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

This is ``onelearn``'s documentation
====================================

.. image:: https://travis-ci.org/onelearn/onelearn.svg?branch=master
   :target: https://travis-ci.org/onelearn/onelearn
.. image:: https://coveralls.io/repos/github/onelearn/onelearn/badge.svg?branch=master
   :target: https://coveralls.io/github/onelearn/onelearn?branch=master

.. image:: iterations.pdf


``onelearn`` stands for ONE-shot LEARNning. It is a small python package for **online learning** with ``Python``.
It provides :

   * **online** (or **one-shot**) learning algorithms: each sample is processed **once**, only a
     single pass is performed on the data
   * including **multi-class classification** and regression algorithms
   * For now, only *ensemble* methods, namely **Random Forests**

Usage
-----

``onelearn`` follows the ``scikit-learn`` API: you call ``fit`` instead ``partial_fit`` each
time a new bunch of data is available and use ``predict_proba`` or ``predict`` whenever you
need predictions.

.. code-block:: python

   from onelearn import AMFClassifier

   amf = AMFClassifier(n_classes=2)
   clf.partial_fit(X_train, y_train)
   y_pred = clf.predict_proba(X_test)[:, 1]

Installation
------------
The easiest way to install ``onelearn`` is using ``pip`` :

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



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
