
.. _playground:

Playgrounds
===========

Two "playgrounds" are proposed in ``onelearn``, in order to help understand the :obj:`AMFClassifier`
algorithm. The playgrounds require ``streamlit``, ``bokeh`` and ``matplotlib`` to run.
If you pip installed ``onelearn``, you can simply use

.. code-block:: python

   from onelearn import run_playground_decision

   run_playground_decision()

to run the decision function playground, and use

.. code-block:: python

   from onelearn import run_playground_tree

   run_playground_tree()

to run the tree playground. If you git cloned ``onelearn`` you can run directly ``streamlit``
using

.. code-block:: bash

   streamlit run examples/playground_decision.py

or

.. code-block:: bash

   streamlit run examples/playground_tree.py

For the ``playground_decision`` playground, the following webpage should automatically open in your web-browser:

.. image:: images/playground.png

