
Classification
==============

For now, ``onelearn`` provides mainly the :obj:`AMFClassifier` class for multi-class
classification.
Its usage follows the ``scikit-learn`` API, namely a ``partial_fit``, ``predict_proba``
and ``predict`` methods to respectively fit, predict class probabilities and labels.
The :obj:`AMFClassifier` with default parameters is created using

.. code-block:: python

    from onelearn import AMFClassifier

    amf = AMFClassifier(n_classes=2)

where ``n_classes`` **must be** provided for the construction of the object.
Also, a baseline dummy classifier is provided by :obj:`OnlineDummyClassifier`, see below.

.. autosummary::
   :toctree: generated/

   onelearn.AMFClassifier
   onelearn.OnlineDummyClassifier


About :obj:`AMFClassifier`
--------------------------

The :obj:`AMFClassifier` class implements the "Aggregated Mondrian Forest" classifier
for online learning (see reference below). This algorithm is **truly online**, in the sense
that a **single pass** is performed, and that predictions can be produced **anytime**.

For multi-class classification with :math:`C` classes, we observe, **before time** :math:`t`,
pairs of features and labels :math:`(x_1, y_1), \ldots, (x_{t-1}, y_{t-1})` where
:math:`y_s \in \{ 1, \ldots, C \}` for each :math:`s = 1, \ldots, t-1`.

Each node in a tree predicts according to the distribution of the labels
it contains. This distribution is regularized using a Dirichlet (a.k.a "Jeffreys") prior
with parameter :math:`\alpha > 0` which corresponds to the ``dirichlet`` parameter in :obj:`AMFClassifier`.
Each node :math:`\mathbf v` of a tree predicts, before time :math:`t`, the probability of
class :math:`c` as

.. math::
  \widehat y_{\mathbf v, t} (c) = \frac{n_{{\mathbf v}, t} (c) + \alpha}{t + C \alpha}

for any :math:`c = 1, \ldots, C`, where :math:`n_{{\mathbf v}, t}(c)` is the number of
samples of class :math:`c` in node :math:`\mathbf v` before time :math:`t`.
This formula is therefore simply a regularized version of the class frequencies.

Each node :math:`\mathbf v` in the tree corresponds to a *cell* denoted :math:`\mathrm{cell}({\mathbf v})`,
which corresponds to a hyper-rectangular subset of the features space.
The predictions of a node, before time :math:`t`, are evaluated by computing its cumulative loss as

.. math::
    L_{\mathbf v, t} = \sum_{1 \leq s \leq t \,:\, x_s \in \mathrm{cell}(\mathbf v)} \ell (\widehat y_{\mathbf v, s}, y_s),

which is the sum of the prediction losses of all the samples whose features belong to
:math:`\mathrm{cell}({\mathbf v})`.
By default, we consider, for multi-class classification, the *logarithmic* loss
:math:`\ell (\widehat y, y) = - \log (\widehat y(y))`  for :math:`y \in \{ 1, \ldots, C \}`.
The loss can be changed using the ``loss`` parameter from :obj:`AMFClassifier` (however
only ``loss="log"`` is supported for now).

Given a vector of features :math:`x` and any subtree :math:`\mathcal T` of the current tree,
we define :math:`\mathbf v_{\mathcal T}(x)` as the leaf of :math:`\mathcal T` containing
:math:`x` (namely :math:`x` belongs to its cell).
The prediction at time :math:`t` of the subtree :math:`\mathcal T` for :math:`x` is given by

.. math::
    {\widehat y}_{\mathcal T, t} (x) = {\widehat y}_{\mathbf v_{\mathcal T} (x), t},

namely the prediction of :math:`\mathcal T` is simply the prediction of the leaf of
:math:`\mathcal T` containing :math:`x`.
We define also the cumulative loss of a subtree :math:`\mathcal T` at time :math:`t` as

.. math::
    L_{t} (\mathcal T) = \sum_{s=1}^t \ell ({\widehat y}_{\mathcal T, t} (x_s), y_s).

When ``use_aggregation`` is ``True`` (the highly recommended default), the prediction function
of a single tree in :obj:`AMFClassifier` is given, at step :math:`t`, by

.. math::
    \widehat {f_t}(x) = \frac{\sum_{\mathcal T} \pi (\mathcal T) e^{-\eta L_{t-1} (\mathcal T)}
    \widehat y_{\mathcal T, t} (x)}{\sum_{\mathcal T} \pi (\mathcal T) e^{-\eta L_{t-1} (\mathcal T)}},

where the sum is over all subtrees :math:`\mathcal T` of the current tree, and where the *prior*
:math:`\pi` on subtrees is the probability distribution defined by

.. math::
    \pi (\mathcal T) = 2^{- | \mathcal T |},

where :math:`|\mathcal T|` is the number of nodes in :math:`\mathcal T` and :math:`\eta > 0`
is the *learning rate* that can be tuned using the ``step`` parameter in :obj:`AMFClassifier`
(theoretically, the default value ``step=1.0`` is the best, and usually performs just fine).

Note that :math:`\pi` is the distribution of the branching process with branching probability :math:`1 / 2`
at each node of the complete binary tree, with exactly two children when it branches.
This aggregation procedure is a **non-greedy way to prune trees**: the weights do not depend only
on the quality of one single split but rather on the performance of each subsequent split.

The computation of :math:`\widehat {f_t}(x)` can seem **computationally infeasible**, since it
involves a sum over all possible subtrees of the current tree, which is exponentially large.
Besides, it requires to keep in memory one weight :math:`e^{-\eta L_{t-1} (\mathcal T)}`
for all the subtrees :math:`\mathcal T`, which seems exponentially prohibitive as well !

This is precisely where the magics of :obj:`AMFClassifier` resides: it turns out that
**we can compute exactly and very efficiently** :math:`\widehat {f_t}(x)` thanks to the
prior choice :math:`\pi` together with an adaptation of the Context Tree Weighting algorithm,
for which more technical details are provided in the paper cited below.
The interested reader can find also, in the paper cited below, the construction details of
the online tree construction, which is based on the *Mondrian process* and *Mondrian Forests*.

Finally, we use :math:`M` trees in the forest, all of them follow the same randomized construction.
The predictions, for a vector :math:`x`, of each tree :math:`m = 1, \ldots, M`, are
denoted :math:`\widehat {f_t}^{(m)}(x)`. The prediction of the forest is simply the average
given by

.. math::
    \frac 1 M \sum_{m=1}^M \widehat {f_t}^{(m)}(x).

The number of trees :math:`M` in the forest can be tuned with the ``n_estimators`` parameter
from :obj:`AMFClassifier`, the default value is 10, but the larger the better
(but requires more computations and memory).

.. note::

    When creating a classifier instance, such as a :obj:`AMFClassifier` object, the
    number ``n_classes`` of classes **must be** provided to the constructor.

.. note::

    All the parameters of :obj:`AMFClassifier` become **read-only** after the first call
    to ``partial_fit``

References
----------

.. code-block:: bibtex

    @article{mourtada2019amf,
      title={AMF: Aggregated Mondrian Forests for Online Learning},
      author={Mourtada, Jaouad and Ga{\"\i}ffas, St{\'e}phane and Scornet, Erwan},
      journal={arXiv preprint arXiv:1906.10529},
      year={2019}
    }
