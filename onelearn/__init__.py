# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
from .dummy import OnlineDummyClassifier
from .forest import AMFClassifier
from .playground import run_playground_decision, run_playground_tree


__all__ = [
    "OnlineDummyClassifier",
    "AMFClassifier",
    "run_playground_decision",
    "run_playground_tree",
]
