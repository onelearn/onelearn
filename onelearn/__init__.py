# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

from .dummy import OnlineDummyClassifier
from .forest import AMFClassifier

__all__ = [
    "OnlineDummyClassifier",
    "AMFClassifier",
]
