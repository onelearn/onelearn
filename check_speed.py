# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import os
import sys
from time import time
import logging
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import onelearn
from onelearn.datasets import loaders_regrets
from onelearn import AMFClassifier

from skgarden import MondrianForestClassifier

sys.path.append("/Users/stephanegaiffas/Code/tick")


from tick.online import OnlineForestClassifier


np.set_printoptions(precision=3)


use_aggregation = True
n_estimators = 10
split_pure = False
dirichlet = 0.5
step = 1.0
random_state = 42

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
)
path = os.path.join(os.path.dirname(onelearn.__file__), "datasets/data")


# AMF est plus lent que OFC lorsque le nombre d'arbres est grands sur blobs...


# Precompile AMF
def precompile_amf():
    X, y = make_blobs(n_samples=5)
    n_classes = int(y.max() + 1)
    amf = AMFClassifier(
        n_classes=n_classes,
        random_state=0,
        use_aggregation=True,
        n_estimators=1,
        split_pure=False,
        dirichlet=0.5,
        step=1.0,
    )
    amf.partial_fit(X, y)
    amf.predict_proba(X)


logging.info("Precompilation of AMF...")
precompile_amf()
logging.info("Done.")

for loader in loaders_regrets:
    X, y, dataset_name = loader(path)
    logging.info("-" * 64)
    logging.info("Dataset %s." % dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    n_classes = int(y.max() + 1)

    amf = AMFClassifier(
        n_classes=n_classes,
        random_state=random_state,
        use_aggregation=use_aggregation,
        n_estimators=n_estimators,
        split_pure=split_pure,
        dirichlet=dirichlet,
        # n_samples_increment=,
        step=step,
        verbose=False,
    )
    ofc = OnlineForestClassifier(
        n_classes=n_classes,
        random_state=random_state,
        use_aggregation=use_aggregation,
        n_estimators=n_estimators,
        split_pure=split_pure,
        dirichlet=dirichlet,
        step=step,
        verbose=False,
    )
    mfc = MondrianForestClassifier(n_estimators=n_estimators, random_state=random_state)

    logging.info("Fitting AMF...")
    t1 = time()
    amf.partial_fit(X_train, y_train)
    t2 = time()
    logging.info("Done. time fit AMF: " + "%.2f" % (t2 - t1) + " seconds")

    logging.info("Fitting OFC...")
    t1 = time()
    ofc.partial_fit(X_train, y_train)
    t2 = time()
    logging.info("Done. time fit OFC:" + "%.2f" % (t2 - t1) + " seconds")

    logging.info("Fitting MFC...")
    t1 = time()
    mfc.partial_fit(X_train, y_train)
    t2 = time()
    logging.info("Done. time fit MFC:" + "%.2f" % (t2 - t1) + " seconds")

    logging.info("Prediction with AMF...")
    t1 = time()
    y_pred = amf.predict_proba(X_test)
    t2 = time()
    logging.info("Done. time predict AMF:" + "%.2f" % (t2 - t1) + " seconds")

    logging.info("Prediction with OFC...")
    t1 = time()
    y_pred = ofc.predict_proba(X_test)
    t2 = time()
    logging.info("Done. time predict OFC:" + "%.2f" % (t2 - t1) + " seconds")

    logging.info("Prediction with MFC...")
    t1 = time()
    y_pred = mfc.predict_proba(X_test)
    t2 = time()
    logging.info("Done. time predict MFC:" + "%.2f" % (t2 - t1) + " seconds")
