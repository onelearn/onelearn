# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

from time import time

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from onelearn.forest import AMFClassifier

np.set_printoptions(precision=3)

n_samples = 5000
n_features = 100
n_classes = 2

X, y = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    cluster_std=3.0,
    centers=n_classes,
    random_state=123,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# Precompilation
amf = AMFClassifier(
    n_classes=n_classes,
    random_state=1234,
    use_aggregation=True,
    n_estimators=1,
    split_pure=True,
    dirichlet=0.5,
    step=1.0,
)
amf.partial_fit(X_train[:5], y_train[:5])

amf.predict_proba(X_test[:5])

# print("Test for batch training")
# repeats = 5
# for repeat in range(1, repeats + 1):
#     print("-" * 16)
#     amf = AMFClassifier(
#         n_classes=n_classes,
#         random_state=1234,
#         use_aggregation=True,
#         n_estimators=20,
#         split_pure=True,
#         dirichlet=0.5,
#         step=1.0,
#     )
#     t1 = time()
#     amf.partial_fit(X_train, y_train)
#     t2 = time()
#     print("time fit % d:" % repeat, t2 - t1, "seconds")
#
#     t1 = time()
#     y_pred = amf.predict_proba(X_test)
#     t2 = time()
#     print("time predict %d:" % repeat, t2 - t1, "seconds")
#     roc_auc = roc_auc_score(y_test, y_pred[:, 1])
#     print("ROC AUC: %.2f" % roc_auc)


print("Test for online training")
repeats = 5
for repeat in range(1, repeats + 1):
    print("-" * 16)
    amf = AMFClassifier(
        n_classes=n_classes,
        random_state=1234,
        use_aggregation=True,
        n_estimators=20,
        split_pure=True,
        dirichlet=0.5,
        step=1.0,
    )
    t1 = time()
    for i in range(X_train.shape[0]):
        amf.partial_fit(X_train[i].reshape(1, n_features), np.array([y_train[i]]))
    t2 = time()
    print("time fit % d:" % repeat, t2 - t1, "seconds")

    t1 = time()
    y_pred = amf.predict_proba(X_test)
    t2 = time()
    print("time predict %d:" % repeat, t2 - t1, "seconds")
    roc_auc = roc_auc_score(y_test, y_pred[:, 1])
    print("ROC AUC: %.2f" % roc_auc)
