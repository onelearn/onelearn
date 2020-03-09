# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import os
import pandas as pd
import numpy as np
import zipfile
from sklearn.preprocessing import MinMaxScaler


def load_adult(path):
    archive = zipfile.ZipFile(os.path.join(path, "adult.csv.zip"), "r")
    with archive.open("adult.csv") as f:
        data = pd.read_csv(f, header=None)
    y = data.pop(13)
    discrete = [1, 3, 4, 5, 6, 7, 8, 12]
    continuous = list(set(range(13)) - set(discrete))
    X_continuous = MinMaxScaler().fit_transform(data[continuous].astype("float32"))
    data_discrete = pd.get_dummies(data[discrete], prefix_sep="#")
    X_discrete = data_discrete.values
    y = pd.get_dummies(y).values[:, 1]
    X = np.hstack((X_continuous, X_discrete)).astype("float32")
    X = MinMaxScaler().fit_transform(X)
    return X, y, "adult"


def load_bank(path):
    archive = zipfile.ZipFile(os.path.join(path, "bank.csv.zip"), "r")
    with archive.open("bank.csv") as f:
        data = pd.read_csv(f)
    y = data.pop("y")
    discrete = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "campaign",
        "poutcome",
    ]
    continuous = ["age", "balance", "duration", "pdays", "previous"]
    X_continuous = MinMaxScaler().fit_transform(data[continuous].astype("float32"))
    data_discrete = pd.get_dummies(data[discrete], prefix_sep="#")
    X_discrete = data_discrete.values
    y = pd.get_dummies(y).values[:, 1]
    X = np.hstack((X_continuous, X_discrete)).astype("float32")
    X = MinMaxScaler().fit_transform(X)
    return X, y, "bank"


def load_car(path):
    archive = zipfile.ZipFile(os.path.join(path, "car.csv.zip"), "r")
    with archive.open("car.csv") as f:
        data = pd.read_csv(f, header=None)
    y = data.pop(6)
    y = np.argmax(pd.get_dummies(y).values, axis=1)
    X = pd.get_dummies(data, prefix_sep="#").values.astype("float32")
    X = MinMaxScaler().fit_transform(X)
    return X, y, "car"


def load_cardio(path):
    archive = zipfile.ZipFile(os.path.join(path, "cardiotocography.csv.zip"), "r")
    with archive.open("cardiotocography.csv",) as f:
        data = pd.read_csv(f, sep=";", decimal=",")

    data.drop(
        [
            "FileName",
            "Date",
            "SegFile",
            "A",
            "B",
            "C",
            "D",
            "E",
            "AD",
            "DE",
            "LD",
            "FS",
            "SUSP",
        ],
        axis=1,
        inplace=True,
    )
    # A 10-class label
    y_class = data.pop("CLASS").values
    y_class -= 1
    # A 3-class label
    y_nsp = data.pop("NSP").values
    y_nsp -= 1
    continuous = [
        "b",
        "e",
        "LBE",
        "LB",
        "AC",
        "FM",
        "UC",
        "ASTV",
        "MSTV",
        "ALTV",
        "MLTV",
        "DL",
        "DS",
        "DP",
        "Width",
        "Min",
        "Max",
        "Nmax",
        "Nzeros",
        "Mode",
        "Mean",
        "Median",
        "Variance",
    ]
    discrete = ["Tendency"]
    X_continuous = MinMaxScaler().fit_transform(data[continuous].astype("float32"))
    data_discrete = pd.get_dummies(data[discrete], prefix_sep="#")
    X_discrete = data_discrete.values
    X = np.hstack((X_continuous, X_discrete)).astype("float32")
    X = MinMaxScaler().fit_transform(X)
    return X, y_nsp, "cardio"


def load_churn(path):
    archive = zipfile.ZipFile(os.path.join(path, "churn.csv.zip"), "r")
    with archive.open("churn.csv") as f:
        data = pd.read_csv(f)
    y = data.pop("Churn?")
    discrete = [
        "State",
        "Area Code",
        "Int'l Plan",
        "VMail Plan",
    ]

    continuous = [
        "Account Length",
        "Day Mins",
        "Day Calls",
        "Eve Calls",
        "Day Charge",
        "Eve Mins",
        "Eve Charge",
        "Night Mins",
        "Night Calls",
        "Night Charge",
        "Intl Mins",
        "Intl Calls",
        "Intl Charge",
        "CustServ Calls",
        "VMail Message",
    ]
    X_continuous = MinMaxScaler().fit_transform(data[continuous].astype("float32"))
    data_discrete = pd.get_dummies(data[discrete], prefix_sep="#")
    X_discrete = data_discrete.values
    y = pd.get_dummies(y).values[:, 1]
    X = np.hstack((X_continuous, X_discrete)).astype("float32")
    X = MinMaxScaler().fit_transform(X)
    return X, y, "churn"


def load_default_cb(path):
    archive = zipfile.ZipFile(os.path.join(path, "default_cb.csv.zip"), "r")
    with archive.open("default_cb.csv") as f:
        data = pd.read_csv(f)
    continuous = [
        "AGE",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "LIMIT_BAL",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]
    discrete = [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
    ]
    _ = data.pop("ID")
    y = data.pop("default payment next month")
    X_continuous = MinMaxScaler().fit_transform(data[continuous].astype("float32"))
    data_discrete = pd.get_dummies(data[discrete], prefix_sep="#")
    X_discrete = data_discrete.values
    y = pd.get_dummies(y).values[:, 1]
    X = np.hstack((X_continuous, X_discrete)).astype("float32")
    X = MinMaxScaler().fit_transform(X)
    return X, y, "default_cb"


def load_letter(path):
    archive = zipfile.ZipFile(os.path.join(path, "letter.csv.zip"), "r")
    with archive.open("letter.csv") as f:
        data = pd.read_csv(f)
    data.drop(["Unnamed: 0"], axis=1, inplace=True)
    y = data.pop("y").values
    X = data.values.astype("float32")
    X = MinMaxScaler().fit_transform(X)
    return X, y, "letter"


def load_satimage(path):
    archive = zipfile.ZipFile(os.path.join(path, "satimage.csv.zip"), "r")
    with archive.open("satimage.csv") as f:
        data = pd.read_csv(f)
    data.drop(["Unnamed: 0"], axis=1, inplace=True)
    y = data.pop("y").values
    X = data.values.astype("float32")
    X = MinMaxScaler().fit_transform(X)
    return X, y, "satimage"


def load_sensorless(path):
    archive = zipfile.ZipFile(os.path.join(path, "sensorless.csv.zip"), "r")
    with archive.open("sensorless.csv") as f:
        data = pd.read_csv(f, sep=" ", header=None)
    y = data.pop(48).values
    y -= 1
    X = MinMaxScaler().fit_transform(data.astype("float32"))
    return X, y, "sensorless"


def load_spambase(path):
    archive = zipfile.ZipFile(os.path.join(path, "spambase.csv.zip"), "r")
    with archive.open("spambase.csv") as f:
        data = pd.read_csv(f, header=None)
    y = data.pop(57).values
    X = MinMaxScaler().fit_transform(data.astype("float32"))
    return X, y, "spambase"


loaders_regrets = [
    load_adult,
    load_bank,
    load_car,
    load_cardio,
    load_churn,
    load_default_cb,
    load_letter,
    load_satimage,
    load_sensorless,
    load_spambase,
]


loaders_online_vs_batch = [load_adult, load_bank, load_default_cb, load_spambase]
