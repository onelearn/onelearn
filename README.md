
[![Build Status](https://travis-ci.com/onelearn/onelearn.svg?branch=master)](https://travis-ci.com/onelearn/onelearn)
[![Coverage Status](https://coveralls.io/repos/github/onelearn/onelearn/badge.svg)](https://coveralls.io/github/onelearn/onelearn)

# `onelearn`: machine learning lgorithms for ONline LEARNing

This `GitHub` repository contains for now the algorithms described in the paper

> *AMF: Aggregated Mondrian Forests for Online Learning*
> 
> by J. Mourtada, S. Gaïffas and E. Scornet
> 
> arXiv link: http://arxiv.org/abs/1906.10529

It provides mainly the `AMFClassifier` and the `AMFRegressor`, together with a weak 
baseline called `OnlineDummyClassifier`. 


Running the experiments requires the installation of scikit-garden, for a comparison
with Mondrian forests.
```bash
git clone https://github.com/scikit-garden/scikit-garden.git && \
    cd scikit-garden && \
    python setup.py build install
```


# Reproducing the experiments from the paper

## Figure 1

The decision function along iterations

```bash
python plot_iterations.py
```

## Figure 3

The decision function over some toy datasets

```bash
python plot_decisions.py
```

## Figure 6

Comparisons of average losses over 10 datasets

## Figure 7

Test AUC

## Figure 8

Sensitivity to the number of trees

