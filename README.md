
[![Build Status](https://travis-ci.com/onelearn/onelearn.svg?branch=master)](https://travis-ci.com/onelearn/onelearn)
[![Documentation Status](https://readthedocs.org/projects/onelearn/badge/?version=latest)](https://onelearn.readthedocs.io/en/latest/?badge=latest)
[![GitHub stars](https://img.shields.io/github/stars/onelearn/onelearn)](https://github.com/onelearn/onelearn/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/onelearn/onelearn)](https://github.com/onelearn/onelearn/issues)
[![GitHub license](https://img.shields.io/github/license/onelearn/onelearn)](https://github.com/onelearn/onelearn/blob/master/LICENSE)
[![Coverage Status](https://coveralls.io/repos/github/onelearn/onelearn/badge.svg)](https://coveralls.io/github/onelearn/onelearn)
	
# `onelearn`: Online learning in Python

---

[Documentation](https://onelearn.readthedocs.io) | [Reproduce experiments](https://onelearn.readthedocs.io/en/latest/experiments.html) |

---

`onelearn` stands for ONE-shot LEARNning. It is a small python package for **online learning** with ``Python``.
It provides :

- **online** (or **one-shot**) learning algorithms: each sample is processed **once**, only a 
  single pass is performed on the data
- including **multi-class classification** and regression algorithms
- For now, only *ensemble* methods, namely **Random Forests**


## Installation

The easiest way to install ``onelearn`` is using ``pip``

    pip install onelearn


But you can also use the latest development from github directly with

    pip install git+https://github.com/onelearn/onelearn.git

## References

    @article{mourtada2019amf,
      title={AMF: Aggregated Mondrian Forests for Online Learning},
      author={Mourtada, Jaouad and Ga{\"\i}ffas, St{\'e}phane and Scornet, Erwan},
      journal={arXiv preprint arXiv:1906.10529},
      year={2019}
    }
 