# PyGIP

[![PyPI - Version](https://img.shields.io/pypi/v/PyGIP)](https://pypi.org/project/PyGIP)
[![Build Status](https://img.shields.io/github/actions/workflow/status/LabRAI/PyGIP/docs.yml)](https://github.com/LabRAI/PyGIP/actions)
[![License](https://img.shields.io/github/license/LabRAI/PyGIP.svg)](https://github.com/LabRAI/PyGIP/blob/main/LICENSE)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pygip)](https://github.com/LabRAI/PyGIP)
[![Issues](https://img.shields.io/github/issues/LabRAI/PyGIP)](https://github.com/LabRAI/PyGIP)
[![Pull Requests](https://img.shields.io/github/issues-pr/LabRAI/PyGIP)](https://github.com/LabRAI/PyGIP)
[![Stars](https://img.shields.io/github/stars/LabRAI/PyGIP)](https://github.com/LabRAI/PyGIP)
[![GitHub forks](https://img.shields.io/github/forks/LabRAI/PyGIP)](https://github.com/LabRAI/PyGIP)

PyGIP is a Python library designed for experimenting with graph-based model extraction attacks and defenses. It provides
a modular framework to implement and test attack and defense strategies on graph datasets.

## How to Cite

If you find it useful, please considering cite the following work:

```bibtex
@article{li2025intellectual,
  title={Intellectual Property in Graph-Based Machine Learning as a Service: Attacks and Defenses},
  author={Li, Lincan and Shen, Bolin and Zhao, Chenxi and Sun, Yuxiang and Zhao, Kaixiang and Pan, Shirui and Dong, Yushun},
  journal={arXiv preprint arXiv:2508.19641},
  year={2025}
}
```


## Installation

PyGIP supports both CPU and GPU environments. Make sure you have Python installed (version >= 3.8, <3.13).

### Base Installation

First, install the core package:

```bash
pip install PyGIP
```

This will install PyGIP with minimal dependencies.

### CPU Version

```bash
pip install "PyGIP[torch,dgl]" \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  -f https://data.dgl.ai/wheels/repo.html
```

### GPU Version (CUDA 12.1)

```bash
pip install "PyGIP[torch,dgl]" \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple \
  -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
```

## Quick Start

Here’s a simple example to launch a Model Extraction Attack using PyGIP:

```python
from datasets import Cora
from models.attack import ModelExtractionAttack0

# Load the Cora dataset
dataset = Cora()

# Initialize the attack with a sampling ratio of 0.25
mea = ModelExtractionAttack0(dataset, 0.25)

# Execute the attack
mea.attack()
```

This code loads the Cora dataset, initializes a basic model extraction attack (`ModelExtractionAttack0`), and runs the
attack with a specified sampling ratio.

And a simple example to run a Defense method against Model Extraction Attack:

```python
from datasets import Cora
from models.defense import RandomWM

# Load the Cora dataset
dataset = Cora()

# Initialize the attack with a sampling ratio of 0.25
med = RandomWM(dataset, 0.25)

# Execute the defense
med.defend()
```

which runs the Random Watermarking Graph to defend against MEA.

If you want to use cuda, please set environment variable:

```shell
export PYGIP_DEVICE=cuda:0
```

## Implementation & Contributors Guideline

Refer to [Implementation Guideline](.github/IMPLEMENTATION.md)

Refer to [Contributors Guideline](.github/CONTRIBUTING.md)

## License

[BSD 2-Clause License](LICENSE)

## Contact

For questions or contributions, please contact blshen@fsu.edu.
