# BayesMCDM: Bayesian Multi-Criteria Decision-Making Toolkit <img src="https://raw.githubusercontent.com/Majeed7/BayesMCDM/main/figures/logo.png"  alt="bayesmcdm_logo" align="right" height="200px"/>

[![PyPI version](https://badge.fury.io/py/bayesmcdm.svg)](https://badge.fury.io/py/bayesmcdm)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
<!-- [![Tests](https://github.com/majidmohammadi/bayesmcdm/actions/workflows/tests.yml/badge.svg)](https://github.com/majidmohammadi/bayesmcdm/actions/workflows/tests.yml) -->
<!-- [![Coverage Status](https://coveralls.io/repos/github/majidmohammadi/bayesmcdm/badge.svg?branch=main)](https://coveralls.io/github/majidmohammadi/bayesmcdm?branch=main) -->
<!-- [![Documentation Status](https://readthedocs.org/projects/bayesmcdm/badge/?version=latest)](https://bayesmcdm.readthedocs.io/en/latest/?badge=latest) -->

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bayesmcdm.svg)](https://pypi.org/project/BayesMCDM)
[![PyPI status](https://img.shields.io/pypi/status/bayesmcdm.svg?color=blue)](https://pypi.org/project/bayesmcdm)
[![Downloads](https://static.pepy.tech/badge/bayesmcdm?style=flat-square)](https://pepy.tech/project/bayesmcdm)

[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](https://github.com/Majeed7/bayesmcdm/issues)
[![Last Commit](https://img.shields.io/github/last-commit/Majeed7/bayesmcdm)](https://github.com/Majeed7/bayesmcdm/commits/main)

BayesMCDM is a Python library for Bayesian modeling of various multi-criteria decision-making (MCDM) methods. This toolkit enables robust, probabilistic analysis of decision problems by incorporating uncertainty in preferences and criteria weights. **The project is under active development**—expect more models and functionalities to be added over time.

**You can solve your decision problems directly in your browser—no software installation needed.**
   
   

## Supported Preference Types

BayesMCDM currently supports several types of preference formats for each method.

Support for **group aggregation** of preferences and **decision-maker clustering** is available or under development for these formats, enabling analysis of collective decisions and identification of groups with homogeneous preferences.

- **Standard (crisp) scale (e.g., 1-9 or any other crisp scale)**
    *Fully implemented, tested, and ready for use.*

- **Interval preferences**  
    *Under construction.*

- **Triangular fuzzy preferences**  
    *Under construction.*

- **Gaussian (normal) preferences**  
    *Under construction.*

## Supported Methods
The following methods are supported, each with an interactive Google Colab link. These links open ready-to-run notebooks containing example scripts for each method. You can enter your own data and solve your MCDM problem directly in your browser—no installation required.

### 1. Analytic Hierarchy Process (AHP)
AHP is a structured technique for organizing and analyzing complex decisions, based on pairwise comparisons among all criteria.
- **Standard scale (crisp 1-9):**
    - [Aggregation](https://colab.research.google.com/drive/1DLjKhuP29mEiDufejISw8mAMg0MRTIca)
    - [Clustering](https://colab.research.google.com/drive/1JkFNUtagwvCac2Uqh86Je2vfSa2Wil1F)

### 2. Best-Worst Method (BWM)
BWM uses the best and worst criteria to derive optimal weights through pairwise comparisons.
- **Standard scale (crisp 1-9):**
    - [Aggregation](https://colab.research.google.com/drive/12X4of4jk5M9mkwQOPehih9XLm7tBkhnL)
    - [Clustering](https://colab.research.google.com/drive/17NRtP7-YXnwl_QcKM31jT2u9VKm90es7)

### 3. SWING Method
The SWING method elicits weights by asking decision-makers to "swing" criteria from worst to best, reflecting their relative importance.
- **Standard scale (crisp):**
    - [Aggregation](https://colab.research.google.com/drive/13KPG9WkHnUrYKdAZq5IJAEkqQHD5izHx)
    - [Clustering](https://colab.research.google.com/drive/13KPG9WkHnUrYKdAZq5IJAEkqQHD5izHx)

### 4. Point Allocation
Point Allocation allows decision-makers to distribute a fixed number of points among criteria to indicate their importance.
- **Standard scale (crisp):**
    - [Aggregation](https://colab.research.google.com/drive/1Dh6AB3kCa1pkXdkiKzKDMix6_1GoJBw-)
    - [Clustering](https://colab.research.google.com/drive/1oJ4q36DVY1ZYDZWMmHWgzwkGv54i_hUb)

### 5. Weight Analyzer
The Weight Analyzer provides tools for analyzing the weights (and not preferences) that are computed by an MCDM method.
- **Standard scale (weights):**
    - [Aggregation](https://colab.research.google.com/drive/1148-72AHpfxGhfimLCjs-84prgoQ0wlr)
    - [Clustering](https://colab.research.google.com/drive/1vnOkOfGZX5782yzlLTQqEPEbncbmLoV4)

## Visualization

BayesMCDM offers several visualization tools to help interpret Bayesian results:

### Credal Ranking

<!-- ![Credal ranking example](https://raw.githubusercontent.com/Majeed7/BayesMCDM/main/figures/credal_ranking.png) -->

<p align="center">
    <img src="https://raw.githubusercontent.com/Majeed7/BayesMCDM/main/figures/credal_ranking.png" alt="Weight Distribution example" height="500px"/>
</p>
*Figure: Example of a credal ranking plot showing the probability of each criterion being more important than another.*

**Credal ranking** visualizes the probabilistic ranking of criteria, showing the likelihood of each criterion occupying each rank based on the posterior weight distributions. This helps in understanding the robustness and uncertainty of the ranking outcomes.

### Weight Distributions

<p align="center">
    <img src="https://raw.githubusercontent.com/Majeed7/BayesMCDM/main/figures/ridge_plot.png" alt="Weight Distribution example" height="500px"/>
</p>
* Figure: Example of weight distributions *


**Weight distributions** plots display the full posterior distributions of criteria weights, allowing users to assess uncertainty, variability, and the impact of preference types on the final weights.




## PyPI Package & Installation

BayesMCDM is available as a PyPI package. You can install it directly using pip:

```bash
pip install bayesmcdm
```

After installation, you can use BayesMCDM in your Python scripts or Jupyter notebooks. Here is a minimal example (from the AHP Aggregation notebook) to get you started:

```python
import numpy as np
from BayesMCDM import AHP

# Define the PCM array (example for 5 criteria, 5 decision makers)
PCM = [
    [
        [1,   3,   5,     4, 7],
        [1/3, 1,   3,     2, 5],
        [1/5, 1/3, 1,   1/2, 3],
        [1/4, 1/2,   2,   1, 3],
        [1/7, 1/5, 1/3, 1/3, 1],
    ],
        [
        [1,     4,   3,    5,  8],
        [1/4,   1,   4,    3,  6],
        [1/3, 1/4,   1,    1,  5],
        [1/5, 1/3,   1,    1,  7],
        [1/8, 1/6, 1/5,  1/7,  1],
    ],
    # ... (other decision makers' PCMs) ...
]
criteria_names = ["C1", "C2", "C3", "C4", "C5"]

# Initialize and run the Bayesian AHP model
ahp = AHP.StandardAHP(PCM=PCM)
ahp.sampling()
```

See the example notebooks for more advanced usage, visualization, and group decision-making features.

## License

This project is licensed under the MIT License.

---

Built by Majid Mohammadi