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

BayesMCDM is a Python library for Bayesian modeling of various multi-criteria decision-making (MCDM) methods. This toolkit enables robust, probabilistic analysis of decision problems by incorporating uncertainty in preferences and criteria weights. The project is under active development—expect more models and functionalities to be added over time.

**You can solve your decision problems directly in your browser—no installation needed.**

## Supported Preference Types

Each method currently supports the standard preference type (e.g., 1-9 scale), which is fully implemented, tested, and ready for use. Additionally, BayesMCDM is being extended to handle a variety of preference formats for each method (these features are under active testing):

- **Interval** preferences
- **Triangular** fuzzy preferences
- **Gaussian** (normal) preferences

Support for **group aggregation** of preferences is also being developed, enabling analysis of collective decisions from multiple decision-makers. Additionally, a **decision-maker clustering** feature is under construction, which will identify groups of decision-makers with homogeneous preferences.

## Supported Methods
The following methods are supported, each with an interactive Google Colab link. These links open ready-to-run notebooks containing example scripts for each method. You can enter your own data and solve your MCDM problem directly in your browser—no installation required.



### 1. Analytic Hierarchy Process (AHP)
AHP is a structured technique for organizing and analyzing complex decisions, based on pairwise comparisons among all criteria.
- [Standard AHP Solver](https://colab.research.google.com/drive/1DLjKhuP29mEiDufejISw8mAMg0MRTIca#scrollTo=c34095ea)  
    *Includes a simple example using standard crisp 1-9 preferences.*

### 2. Best-Worst Method (BWM)
BWM uses the best and worst criteria to derive optimal weights through pairwise comparisons.
- [Standard BWM Solver](https://colab.research.google.com/drive/12X4of4jk5M9mkwQOPehih9XLm7tBkhnL)  
    *Includes a simple example using standard crisp 1-9 preferences.*

### 3. SWING Method
The SWING method elicits weights by asking decision-makers to "swing" criteria from worst to best, reflecting their relative importance.
- [Standard SWING Solver](https://colab.research.google.com/drive/13KPG9WkHnUrYKdAZq5IJAEkqQHD5izHx?usp=drive_open)  
    *Includes a simple example using standard crisp preferences.*

### 4. Point Allocation
Point Allocation allows decision-makers to distribute a fixed number of points among criteria to indicate their importance.
- [Standard Point Allocation Solver](https://colab.research.google.com/drive/1Dh6AB3kCa1pkXdkiKzKDMix6_1GoJBw-?usp=drive_open)  
    *Includes a simple example using standard crisp preferences.*

### 5. Weight Analyzer
The Weight Analyzer provides tools for analyzing the computed weights (and not preferences).
- [Standard Weight Analyzer](https://colab.research.google.com/drive/1148-72AHpfxGhfimLCjs-84prgoQ0wlr#scrollTo=GbxNVoSK4Ft9)  
    *Includes a simple example on how to aggregate weights in a probabilistic manner.*

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
*Figure: Example of weight distributions*


**Weight distributions** plots display the full posterior distributions of criteria weights, allowing users to assess uncertainty, variability, and the impact of preference types on the final weights.

## License

This project is licensed under the MIT License.

---

Built by Majid Mohammadi  