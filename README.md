# What-If

**What-If** is a Python library for **counterfactual probability estimation** and **uplift modeling**.

It implements several estimators of counterfactual probabilities, together with wrappers around uplift models and additional code for benchmarking and experimentation. The repository accompanies recent work on the identification and estimation of counterfactual probabilities from uplift scores and bivariate probabilistic models.

## Overview

In many decision-making settings, we observe only one outcome per individual: the outcome under the action that was actually taken. Counterfactual analysis asks what would have happened under a different action.

This repository focuses on that problem through two complementary lines of work:

- **partial counterfactual identification**, using uplift modeling to derive bounds on counterfactual probabilities;
- **probabilistic counterfactual estimation**, using bivariate distributions to model the joint law of potential outcomes.

The package is designed for research use, reproducible experiments, and methodological comparisons.

## Implemented papers

This repository contains code related to the following papers.

### 1. Identifying counterfactual probabilities using bivariate distributions and uplift modeling

- Preprint: https://arxiv.org/abs/2512.08805

```bibtex
@misc{verhelst2025identifying,
  title={Identifying counterfactual probabilities using bivariate distributions and uplift modeling},
  author={Verhelst, Th{\'e}o and Bontempi, Gianluca},
  year={2025},
  eprint={2512.08805},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

This work studies how to estimate counterfactual probabilities by combining uplift scores with structured bivariate probabilistic models.

### 2. Partial counterfactual identification and uplift modeling: theoretical results and real-world assessment

- Paper: https://link.springer.com/article/10.1007/s10994-023-06317-w

```bibtex
@article{verhelst2023partial,
  title={Partial counterfactual identification and uplift modeling: theoretical results and real-world assessment},
  author={Verhelst, Th{\'e}o and Mercier, Denis and Shrestha, Jeevan and Bontempi, Gianluca},
  journal={Machine Learning},
  pages={1--25},
  year={2023},
  publisher={Springer},
  doi={10.1007/s10994-023-06317-w}
}
```

This paper develops theoretical results on partial counterfactual identification and evaluates them on real-world data.

## Repository structure

- `whatif/` — core package code
- `examples/` — Jupyter notebooks showing how to use the library
- `docs/` — Sphinx configuration and documentation sources

## Citation

If you use this repository in academic work, please cite the relevant paper(s) above.
