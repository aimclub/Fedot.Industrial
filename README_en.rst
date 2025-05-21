Kernel-Based framework for time-series classification, regression, and forecasting


================================================================================



This repository is a copy of newly developed framework with actual results on classification, regression and forecasting experiments.

It is organized as follows:

- `benchmark/results/time_series_multi_reg_comparasion.csv` directory contains results on multivariate regression problem
- `benchmark/results/results_ucr.csv` directory contains results on univariate and multivariate classification problem
- `benchmark/results/m4_gluon_nbeats` directory contains experiment results on M4 benchmark 

- notebook `benchmark/results/m4_gluon_nbeats/m4_analysis.ipynb` is about analysis of forecasting experiments
- notebook 'benchmark/average_metrics.ipynb' is for analysis of metrics obtained during classification and regression experiments


- module `benchmark/benchmark_TSC.py` is for classification experiments
- module `benchmark/benchmark_TSER.py` is for regression experiments
- module `benchmark/benchmark_TSF.py` is for forecasting experiments