.. image:: /docs/img/fedot-industrial.png
    :width: 600px
    :align: center
    :alt: Fedot Industrial logo


================================================================================


.. start-badges
.. list-table::
   :stub-columns: 1

   * - Code
     - | |version| |python|
   * - CI/CD
     - | |coverage| |mirror| |integration|
   * - Docs & Examples
     - |docs| |binder|
   * - Downloads
     - | |downloads|
   * - Support
     - | |support|
   * - Languages
     - | |eng| |rus|
   * - Funding
     - | |itmo| |sai|
.. end-badges

.. |version| image:: https://badge.fury.io/py/fedot-ind.svg
    :target: https://badge.fury.io/py/fedot-ind
    :alt: PyPi version

.. |python| image:: https://img.shields.io/pypi/pyversions/fedot_ind.svg
   :alt: Supported Python Versions
   :target: https://img.shields.io/pypi/pyversions/fedot_ind

.. |build| image:: https://badgen.net/#badge/build/error/red?icon=pypi
   :alt: Build Status

.. |integration| image:: https://github.com/aimclub/Fedot.Industrial/actions/workflows/integration_tests.yml/badge.svg?branch=main
   :alt: Integration Tests Status
   :target: https://github.com/aimclub/Fedot.Industrial/actions/workflows/integration_tests.yml

.. |coverage| image:: https://codecov.io/gh/aimclub/Fedot.Industrial/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/aimclub/Fedot.Industrial/

.. |mirror| image:: https://img.shields.io/badge/mirror-GitLab-orange
   :alt: GitLab mirror for this repository
   :target: https://gitlab.actcognitive.org/itmo-nss-team/Fedot.Industrial

.. |docs| image:: https://readthedocs.org/projects/ebonite/badge/
    :target: https://fedotindustrial.readthedocs.io/en/latest/
    :alt: Documentation Status

.. |binder| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/aimclub/Fedot.Industrial/HEAD

.. |downloads| image:: https://static.pepy.tech/personalized-badge/fedot-ind?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads
    :target: https://pepy.tech/project/fedot-ind
    :alt: Downloads

.. |support| image:: https://img.shields.io/badge/Telegram-Group-blue.svg
    :target: https://t.me/FEDOT_helpdesk
    :alt: Support

.. |rus| image:: https://img.shields.io/badge/lang-ru-yellow.svg
    :target: /README.rst

.. |eng| image:: https://img.shields.io/badge/lang-eng-green.svg
    :target: /README_en.rst

.. |itmo| image:: https://github.com/aimclub/open-source-ops/blob/master/badges/ITMO_badge_flat.svg
   :alt: Acknowledgement to ITMO
   :target: https://en.itmo.ru/en/

.. |sai| image:: https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/SAI_badge_flat.svg
   :alt: Acknowledgement to SAI
   :target: https://sai.itmo.ru/



Fedot.Ind is a automated machine learning framework designed to solve industrial problems related
to time series forecasting, classification, and regression. It is based on
the `AutoML framework FEDOT`_ and utilizes its functionality to build and tune pipelines.


Installation
============

Fedot.Ind is available on PyPI and can be installed via pip:

.. code-block:: bash

    pip install fedot_ind

To install the latest version from the `main branch`_:

.. code-block:: bash

    git clone https://github.com/aimclub/Fedot.Industrial.git
    cd FEDOT.Industrial
    pip install -r requirements.txt
    pytest -s test/

How to Use
==========

Fedot.Ind provides a high-level API that allows you to use its capabilities in a simple way.
The API can be used for classification, regression, and time series forecasting problems, as well as
for anomaly detection.

To use the API, follow these steps:

1. Import ``FedotIndustrial`` class

.. code-block:: python

 from fedot_ind.api.main import FedotIndustrial

2. Initialize the FedotIndustrial object and define the type of modeling task.
It provides a fit/predict interface:

- ``FedotIndustrial.fit()`` begins the feature extraction, optimization and returns the resulting composite pipeline;
- ``FedotIndustrial.predict()`` predicts target values for the given input data using an already fitted pipeline;
- ``FedotIndustrial.get_metrics()`` estimates the quality of predictions using selected metrics.

NumPy arrays or Pandas DataFrames can be used as sources of input data.
In the case below, ``x_train / x_test``, ``y_train / y_test`` are ``pandas.DataFrame()`` and ``numpy.ndarray`` respectively:

.. code-block:: python

    dataset_name = 'Epilepsy'
    industrial = FedotIndustrial(problem='classification',
                                 metric='f1',
                                 timeout=5,
                                 n_jobs=2,
                                 logging_level=20)

    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()

    model = industrial.fit(train_data)

    labels = industrial.predict(test_data)
    probs = industrial.predict_proba(test_data)
    metrics = industrial.get_metrics(target=test_data[1],
                                     rounding_order=3,
                                     metric_names=['f1', 'accuracy', 'precision', 'roc_auc'])

More information about the API is available in the `documentation <https://fedotindustrial.readthedocs.io/en/latest/API/index.html>`__ section.


Documentation and examples
==========================

The comprehensive documentation is available on `readthedocs`_.

Useful tutorials and examples can be found in the `examples`_ folder.


.. list-table::
   :widths: 100 70
   :header-rows: 1

   * - Topic
     - Example
   * - Time series classification
     - `Basic_TSC <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_classification/basic_example.py>`_ and `Advanced_TSC <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_classification/advanced_example.py>`_
   * - Time series regression
     - `Basic_TSR <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_regression/basic_example.py>`_, `Advanced_TSR <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_regression/advanced_regression.py>`_, `Multi-TS <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_regression/multi_ts_example.py>`_
   * - Forecasting
     - `SSA example <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_forecasting/ssa_forecasting.py>`_
   * - Anomaly detection
     - soon will be available
   * - Model ensemble
     - `Notebook <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/notebook_examples/rank_ensemle.ipynb>`_

Benchmarking
============

Univariate time series classification
-------------------------------------

Benchmarking was performed on the collection of 112 out of 144 datasets from the `UCR archive`_.

.. list-table::
   :widths: 100 30 30 30 30

   * - Algorithm
     - Top-1
     - Top-3
     - Top-5
     - Top-Half
   * - **Fedot_Industrial**
     - 17.0
     - 23.0
     - 26.0
     - 38
   * - HC2
     - 16.0
     - 55.0
     - 77.0
     - 88
   * - FreshPRINCE
     - 15.0
     - 22.0
     - 32.0
     - 48
   * - InceptionT
     - 14.0
     - 32.0
     - 54.0
     - 69
   * - Hydra-MR
     - 13.0
     - 48.0
     - 69.0
     - 77
   * - RDST
     - 7.0
     - 21.0
     - 50.0
     - 73
   * - RSTSF
     - 6.0
     - 19.0
     - 35.0
     - 65
   * - WEASEL_D
     - 4.0
     - 20.0
     - 36.0
     - 59
   * - TS-CHIEF
     - 3.0
     - 11.0
     - 21.0
     - 30
   * - HIVE-COTE v1.0
     - 2.0
     - 9.0
     - 18.0
     - 27
   * - PF
     - 2.0
     - 9.0
     - 27.0
     - 40


Multivariate time series classification
---------------------------------------

Benchmarking was performed on the following datasets:
BasicMotions, Cricket, LSST, FingerMovements, HandMovementDirection, NATOPS, PenDigits, RacketSports, Heartbeat, AtrialFibrillation, SelfRegulationSCP2

.. list-table::
   :widths: 100 30

   * - Algorithm
     - Mean Rank
   * - HC2
     - 5.038
   * - ROCKET
     - 6.481
   * - Arsenal
     - 7.615
   * - **Fedot_Industrial**
     - 7.712
   * - DrCIF
     - 7.712
   * - CIF
     - 8.519
   * - MUSE
     - 8.700
   * - HC1
     - 9.212
   * - TDE
     - 9.731
   * - ResNet
     - 10.346
   * - mrseql
     - 10.625


Time series regression
----------------------

Benchmarking was performed on the following datasets:
HouseholdPowerConsumption1, AppliancesEnergy, HouseholdPowerConsumption2, IEEEPPG, FloodModeling1, BeijingPM25Quality, BenzeneConcentration, FloodModeling3, BeijingPM10Quality, FloodModeling2, AustraliaRainfall


.. list-table::
   :widths: 100 30

   * - Algorithm
     - Mean Rank
   * - FreshPRINCE
     - 6.014
   * - DrCIF
     - 6.786
   * - **Fedot_Industrial**
     - 8.114
   * - InceptionT
     - 8.957
   * - RotF
     - 9.414
   * - RIST
     - 9.786
   * - TSF
     - 9.929
   * - RandF
     - 10.286
   * - MultiROCKET
     - 10.557
   * - ResNet
     - 11.171
   * - SingleInception
     - 11.571



Real world cases
================

Building energy consumption
----------------------------

Link to the dataset `on Kaggle <https://www.kaggle.com/competitions/ashrae-energy-prediction>`_

Full notebook with solution `is here <https://github.com/ITMO-NSS-team/Fedot.Industrial/blob/14bdb2f488c1246376fa138f5a2210795fcc16aa/cases/industrial_examples/energy_monitoring/building_energy_consumption.ipynb>`_

The challenge is to develop accurate counterfactual models that estimate energy consumption savings
post-retrofit. Leveraging a dataset comprising three years of hourly meter readings from over a
thousand buildings, the goal is to predict energy consumption (in kWh). Key predictors include **air temperature**,
**dew temperature**, **wind direction**, and **wind speed**.


.. image:: /docs/img/building-target.png
    :align: center
    :alt: building target

.. image:: /docs/img/building_energy.png
    :align: center
    :alt: building results


Results:

.. list-table::
   :widths: 100 60
   :header-rows: 1

   * - Algorithm
     - RMSE_average
   * - `FPCR <https://onlinelibrary.wiley.com/doi/10.1111/insr.12116>`_
     - 455.941
   * - `Grid-SVR <https://proceedings.neurips.cc/paper/1996/file/d38901788c533e8286cb6400b40b386d-Paper.pdf>`_
     - 464.389
   * - `FPCR-Bs <https://www.sciencedirect.com/science/article/abs/pii/S0167947313003629>`_
     - 465.844
   * - `5NN-DTW <https://link.springer.com/article/10.1007/s10618-016-0455-0>`_
     - 469.378
   * - `CNN <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7870510>`_
     - 484.637
   * - **Fedot.Industrial**
     - **486.398**
   * - `RDST <https://arxiv.org/abs/2109.13514>`_
     - 527.927
   * - `RandF <https://link.springer.com/article/10.1023/A:1010933404324>`_
     - 527.343


Permanent magnet synchronous motor (PMSM) rotor temperature
-----------------------------------------------------------
Link to the dataset on `Kaggle <https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature>`_

Full notebook with solution is `here <https://github.com/ITMO-NSS-team/Fedot.Industrial/blob/d3d5a4ddc2f4861622b6329261fc7b87396e0a6d/cases/industrial_examples/equipment_monitoring/motor_temperature.ipynb>`_

This dataset focuses on predicting the maximum recorded rotor temperature of a permanent magnet synchronous
motor (PMSM) during 30-second intervals. The data, sampled at 2 Hz, includes sensor readings such as
**ambient temperature**, **coolant temperatures**, **d and q components** of voltage, and **current**.
These readings are aggregated into 6-dimensional time series of length 60, representing 30 seconds.

The challenge is to develop a predictive model using the provided predictors to accurately estimate the
maximum rotor temperature, crucial for monitoring the motor's performance and ensuring optimal operating conditions.

.. image:: /docs/img/rotor-temp.png
    :align: center
    :alt: rotor temp

.. image:: /docs/img/motor-temperature.png
    :align: center
    :alt: solution


Results:

.. list-table::
   :widths: 100 70
   :header-rows: 1

   * - Algorithm
     - RMSE_average
   * - **Fedot.Industrial**
     - **1.158612**
   * - `FreshPRINCE <https://arxiv.org/abs/2305.01429>`_
     - 1.490442
   * - `RIST <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3486435/>`_
     - 1.501047
   * - `RotF <https://ieeexplore.ieee.org/document/1677518>`_
     - 1.559385
   * - `DrCIF <https://arxiv.org/abs/2305.01429>`_
     - 1.594442
   * - `TSF <https://arxiv.org/abs/1302.2277>`_
     - 1.684828


================================================================================

R&D plans
=========

– Expansion of anomaly detection model list.

– Development of new time series forecasting models.

– Implementation of explainability module (`Issue <https://github.com/aimclub/Fedot.Industrial/issues/93>`_)


Citation
========

Here we will provide a list of citations for the project as soon as the articles
are published.

.. code-block:: bibtex

    @article{REVIN2023110483,
    title = {Automated machine learning approach for time series classification pipelines using evolutionary optimisation},
    journal = {Knowledge-Based Systems},
    pages = {110483},
    year = {2023},
    issn = {0950-7051},
    doi = {https://doi.org/10.1016/j.knosys.2023.110483},
    url = {https://www.sciencedirect.com/science/article/pii/S0950705123002332},
    author = {Ilia Revin and Vadim A. Potemkin and Nikita R. Balabanov and Nikolay O. Nikitin
    }



.. _AutoML framework FEDOT: https://github.com/aimclub/FEDOT
.. _UCR archive: https://www.cs.ucr.edu/~eamonn/time_series_data/
.. _main branch: https://github.com/aimclub/Fedot.Industrial
.. _readthedocs: https://fedotindustrial.readthedocs.io/en/latest/
.. _examples: https://github.com/aimclub/Fedot.Industrial/tree/main/examples
