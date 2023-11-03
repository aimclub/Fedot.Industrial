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

.. |mirror| image:: https://camo.githubusercontent.com/9bd7b8c5b418f1364e72110a83629772729b29e8f3393b6c86bff237a6b784f6/68747470733a2f2f62616467656e2e6e65742f62616467652f6769746c61622f6d6972726f722f6f72616e67653f69636f6e3d6769746c6162
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
    :target: https://t.me/fedotindustrial_support
    :alt: Support

.. |rus| image:: https://img.shields.io/badge/lang-ru-yellow.svg
    :target: /README.rst

.. |eng| image:: https://img.shields.io/badge/lang-eng-green.svg
    :target: /README_en.rst

.. |itmo| image:: https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/ITMO_badge_flat_rus.svg
   :alt: Acknowledgement to ITMO
   :target: https://en.itmo.ru/en/

.. |sai| image:: https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/SAI_badge_flat.svg
   :alt: Acknowledgement to SAI
   :target: https://sai.itmo.ru/



Fedot.Ind is a automated machine learning framework designed to solve industrial problems related
to time series forecasting, classification, regression, and anomaly detection. It is based on
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
In the case below, ``x_train``, ``y_train`` and ``x_test`` are ``numpy.ndarray()``:

.. code-block:: python

    model = Fedot(task='ts_classification', timeout=5, strategy='quantile', n_jobs=-1, window_mode=True, window_size=20)
    model.fit(features=x_train, target=y_train)
    prediction = model.predict(features=x_test)
    metrics = model.get_metrics(target=y_test)

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
     - `Basic <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_classification/basic_example.py>`_ and `Advanced <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_classification/advanced_example.py>`_
   * - Time series regression
     - `Basic <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_regression/basic_example.py>`_, `Advanced <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_regression/advanced_regression.py>`_, `Multi-TS <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_regression/multi_ts_example.py>`_
   * - Forecasting
     - `SSA example <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_forecasting/ssa_forecasting.py>`_
   * - Anomaly detection
     - soon will be available
   * - Computer vision
     - `Classification <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/api_example/computer_vision/image_classification/image_clf_example.py>`_, `Object detection <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/api_example/computer_vision/object_detection/obj_rec_example.py>`_
   * - Model ensemble
     - `Notebook <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/notebook_examples/rank_ensemle.ipynb>`_

Real world cases
================

Building energy consumption
----------------------------

.. list-table::
   :widths: 100 60
   :header-rows: 1

   * - Link to dataset
     - Solution
   * - `Kaggle <https://www.kaggle.com/code/fatmanuranl/ashrae-energy-prediction2>`_
     - `Notebook <https://github.com/ITMO-NSS-team/Fedot.Industrial/blob/14bdb2f488c1246376fa138f5a2210795fcc16aa/cases/industrial_examples/energy_monitoring/building_energy_consumption.ipynb>`_

Dimensions correspond to the air temperature, dew temperature, wind direction and wind speed:

.. image:: /docs/img/building_energy.svg
    :align: center
    :alt: madrid results

The goal is to estimate the **energy consumption in kWh**

Results:

.. list-table::
   :widths: 100 60
   :header-rows: 1

   * - Algorithm
     - RMSE_average
   * - FPCR_RMSE
     - 455.941
   * - Grid-SVR_RMSE
     - 464.389
   * - FPCR-Bs_RMSE
     - 465.844
   * - 5NN-DTW_RMSE
     - 469.378
   * - CNN_RMSE
     - 484.637
   * - **Fedot_Industrial**
     - **486.398**
   * - RDST_RMSE
     - 527.927
   * - RandF_RMSE
     - 527.343

Permanent magnet synchronous motor (PMSM) rotor temperature
-----------------------------------------------------------
Link to the dataset on `Kaggle <https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature>`_
Full notebook with solution is `here <https://github.com/ITMO-NSS-team/Fedot.Industrial/blob/d3d5a4ddc2f4861622b6329261fc7b87396e0a6d/cases/industrial_examples/equipment_monitoring/motor_temperature.ipynb>`_

Sample features:

.. image:: /docs/img/motor_temp.svg
    :align: center
    :alt: motor results

Results:

.. list-table::
   :widths: 100 70
   :header-rows: 1

   * - Algorithm
     - RMSE_average
   * - **Fedot_Industrial_AutoML**
     - **1.158612**
   * - FreshPRINCE_RMSE
     - 1.490442
   * - RIST_RMSE
     - 1.501047
   * - Fedot_Industrial_baseline
     - 1.538009
   * - RotF_RMSE
     - 1.559385
   * - DrCIF_RMSE
     - 1.594442
   * - TSF_RMSE
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