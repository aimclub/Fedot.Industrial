Fedot.Ind is a automated machine learning framework designed to solve industrial problems related
to time series forecasting, classification, and regression. It is based on
the `AutoML framework FEDOT`_ and utilizes its functionality to build and tune pipelines.


Installation
============

Fedot.Ind is available on PyPI and can be installed via pip:

.. code-block:: bash

    pip install fedot_ind

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


================================================================================

R&D plans
=========

– Expansion of anomaly detection model list.

– Development of new time series forecasting models.

– Implementation of explainability module (`Issue <https://github.com/aimclub/Fedot.Industrial/issues/93>`_)



.. _AutoML framework FEDOT: https://github.com/aimclub/FEDOT
.. _UCR archive: https://www.cs.ucr.edu/~eamonn/time_series_data/
.. _main branch: https://github.com/aimclub/Fedot.Industrial
.. _readthedocs: https://fedotindustrial.readthedocs.io/en/latest/
.. _examples: https://github.com/aimclub/Fedot.Industrial/tree/main/examples
