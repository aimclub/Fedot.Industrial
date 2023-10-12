Quickstart
==========

First of all, you need to :ref:`install <installation>` FEDOT.Industrial.

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

In case of time series classification task you also can use our `DataLoader` to download data
from `UCR/UEA archive <https://www.timeseriesclassification.com>`_:

.. code-block:: python

    from fedot_ind.tools.loader import DataLoader

    loader = DataLoader(dataset_name='ECG200')
    train_data, test_data = loader.load_data()


You can also specify the folder to search for data if your own data is in `.ts`, `.tsv` or `arff` format:

.. code-block:: python

    loader = DataLoader(dataset_name='YourDatasetName', folder_path='path/to/folder')
    train_data, test_data = loader.load_data()

