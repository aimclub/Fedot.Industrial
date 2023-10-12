.. _basic-tsc-example:


Time series classification example
==================================
This example shows how to use the framework to perform time series classification.

Basic TSC experiment
--------------------

The solution of time series classification problem with the framework is implemented in two modes (strategies):

- **Static**: With the choice of a specific feature generator (quantile (statistical), wavelet-based, recurrent, topological);
- **Dynamic**: Based on the preprocessing pipeline, which implies selection of one or several types of bases (DataDriven, Fourier, Wavelet). In this case, the selected types of bases, being nodes of the pipelines, are subjected to hyperparameter tuning.

The second variant takes more time due to the selection of optimal hyperparameters, but often shows higher metrics,
while the static approach allows to obtain an initial approximation relatively quickly.


First, we import the necessary modules. Base class :ref:`FedotIndustrial<industrial-class-label>` provides all the capabilities
for time series classification.

.. code-block:: python

    from fedot_ind.api.main import FedotIndustrial

It is of a great importance to define corresponding parameters of experiment. The following parameters are required:

- ``task`` - type of the task to be solved (``ts_classification``, but ``ts_regression`` is also supported)
- ``dataset`` - name of the dataset for the experiment (optional argument, used to save the results, but mustn't ne empty of None)
- ``strategy`` - method of solving the problem: specific generator or in fedot_preset mode
- ``use_cache`` - whether to use cached features or not (Optional, default=False)
- ``timeout`` - the maximum amount of time for classification pipeline composition
- ``n_jobs`` - number of jobs to run in parallel when Fedot composes a model (Optional, default=-1)
- ``window_size`` - window size for feature extraction (Optional)
- ``output_folder`` - path to the folder for saving the results (Optional argument)

Then we want to create an instance of the class :ref:`FedotIndustrial<industrial-class-label>`:

.. code-block:: python

    dataset_name = 'Ham'
    industrial = FedotIndustrial(task='ts_classification',
                                 dataset=dataset_name,
                                 strategy='quantile',
                                 use_cache=False,
                                 timeout=5,
                                 n_jobs=-1)

    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()


After that, we can run the experiment:

    .. code-block:: python

        model = industrial.fit(features=train_data)
        labels = industrial.predict(features=test_data)
        probs = industrial.predict_proba(features=test_data)
        metrics = industrial.get_metrics(target=test_data[1], metric_names=['roc_auc'])


For dynamic mode we initialise the ``FedotIndustrial`` class a little differently:

- Choose the ``strategy = 'fedot_preset'``
- Select one or more bases and list them in the ``branch_nodes`` argument. There are the following types of branch nodes: ``eigen_basis``, ``fourier_basis``, and ``wavelet_basis``, One can you either one on them or several at once.
- Optionally, configure the tuning process by specifying the maximum number of iterations - `tuning_iterations` and the time limit - `tuning_timeout`.

So, in this case code for the experiment will look like this:

.. code-block::

    dataset_name = 'Ham'
    industrial = FedotIndustrial(task='ts_classification',
                                 dataset=dataset_name,
                                 strategy='fedot_preset',
                                 branch_nodes=['eigen_basis', 'fourier_basis'],
                                 tuning_iterations=10,
                                 tuning_timeout=10.0,
                                 use_cache=False,
                                 timeout=5,
                                 n_jobs=-1)

    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()
    model = industrial.fit(features=train_data)
    labels = industrial.predict(features=test_data)
    probs = industrial.predict_proba(features=test_data)
    metrics = industrial.get_metrics(target=test_data[1], metric_names=['roc_auc'])
