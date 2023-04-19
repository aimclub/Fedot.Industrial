.. _basic-tsc-example:


Time series classification example
==================================
This example shows how to use the framework to perform time series classification.

Basic TSC experiment
--------------------

First, we import the necessary modules. Base class :ref:`Industrial<industrial-class-label>` provides all the capabilities
for time series classification.

.. code-block:: python

    from core.api.main import FedotIndustrial

Then, the config dict with experiment parameters must be defined.

.. code-block:: python

    config = dict(task='ts_classification',
                  dataset='ItalyPowerDemand',
                  feature_generator='quantile',
                  use_cache=False,
                  timeout=5,
                  n_jobs=-1,
                  window_sizes='auto')

It is of a great importance to define corresponding parameters of experiment. The following parameters are required:

- ``task`` - type of task to solve. In this case, it is obviously time series classification
- ``dataset`` - name of dataset to use in the experiment
- ``feature_generator`` - feature extractor to use in the experiment
- ``use_cache`` - whether to use cached features or not
- ``timeout`` - the maximum amount of time for classification pipeline composition
- ``n_jobs`` - number of jobs to run in parallel when Fedot composes a model
- ``window_sizes`` - window sizes for feature extraction. Mode ``auto`` defines that the range of window sizes will be selected automatically.

Finally, we create an instance of the class :ref:`FedotIndustrial<industrial-class-label>` and run the experiment.

.. code-block:: python

    industrial = FedotIndustrial(input_config=config,
                                 output_folder=None)

To accelerate repetitive experiments, the feature caching mechanism is implemented. It allows to dump generated features
to the disk and load them later. To enable this feature, set ``use_cache`` parameter to ``True`` in the config.

Results of experiment will be available in the ``results_of_experiment`` folder. For each feature generator there will be a
folder containing sub-folders for every dataset.

Analysis of obtained results can be done manually or using the :ref:`Results Picker<resultspicker-class-label>` class:

.. code-block:: python

    from core.architecture.postprocessing.results_picker import ResultsPicker

    collector = ResultsPicker(path='to_your_results_folder', launch_type='max')
    metrics_df = parser.run(get_metrics_df=True)


where ``metrics_df`` is a dataframe of the following structure:

+------------+------------+-----------+-----------+-----------+
| dataset    | f1         | roc_auc   | generator | n_classes |
+============+============+===========+===========+===========+
| Beef       | 0.878      | 0.654     | quantile  |     5     |
+------------+------------+-----------+-----------+-----------+
| Beef       | 0.989      | 0.898     | ensemble  |    5      |
+------------+------------+-----------+-----------+-----------+
| Earthquakes| 0.765      | 0.781     | spectral  |    2      |
+------------+------------+-----------+-----------+-----------+
| Lightning7 | 0.501      | 0.409     | wavelet   |    7      |
+------------+------------+-----------+-----------+-----------+


Feature ensemble experiment
---------------------------

The feature ensemble experiment is a special case of the basic experiment. It allows to combine
multiple feature spaces obtained with corresponding generators into one.
To enable feature generators ensemble, set the following option in the config:

.. code-block:: python

    feature_generator = 'ensemble: topological quantile wavelet'

This way the ensemble of feature space of ``topological``, ``wavelet``, ``quantile`` feature generators will be used as a single feature space.


.. note::
    See also :ref:`Advanced TSC approach<tsc_advanced>` section for more details on
    time series classification experiment and :ref:`Model Ensemble<tsc-ensembling>` section for information
    on model ensemble approach.