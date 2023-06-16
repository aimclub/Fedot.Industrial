.. _basic-tsc-example:


Time series classification example
==================================
This example shows how to use the framework to perform time series classification.

Basic TSC experiment
--------------------

First, we import the necessary modules. Base class :ref:`FedotIndustrial<industrial-class-label>` provides all the capabilities
for time series classification.

.. code-block:: python

    from fedot_ind.core.api.main import FedotIndustrial

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

    industrial = FedotIndustrial(task='ts_classification',
                                 dataset='ItalyPowerDemand',
                                 feature_generator='quantile',
                                 use_cache=False,
                                 timeout=5,
                                 n_jobs=-1,
                                 window_sizes='auto')

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

    feature_generator = 'ensemble: topological quantile'

This way the ensemble of feature space of ``topological``, ``quantile`` feature generators will be used as a single feature space.


Ensemble of models predictions
------------------------------

The process of ensemble consists of 3 stages. At the first stage, a dictionary is created that contains the name of the
model as a key and the best metric value for this dataset as a value. The second stage is the creation of a ranked list
in the form of a dictionary (self.sorted_dict), also at this stage parameters such as the best model and the best value
of the quality metric are determined, which are stored in the dictionary self.best_base_results. The third stage is
iterative, in accordance with the assigned rank, adding models to a single composite model and ensemble their predictions.

The framework allows to combine predictions of multiple models into one. To use this feature, import the following class:

.. code-block:: python

    from fedot_ind.core.architecture.postprocessing.results_picker import ResultsPicker
    from fedot_ind.core.ensemble.static.RankEnsembler import RankEnsemble

Then, create an instance of the class :ref:`ResultsPicker<resultspicker-class-label>` and run results collection:

.. code-block:: python

    output_folder = 'path_to_your_results_folder'
    picker = ResultsPicker(path=output_folder)
    proba_dict, metric_dict = picker.run()

One can also use the ``get_metrics_df`` parameter to get a dataframe with metrics for each model.

.. code-block:: python

    metrics_df = picker.run(get_metrics_df=True, add_info=True)

The ``add_info`` parameter allows to add additional information about datasets so the result table would looks more
comprehensive:

+---+------------------------------+-------------+-------+----------+-------------+------------+--------+--------------------+---------------------+
|   | dataset                      | experiment  | f1    | roc\_auc | train\_size | test\_size | length | multivariate\_flag | number\_of\_classes |
+===+==============================+=============+=======+==========+=============+============+========+====================+=====================+
| 0 | ECG5000                      | recurrence  | 0.006 | 0.857    | 500         | 4500       | 140    | 0                  | 5                   |
+---+------------------------------+-------------+-------+----------+-------------+------------+--------+--------------------+---------------------+
| 1 | ECG5000                      | quantile    | 0.007 | 0.939    | 500         | 4500       | 140    | 0                  | 5                   |
+---+------------------------------+-------------+-------+----------+-------------+------------+--------+--------------------+---------------------+
| 2 | ECG5000                      | topological | 0.002 | 0.801    | 500         | 4500       | 140    | 0                  | 5                   |
+---+------------------------------+-------------+-------+----------+-------------+------------+--------+--------------------+---------------------+
| 3 | DistalPhalanxOutlineAgeGroup | recurrence  | 0.686 | 0.832    | 400         | 139        | 80     | 0                  | 3                   |
+---+------------------------------+-------------+-------+----------+-------------+------------+--------+--------------------+---------------------+
| 4 | DistalPhalanxOutlineAgeGroup | quantile    | 0.735 | 0.891    | 400         | 139        | 80     | 0                  | 3                   |
+---+------------------------------+-------------+-------+----------+-------------+------------+--------+--------------------+---------------------+
| 5 | DistalPhalanxOutlineAgeGroup | topological | 0.688 | 0.805    | 400         | 139        | 80     | 0                  | 3                   |
+---+------------------------------+-------------+-------+----------+-------------+------------+--------+--------------------+---------------------+


Then, create an instance of the :ref:`RankEnsemble<rank_ensemble_label>` class and run the ensemble:

.. code-block:: python

    ensembler = RankEnsemble(dataset_name=dataset_name,
                             proba_dict=proba_dict,
                             metric_dict=metric_dict)
    ensembler.ensemble()

The output of the ensemble is a dictionary with the following structure:

.. code-block:: python

    {'Base_model': 'quantile',
     'Base_metric': 0.735,
     'Ensemble_models': ['quantile', 'topological'],
     'Ensemble_method': 'MeanEnsemble',
     'Best_ensemble_metric': 0.748}

The ``Ensemble_models`` field contains the names of the models that were included in the ensemble. The ``Ensemble_method``
field contains the name of the ensemble method. The ``Best_ensemble_metric`` field contains the value of the quality metric
for the ensemble model.

The framework supports the following ensemble methods: ``MeanEnsemble``, ``MedianEnsemble``, ``MinEnsemble``, ``MaxEnsemble``, ``ProductEnsemble``.