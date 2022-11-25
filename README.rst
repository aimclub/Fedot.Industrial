.. image:: docs/img/fedot-industrial.png
    :width: 600px
    :align: center
    :alt: Fedot Industrial logo

.. container:: center

    .. image:: https://img.shields.io/github/issues/ITMO-NSS-team/Fedot.Industrial?style=for-the-badge
        :target: https://github.com/ITMO-NSS-team/Fedot.Industrial/issues
        :alt: Issues
    .. image:: https://img.shields.io/github/stars/ITMO-NSS-team/Fedot.Industrial?style=for-the-badge
        :target: https://github.com/ITMO-NSS-team/Fedot.Industrial/stargazers
        :alt: Stars
    .. image:: https://img.shields.io/badge/python-3.8-limegreen?style=for-the-badge&logo=python
        :target: https://www.python.org/downloads/release/python-380/
        :alt: Python 3.8
    .. image:: https://img.shields.io/github/license/ITMO-NSS-team/Fedot.Industrial?style=for-the-badge
        :target: https://github.com/ITMO-NSS-team/Fedot.Industrial/blob/main/LICENSE.md
        :alt: License
    .. image:: https://readthedocs.org/projects/ebonite/badge/?style=for-the-badge
        :target: https://fedotindustrial.readthedocs.io/en/latest/
        :alt: Documentation Status

Instead of using complex and resource-demanding deep learning techniques, which could be considered state-of-the-art
solutions, we propose using a combination of feature extractors with an ensemble of lightweight models obtained by the
algorithmic kernel of `AutoML framework FEDOT`_.

Application field of the framework is the following:

- **Classification (time series or image)**

For this purpose we introduce four feature
generators:

.. image:: docs/img/all-generators.png
    :width: 700px
    :align: center
    :alt: All generators

After feature generation process apply evolutionary
algorithm of FEDOT to find the best model for classification task.

- **Anomaly detection (time series or image)**

*--work in progress--*

- **Change point detection (only time series)**

*--work in progress--*

- **Object detection (only image)**

*--work in progress--*

Usage
-----

FEDOT.Industrial provides a high-level API that allows you
to use its capabilities in a simple way.

Classification
______________

To conduct time series classification you need to first
set experiment configuration using file ``cases/config/Config_Classification.yaml``
and then run ``cases/classification_experiment.py`` script, or create your own
with the following code:

.. code-block:: python

    from core.api.tsc_API import Industrial

    if __name__ == '__main__':
        config_name = 'Config_Classification.yaml'
        ExperimentHelper = Industrial()
        ExperimentHelper.run_experiment(config_name)


Config file contains the following parameters:

- ``feature_generators`` - list of feature generators to use in the experiment
- ``datasets_list`` - list of datasets to use in the experiment
- ``launches`` - number of launches for each dataset
- ``feature_generator_params`` - specification for feature generators
- ``fedot_params`` - specification for FEDOT algorithmic kernel
- ``error_correction`` - flag for application of error correction model in the experiment
- ``n_ecm_cycles`` - number of cycles for error correction model

Datasets for classification should be stored in the ``data`` directory and
divided into ``train`` and ``test`` sets with ``.tsv`` extension. So the name of folder
in the ``data`` directory should be equal to the name of dataset that you want
to use in the experiment. In case of data absence in the local folder, implemented ``DataLoader``
class will try to load data from the `UCR archive`_.

Possible feature generators which could be specified in configuration are
``window_quantile``, ``quantile``, ``spectral_window``, ``spectral``,
``wavelet`` and ``topological``.

There is also a possibility to ensemble several feature generators.
It could be done by the following instruction in
``feature_generator`` field of ``Config_Classification.yaml`` file where
you need to specify the list of feature generators:

.. code-block:: yaml

    'ensemble: topological wavelet window_quantile quantile spectral spectral_window'

Results of experiment which include generated features, predicted classes, metrics and
pipelines are stored in ``results_of_experiments/{feature_generator name}`` directory.
Logs of experiment are stored in ``log`` directory.

Error correction model
++++++++++++++++++++++

It is up to you to decide whether to use error correction model or not. To apply it the ``error_correction``
flag in the ``Config_Classification.yaml`` file should be set to ``True`` and number of
cycles ``n_ecm_cycles`` should be provided.
In this case after each launch of FEDOT algorithmic kernel the error correction model will be trained on the
produced error.

.. image:: docs/img/error_corr_model.png
    :width: 900px
    :align: center
    :alt: Error correction model

The error correction model is a linear regression model of
three stages: at every next stage the model learn the error of
prediction. The type of ensemble model for error correction is dependent
on the number of classes:
- For ``binary classification`` the ensemble is also
linear regression, trained on predictions of correction stages.
- For ``multiclass classification`` the ensemble is a sum of previous predictions.

Feature caching
+++++++++++++++

To speed up the experiment, you can cache the features generated by the feature generators.
If ``use_cache`` bool flag in ``Config.yaml`` is ``True``, then every feature space generated during experiment is
cached into corresponding folder. To do so a hash from function ``get_features`` arguments and generator attributes
is obtained. Then resulting feature space is dumped via ``pickle`` library.

The next time when the same feature space is requested, the hash is calculated again and the corresponding
feature space is loaded from the cache which is much faster than generating it from scratch.

Anomaly detection
_________________

*--work in progress--*

Change point detection
______________________

*--work in progress--*

Object detection
________________

*--work in progress--*

Examples & Tutorials
--------------------

Comprehensive tutorial will be available soon.

Publications about FEDOT.Industrial
-----------------------------------

Our plan for publication activity is to publish papers related to
framework's usability and its applications. Here is a list of articles which are
under review process:

.. [1] AUTOMATED MACHINE LEARNING APPROACH FOR TIME SERIES
       CLASSIFICATION PIPELINES USING EVOLUTIONARY OPTIMISATION` by Ilya E. Revin,
       Vadim A. Potemkin, Nikita R. Balabanov, Nikolay O. Nikitin

.. [2] AUTOMATED ROCKBURST FORECASTING USING COMPOSITE MODELLING FOR SEISMIC SENSORS DATA
       by Ilya E. Revin, Vadim A. Potemkin, and Nikolay O. Nikitin

Stay tuned!

Project structure
-----------------

The latest stable release of FEDOT.Industrial is on the `main
branch`_.

The repository includes the following directories:

- Package ``core`` contains the main classes and scripts
- Package ``cases`` includes several how-to-use-cases where you can start to discover how framework works
- All unit and integration tests will be observed in the ``test`` directory
- The sources of the documentation are in the ``docs``

Current R&D and future plans
----------------------------

- |check_| Implement feature space caching for feature generators

- |uncheck_| Development of meta-knowledge storage for data obtained from the experiments

- |uncheck_| Research on time series clusterization

Documentation
-------------

Comprehensive documentation is available at readthedocs_.

Supported by
------------

The study is supported by Research Center
`Strong Artificial Intelligence in Industry`_
of `ITMO University`_ (Saint Petersburg, Russia)

Citation
--------

Here will be provided a list of citations for the project as soon as articles
will be published.

So far you can use citation for this repository:

.. code-block:: bibtex
    @online{fedot_industrial,
      author = {Revin, Ilya and Potemkin, Vadim and Balabanov, Nikita and Nikitin, Nikolay},
      title = {FEDOT.Industrial - Framework for automated time series analysis},
      year = 2022,
      url = {https://github.com/ITMO-NSS-team/Fedot.Industrial},
      urldate = {2022-05-05}
    }


.. _AutoML framework FEDOT: https://github.com/nccr-itmo/FEDOT
.. _UCR archive: https://www.cs.ucr.edu/~eamonn/time_series_data/
.. _main branch: https://github.com/ITMO-NSS-team/Fedot.Industrial
.. _Strong Artificial Intelligence in Industry: https://sai.itmo.ru/
.. _ITMO University: https://itmo.ru
.. _readthedocs: https://fedotindustrial.readthedocs.io/en/latest/


.. |check_| raw:: html

    <input checked=""  disabled="" type="checkbox">

.. |uncheck_| raw:: html

    <input disabled="" type="checkbox">