.. image:: /docs/img/fedot-industrial.png
    :width: 600px
    :align: left
    :alt: Fedot Industrial logo

================================================================================

|sai| |itmo|

|issues|  |stars|  |python| |license| |docs| |support| |rus|

.. |issues| image:: https://img.shields.io/github/issues/ITMO-NSS-team/Fedot.Industrial?style=flat-square
            :target: https://github.com/ITMO-NSS-team/Fedot.Industrial/issues
            :alt: Issues


.. |stars| image:: https://img.shields.io/github/stars/ITMO-NSS-team/Fedot.Industrial?style=flat-square
            :target: https://github.com/ITMO-NSS-team/Fedot.Industrial/stargazers
            :alt: Stars

.. |python| image:: https://img.shields.io/badge/python-3.8-44cc12?style=flat-square&logo=python
            :target: https://www.python.org/downloads/release/python-380/
            :alt: Python 3.8

.. |license| image:: https://img.shields.io/github/license/ITMO-NSS-team/Fedot.Industrial?style=flat-square
            :target: https://github.com/ITMO-NSS-team/Fedot.Industrial/blob/main/LICENSE.md
            :alt: License

.. |docs| image:: https://readthedocs.org/projects/ebonite/badge/?style=flat-square
            :target: https://fedotindustrial.readthedocs.io/en/latest/
            :alt: Documentation Status

.. |support| image:: https://img.shields.io/badge/Telegram-Group-blue.svg
            :target: https://t.me/fedotindustrial_support
            :alt: Support

.. |rus| image:: https://img.shields.io/badge/lang-ru-yellow.svg
            :target: /README.rst

.. |itmo| image:: https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/ITMO_badge_flat_rus.svg
   :alt: Acknowledgement to ITMO
   :target: https://en.itmo.ru/en/

.. |sai| image:: https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/SAI_badge_flat.svg
   :alt: Acknowledgement to SAI
   :target: https://sai.itmo.ru/

.. |mirror| image:: https://camo.githubusercontent.com/9bd7b8c5b418f1364e72110a83629772729b29e8f3393b6c86bff237a6b784f6/68747470733a2f2f62616467656e2e6e65742f62616467652f6769746c61622f6d6972726f722f6f72616e67653f69636f6e3d6769746c6162
   :alt: GitLab mirror for this repository
   :target: https://gitlab.actcognitive.org/itmo-nss-team/GOLEM


Instead of using complex and resource-demanding deep learning techniques, which could be considered state-of-the-art
solutions, we propose using a combination of feature extractors with an ensemble of lightweight models obtained by the
algorithmic kernel of the `AutoML framework FEDOT`_.

The application fields of the framework are the following:

- **Classification (time series or image)**

For this purpose we introduce four feature
generators:

.. image:: /docs/img/all-generators.png
    :width: 700px
    :align: center
    :alt: All generators

Once the feature generation process is complete, you can apply FEDOT's evolutionary
algorithm to find the best model for the classification task.

- **Anomaly detection (time series or image)**

- **Change point detection (only time series)**

- **Object detection (only image)**


Usage
-----

FEDOT.Industrial provides a high-level API that allows you
to use its capabilities in a simple way.

Classification
______________

To conduct time series classification you need to set the experiment configuration via a dictionary,
then create an instance of the ``Industrial`` class, and call its ``run_experiment`` method:

.. code-block:: python

    from core.api.main import FedotIndustrial

    industrial = FedotIndustrial(task='ts_classification',
                                 dataset=dataset_name,
                                 strategy='statistical',
                                 use_cache=True,
                                 timeout=1,
                                 n_jobs=2,
                                 window_sizes='auto',
                                 logging_level=20,
                                 output_folder=None)

You can then load the data and run the experiment:

.. code-block:: python

    train_data, test_data, _ = industrial.reader.read(dataset_name='ItalyPowerDemand')

    model = industrial.fit(train_features=train_data[0], train_target=train_data[1])
    labels = industrial.predict(test_features=test_data[0])
    metric = industrial.get_metrics(target=test_data[1], metric_names=['f1', 'roc_auc'])


The config contains the following parameters:

- ``task`` - type of task to be solved (``ts_classification``)
- ``dataset`` - name of the data set for the experiment
- ``strategy`` - the way to solve the problem: a specific generator or in ``fedot_preset`` mode
- ``use_cache`` - a flag to use caching of extracted features
- ``timeout`` - maximum amount of time to compile a pipeline for the classification
- ``n_jobs`` - number of processes for parallel execution
- ``window_sizes`` - window sizes for window generators
- ``logging_level`` - logging level
- ``output_folder`` - path to folder to save results


Datasets for classification should be stored in the ``data`` directory and
divided into ``train`` and ``test`` sets with ``.tsv`` extension. So the folder name
in the ``data`` directory should be set to the name of the dataset that you want
to use in the experiment. In case there is no data in the local folder, the ``DataLoader``
class will try to load data from the `UCR archive`_.

Possible feature generators which could be specified in the configuration are
``quantile``, ``wavelet``, ``recurrence`` и ``topological``.

It is also possible to ensemble several feature generators.
It could be done by setting the ``strategy`` field of the config, where
you need to specify the list of feature generators, to the following value:

.. code-block:: python

    'ensemble: topological wavelet quantile'


Feature caching
+++++++++++++++

To speed up the experiment, you can cache the features produced by the feature generators.
If ``use_cache`` bool flag in config is ``True``, then every feature space generated during the experiment is
cached into the corresponding folder.

The next time when the same feature space is requested, the hash is calculated again and the corresponding
feature space is loaded from the cache which is much faster than generating it from scratch.


Stay tuned!

Project structure
-----------------

The latest stable release of FEDOT.Industrial is in the `main
branch`_.

The repository includes the following directories:

- The ``api`` folder contains the main interface classes and scripts
- Package ``core`` contains the main classes and scripts
- Package ``examples`` includes several how-to-use-cases where you can start to discover how the framework works
- All unit and integration tests are in the ``test`` directory
- The sources of the documentation are in ``docs``

Current R&D and future plans
----------------------------

– Implement feature space caching for feature generators (DONE)

– Development of model containerization module

– Development of meta-knowledge storage for data obtained from the experiments

– Research on time series clusterization

Documentation
-------------

A comprehensive documentation is available at readthedocs_.

Supported by
------------

The study is supported by the Research Center Strong Artificial Intelligence in Industry of ITMO University
as part of the plan of the center's program: Development of AutoML framework for industrial tasks.

Citation
--------

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