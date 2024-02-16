.. image:: /docs/img/fedot-industrial.png
    :width: 600px
    :align: left
    :alt: Fedot Industrial logo

================================================================================


.. start-badges
.. list-table::
   :stub-columns: 1

   * - Код
     - | |version| |python|
   * - CI/CD
     - |  |coverage| |mirror| |integration|
   * - Документация и примеры
     - |docs| |binder|
   * - Статистика загрузок
     - | |downloads|
   * - Подержка
     - | |support|
   * - Язфка
     - | |eng| |rus|
   * - Аккредитация
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



Fedot.Ind - это автоматизированный фреймворк машинного обучения,
разработанный для решения промышленных задач, связанных с прогнозированием
временных рядов, классификацией и регрессией.
Он основан на `AutoML фреймворке FEDOT`_ и использует его функциональность
для композирования и тюнинга пайплайнов.

Установка
============

Fedot.Ind доступен на PyPI и может быть установлен с помощью pip:

.. code-block:: bash

    pip install fedot_ind

Для установки последней версии из `main branch`_:

.. code-block:: bash

    git clone https://github.com/aimclub/Fedot.Industrial.git
    cd FEDOT.Industrial
    pip install -r requirements.txt
    pytest -s test/

Как пользоваться
================

Fedot.Ind предоставляет высокоуровневый API, который позволяет использовать
его возможности в простом и удобном виде. Этот API может быть использован
для решения задач классификации, регрессии, прогнозирования временных рядов,
а также для обнаружения аномалий.

Для использования API необходимо:

1. Импортировать класс ``FedotIndustrial``

.. code-block:: python

 from fedot_ind.api.main import FedotIndustrial

2. Инициализировать объект FedotIndustrial и определить тип задачи.
Данный объект предоставляет интерфейс для методов fit/predict.:

- ``FedotIndustrial.fit()`` – запуск извлечения признаков, оптимизации; возвращает получившийся композитный пайплайн;
- ``FedotIndustrial.predict()`` прогнозирует значения целевой переменной для заданных входных данных, используя полученный ранее пайплайн;
- Метод ``FedotIndustrial.get_metrics()`` оценивает качество прогнозов с использованием выбранных метрик.

В качестве источников входных данных можно использовать массивы NumPy или
объекты DataFrame из библиотеки Pandas. В данном случае, ``x_train / x_test`` и ``y_train / y_test`` – ``pandas.DataFrame()`` и ``numpy.ndarray`` соответственно:

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

Больше информации об использовании API доступно в `соответствующей секции <https://fedotindustrial.readthedocs.io/en/latest/API/index.html>`__ документации.


Документация и примеры
==========================

Наиболее оплная документация собрана в `readthedocs`_.

Полезные материалы и примеры использования находятся в папке `examples`_ репозитория.


.. list-table::
   :widths: 100 70
   :header-rows: 1

   * - Тема
     - Пример
   * - Классификация временных рядов
     - `Basic <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_classification/basic_example.py>`_ and `Advanced <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_classification/advanced_example.py>`_
   * - Регрессия
     - `Basic <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_regression/basic_example.py>`_, `Advanced <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_regression/advanced_regression.py>`_, `Multi-TS <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_regression/multi_ts_example.py>`_
   * - Прогнозирование
     - `SSA example <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_forecasting/ssa_forecasting.py>`_
   * - Детектирование аномалий
     - скоро будет в доступе
   * - Ансамблирование
     - `Notebook <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/notebook_examples/rank_ensemle.ipynb>`_


Бенчмаркинг
============

Классификация одномерных временных рядов
-----------------------------------------

Бенчмаркинг проводился на выборке из 112/144 датасетов из архива `UCR`..

.. list-table::
   :widths: 100 30 30 30 30

   * - Алгоритм
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


Классификация многомерных временных рядов
------------------------------------------

Бенчмаркинг проводился на следубщей выборке датасетов:
BasicMotions, Cricket, LSST, FingerMovements, HandMovementDirection, NATOPS, PenDigits, RacketSports, Heartbeat, AtrialFibrillation, SelfRegulationSCP2

.. list-table::
   :widths: 100 30

   * - Алгоритм
     - Средний ранг
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


Регрессия временных рядов
--------------------------

Бенчмаркинг проводился на следующих датасетах:
HouseholdPowerConsumption1, AppliancesEnergy, HouseholdPowerConsumption2, IEEEPPG, FloodModeling1, BeijingPM25Quality, BenzeneConcentration, FloodModeling3, BeijingPM10Quality, FloodModeling2, AustraliaRainfall


.. list-table::
   :widths: 100 30

   * - Алгоритм
     - Средний ранг
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




Применение на реальных данных
==============================

Энергопотребление здания
----------------------------

Ссылка на данные `Kaggle <https://www.kaggle.com/competitions/ashrae-energy-prediction>`_

Ноутбук с решением `here <https://github.com/ITMO-NSS-team/Fedot.Industrial/blob/14bdb2f488c1246376fa138f5a2210795fcc16aa/cases/industrial_examples/energy_monitoring/building_energy_consumption.ipynb>`_

Задача состоит в разработке точных контрфактических моделей, позволяющих оценить экономию энергопотребления
после модернизации. Используя набор данных, состоящий из трехлетних почасовых показаний счетчиков более чем
тысячи зданий, ставится задача прогнозирования энергопотребления (в кВт-ч). Ключевыми предикторами
являются **температура воздуха**, **температура росы**, **направление ветра** и **скорость ветра**.


.. image:: /docs/img/building-target.png
    :align: center
    :alt: building target

.. image:: /docs/img/building_energy.png
    :align: center
    :alt: building results


Результаты сравнения с SOTA-алгоритмами:

.. list-table::
   :widths: 100 60
   :header-rows: 1

   * - Алгоритм
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


Температура ротора синхронного двигателя с постоянными магнитами (СДПМ)
-----------------------------------------------------------------------
Ссылка на данные `Kaggle <https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature>`_

Ноутбук с решением `here <https://github.com/ITMO-NSS-team/Fedot.Industrial/blob/d3d5a4ddc2f4861622b6329261fc7b87396e0a6d/cases/industrial_examples/equipment_monitoring/motor_temperature.ipynb>`_

Данный набор данных предназначен для прогнозирования максимальной зарегистрированной температуры
ротора синхронного двигателя с постоянными магнитами (СДПМ) в течение 30-секундных интервалов.
Данные, дискретизированные с частотой 2 Гц, включают показания датчиков, такие как
**температура окружающей среды**, **температура охлаждающей жидкости**, **d и q компоненты** напряжения
и **тока**.

Эти показания агрегируются в 6-мерный временной ряд длиной 60, что соответствует 30 секундам.

Задача заключается в разработке прогнозирующей модели с использованием предоставленных предикторов для
точной оценки максимальной температуры ротора, что крайне важно для мониторинга работы двигателя и
обеспечения оптимальных условий эксплуатации.


.. image:: /docs/img/rotor-temp.png
    :align: center
    :alt: rotor temp

.. image:: /docs/img/motor-temperature.png
    :align: center
    :alt: solution


Результаты сравнения с SOTA-алгоритмами:

.. list-table::
   :widths: 100 70
   :header-rows: 1

   * - Алгоритм
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



Дальнейшие R&D планы
=====================

– Расширение списка моделей обнаружения аномалий.

– Разработка новых моделей прогнозирования временных рядов.

– Внедрение модуля объяснимости (Задача <https://github.com/aimclub/Fedot.Industrial/issues/93>_)


Цитирование
===========

Здесь мы предоставим список цитирования проекта, как только статьи будут опубликованы.

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
