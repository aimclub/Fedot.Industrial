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
     - |  |coverage| |mirror|
   * - Документация
     - |docs|
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

.. |integration| image:: https://github.com/aimclub/Fedot.Industrial/workflows/Integration/badge.svg?branch=main
   :alt: Integration Tests Status
   :target: https://github.com/aimclub/Fedot.Industrial/actions/workflows/integration-build.yml

.. |coverage| image:: https://codecov.io/gh/aimclub/Fedot.Industrial/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/aimclub/Fedot.Industrial/

.. |mirror| image:: https://camo.githubusercontent.com/9bd7b8c5b418f1364e72110a83629772729b29e8f3393b6c86bff237a6b784f6/68747470733a2f2f62616467656e2e6e65742f62616467652f6769746c61622f6d6972726f722f6f72616e67653f69636f6e3d6769746c6162
   :alt: GitLab mirror for this repository
   :target: https://gitlab.actcognitive.org/itmo-nss-team/Fedot.Industrial

.. |docs| image:: https://readthedocs.org/projects/ebonite/badge/
    :target: https://fedotindustrial.readthedocs.io/en/latest/
    :alt: Documentation Status

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
временных рядов, классификацией, регрессией и обнаружением аномалий.
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
объекты DataFrame из библиотеки Pandas. В данном случае, `x_train`,
`y_train` и `x_test` представлены в виде объектов `numpy.ndarray()`:

.. code-block:: python

    model = Fedot(task='ts_classification', timeout=5, strategy='quantile', n_jobs=-1, window_mode=True, window_size=20)
    model.fit(features=x_train, target=y_train)
    prediction = model.predict(features=x_test)
    metrics = model.get_metrics(target=y_test)

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
     - `Базовый <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_classification/basic_example.py>`_ и `Расширенный <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_classification/advanced_example.py>`_
   * - Регрессия временных рядов
     - `Пример <google.com>`_
   * - Прогнозирование
     - `SSA example <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/pipeline_example/time_series/ts_forecasting/ssa_forecasting.py>`_
   * - Детектирование аномалий
     - скоро будет в доступе
   * - Компьютерное зрение
     - `Классификация <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/api_example/computer_vision/image_classification/image_classification_example.ipynb>`_, `Детектирование объектов <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/api_example/computer_vision/object_detection/object_detection_example.ipynb>`_
   * - Ансамблирование моделей
     - `Ноутбук <https://github.com/aimclub/Fedot.Industrial/blob/main/examples/api_example/ensembling/rank_ensemle.ipynb>`_


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