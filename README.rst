.. image:: /docs/img/fedot-industrial.png
    :width: 600px
    :align: left
    :alt: Fedot Industrial logo

================================================================================

|sai| |itmo|

|issues|  |stars|  |python| |license| |docs| |support| |eng| |mirror|

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

.. |eng| image:: https://img.shields.io/badge/lang-en-red.svg
            :target: /README_en.rst

.. |itmo| image:: https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/ITMO_badge_flat_rus.svg
   :alt: Acknowledgement to ITMO
   :target: https://en.itmo.ru/en/

.. |sai| image:: https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/SAI_badge_flat.svg
   :alt: Acknowledgement to SAI
   :target: https://sai.itmo.ru/

.. |mirror| image:: https://camo.githubusercontent.com/9bd7b8c5b418f1364e72110a83629772729b29e8f3393b6c86bff237a6b784f6/68747470733a2f2f62616467656e2e6e65742f62616467652f6769746c61622f6d6972726f722f6f72616e67653f69636f6e3d6769746c6162
   :alt: GitLab mirror for this repository
   :target: https://gitlab.actcognitive.org/itmo-nss-team/GOLEM


Вместо сложных и ресурсоёмких методов глубокого обучения мы предлагаем использовать методы для
выделения признаков с комплексом небольших моделей, полученных алгоритмическим ядром `AutoML фреймворка FEDOT`_.

Области применения фреймворка:

- **Классификация (для временных рядов или изображений)**

Для этой цели мы предоставляем четыре генератора признаков:

.. image:: /docs/img/all-generators-rus.png
    :width: 700px
    :align: center
    :alt: All generators RUS

После завершения выделения признаков, можно применить эволюционный
алгоритм FEDOT, чтобы найти лучшую модель для заданной задачи классификации.

- **Обнаружение аномалий (для временных рядов или изображений)**

- **Выявление переломных точек (для временных рядов)**

- **Обнаружение объектов на изображениях**


Применение
----------

FEDOT.Industrial предоставляет высокоуровневый API, который позволяет
просто использовать его возможности.

Классификация
_____________

Чтобы провести классификацию временных рядов, необходимо задать конфигурацию эксперимента в виде
словаря, затем создать экземпляр класса `Industrial` и вызвать его метод `run_experiment`:

.. code-block:: python

    from core.api.API import Industrial

    if __name__ == '__main__':
        config = {'feature_generator': ['spectral', 'wavelet'],
                  'datasets_list': ['UMD', 'Lightning7'],
                  'use_cache': True,
                  'error_correction': False,
                  'launches': 3,
                  'timeout': 15}

        ExperimentHelper = Industrial()
        ExperimentHelper.run_experiment(config)


В конфигурации содержатся следующие параметры:

- ``feature_generator`` - список генераторов признаков для использования в эксперименте
- ``use_cache`` - флаг для использования кеширования
- ``datasets_list`` - список наборов данных для использования в эксперименте
- ``launches`` - количество за пусков для каждого набора данных
- ``error_correction`` - флаг для применения модели исправления ошибок в эксперименте
- ``n_ecm_cycles`` - количество циклов для модели исправления ошибок
- ``timeout`` - максимальное количество времени для составления пайплайна для классификации

Наборы данных для классификации должны храниться в каталоге ``data`` и
разделяться на наборы ``train`` и ``test``  с расширением ``.tsv``. Таким образом, имя папки
в каталоге ``data``  должно соответствовать названию набора данных, который будет
использоваться в эксперименте. В случае, если в локальной папке нет данных,
класс ``Data Loader`` попытается загрузить данные из `архива UCR`_.

Генераторы признаков, которые могут быть указаны в конфигурации:
``window_quantile``, ``quantile``, ``spectral_window``, ``spectral``,
``wavelet``, ``recurrence`` и ``topological``.

Также можно объединить несколько генераторов признаков.
Для этого в конфигурации, где задаётся их список,
необходимо присвоить полю ``feature_generator`` следующее значение:

.. code-block:: python

    'ensemble: topological wavelet window_quantile quantile spectral spectral_window'

Результаты эксперимента, которые включают сгенерированные признаки, предсказанные классы, метрики и
пайплайны, хранятся в каталоге ``results_of_experiments/{feature_generator_name}``.
Логи экспериментов хранятся в каталоге ``log``.

Модель исправления ошибок
+++++++++++++++++++++++++

Использование модели исправления ошибок опционально. Чтобы применить её,
необходимо установить значение ``True`` для флага ``error_correction``.
По умолчанию количество циклов равно трём ``n_ecm_cycles=3``, но, используя для настройки экспериментов
конфигурационный файл ``YAML``, можно легко изменить этот параметр.
В этом случае после каждого запуска алгоритмического ядра FEDOT модель исправления ошибок будет обучаться на
полученной ошибке.

.. image:: /docs/img/error_corr_model-rus.png
    :width: 900px
    :align: center
    :alt: Error correction model

Модель для исправления ошибок основана на линейной регрессии и состоит из
трёх этапов: на каждом следующем этапе модель усваивает ошибку
прогнозирования. Этот тип групповой модели для исправления ошибок зависит
от количества классов:

- Для ``бинарной классификации`` модель представляет собой линейную регрессию,
  обученную на предсказаниях этапов коррекции.
- Для ``многоклассовой классификации`` модель представляет собой сумму предыдущих прогнозов.

Кеширование признаков
+++++++++++++++++++++

Чтобы ускорить эксперимент, можно кэшировать признаки, созданные генераторами.
Если у флага ``use_cache`` в конфигурации установлено значение ``True``,
то каждое пространство признаков, сгенерированное во время эксперимента,
кэшируется в соответствующую папку. Для этого вычисляется хэш на основе аргументов
функции ``get_features`` и атрибутов генератора. Затем полученное пространство признаков
записывается на диск с помощью библиотеки ``pickle``.

В следующий раз, когда будет запрашиваеться то же пространство объектов, хэш вычисляется снова и
соответствующее пространство объектов загружается из кэша, что намного быстрее, чем генерировать
его с нуля.

Публикации о FEDOT.Industrial
-----------------------------------

@article{REVIN2023110483,
title = {Automated machine learning approach for time series classification pipelines using evolutionary optimisation},
journal = {Knowledge-Based Systems},
pages = {110483},
year = {2023},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2023.110483},
author = {Ilia Revin and Vadim A. Potemkin and Nikita R. Balabanov and Nikolay O. Nikitin}
}

Структура проекта
-----------------

Последняя стабильная версия FEDOT.Industrial находится в ветке `main`_.

В репозиторий включены следующие каталоги:

- В папке ``core`` содержатся основные классы и скрипты
- В папке ``cases`` содержится несколько примеров использования, которые помогают разобраться, как начать работать с фреймворком
- Все интеграционные и юнит тесты находятся в папке ``test``
- Исходники документации находятся в папке ``docs``

Текущие исследования/разработки и планы на будущее
--------------------------------------------------

– Реализовать кэширование пространства признаков для генераторов признаков (ГОТОВО)

– Разработка модуля для контейнеризации модели

– Разработка хранилища метазнаний для данных, полученных в результате экспериментов

– Исследование кластеризации временных рядов

Документация
------------

Подробная документация доступна в разделе readthedocs_.

Разработка ведётся при поддержке
--------------------------------

Исследование проводится при поддержке Исследовательского центра сильного искусственного интеллекта в
промышленности Университета ИТМО в рамках мероприятия программы центра:
Разработка фреймворка автоматического машинного обучения для промышленных задач.


Цитирование
-----------

Список цитирований для проекта:

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

Также можно цитировать этот репозиторий:

.. code-block:: bibtex

    @online{fedot_industrial,
      author = {Revin, Ilya and Potemkin, Vadim and Balabanov, Nikita and Nikitin, Nikolay},
      title = {FEDOT.Industrial - Framework for automated time series analysis},
      year = 2022,
      url = {https://github.com/ITMO-NSS-team/Fedot.Industrial},
      urldate = {2022-05-05}
    }


.. _AutoML фреймворка FEDOT: https://gitlab.actcognitive.org/itmo-nss-team/FEDOT
.. _архива UCR: https://www.cs.ucr.edu/~eamonn/time_series_data/
.. _main: https://gitlab.actcognitive.org/itmo-nss-team/FEDOT-Industrial
.. _Сильный искусственный интеллект в промышленности: https://sai.itmo.ru/
.. _Университета ИТМО: https://itmo.ru
.. _readthedocs: https://fedotindustrial.readthedocs.io/en/latest/
