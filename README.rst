.. image:: /docs/img/fedot-industrial.png
    :width: 600px
    :align: left
    :alt: Fedot Industrial logo

================================================================================

|sai| |itmo|

|issues|  |stars| |coverage| |python| |license| |docs| |support| |eng| |mirror| |downloads|

.. |issues| image:: https://img.shields.io/github/issues/ITMO-NSS-team/Fedot.Industrial?style=flat-square
            :target: https://github.com/ITMO-NSS-team/Fedot.Industrial/issues
            :alt: Issues

.. |downloads| image:: https://static.pepy.tech/personalized-badge/fedot-ind?period=total&units=international_system&left_color=black&right_color=green&left_text=Downloads
               :target: https://pepy.tech/project/fedot-ind
               :alt: Downloads

.. |stars| image:: https://img.shields.io/github/stars/ITMO-NSS-team/Fedot.Industrial?style=flat-square
            :target: https://github.com/ITMO-NSS-team/Fedot.Industrial/stargazers
            :alt: Stars

.. |python| image:: https://img.shields.io/badge/python-3.8-44cc12?style=flat-square&logo=python
            :target: https://www.python.org/downloads/release/python-380/
            :alt: Python 3.8

.. |coverage| image:: https://codecov.io/gh/aimclub/Fedot.Industrial/branch/main/graph/badge.svg
              :target: https://codecov.io/gh/aimclub/Fedot.Industrial

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

.. |itmo| image:: https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/ITMO_badge_flat.svg
   :alt: Acknowledgement to ITMO
   :target: https://en.itmo.ru/en/

.. |sai| image:: https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/SAI_badge_flat.svg
   :alt: Acknowledgement to SAI
   :target: https://sai.itmo.ru/

.. |mirror| image:: https://camo.githubusercontent.com/9bd7b8c5b418f1364e72110a83629772729b29e8f3393b6c86bff237a6b784f6/68747470733a2f2f62616467656e2e6e65742f62616467652f6769746c61622f6d6972726f722f6f72616e67653f69636f6e3d6769746c6162
   :alt: GitLab mirror for this repository
   :target: https://gitlab.actcognitive.org/itmo-nss-team/Fedot-Industrial


Вместо сложных и ресурсоёмких методов глубокого обучения мы предлагаем использовать методы для
выделения признаков с комплексом небольших моделей, полученных алгоритмическим ядром `AutoML фреймворка FEDOT`_.

Области применения фреймворка:

- **Классификация (для временных рядов или изображений)**

Для этой цели мы предоставляем четыре генератора признаков:

.. image:: /docs/img/all-generators.png
    :width: 700px
    :align: center
    :alt: All generators RUS

После завершения выделения признаков, можно применить эволюционный
алгоритм FEDOT, чтобы найти лучшую модель для заданной задачи классификации.

- **Обнаружение аномалий (для временных рядов или изображений)**

.. note::

        В настоящее время мы работаем над этой функцией.

- **Компьютерное зрение**

В данном разделе в настоящий момент решаются два типа задач: классификация изображений и детектирование объектов.



Применение
----------

FEDOT.Industrial предоставляет высокоуровневый API, который позволяет
просто использовать его возможности.

Классификация
_____________

Чтобы выполнить эксперимент по классификации временных рядов, необходимо инициализировать экземпляр класса ``FedotIndustrial``,
и передать ему ряд именованных аргументов:

.. code-block:: python

    from core.api.main import FedotIndustrial

    industrial = FedotIndustrial(task='ts_classification',
                                 dataset='ItalyPowerDemand,
                                 strategy='statistical',
                                 use_cache=True,
                                 timeout=15,
                                 n_jobs=4,
                                 window_sizes='auto',
                                 logging_level=20,
                                 output_folder=None)

Затем можно загрузить данные и запустить эксперимент:

.. code-block:: python

    train_data, test_data, _ = industrial.reader.read(dataset_name='ItalyPowerDemand')

    model = industrial.fit(features=train_data[0], target=train_data[1])
    labels = industrial.predict(features=test_data[0])
    metric = industrial.get_metrics(target=test_data[1], metric_names=['f1', 'roc_auc'])

В конфигурации могут содержаться следующие параметры:

- ``task`` – тип решаемой задачи (``ts_classification``)
- ``dataset`` – имя набора данных для эксперимента
- ``strategy`` – способ решения задачи: конкретный генератор или в режиме ``fedot_preset``
- Для режима ``fedot_preset`` дополнительно можно указать два параметра для пайплайна препроцессинга:
    - ``branch_nodes`` - список узлов, которые будут участвовать в тюнинге гиперпараметров
    - ``tuning_iterations`` - количество итераций тюнинга гиперпараметров
- ``use_cache`` - флаг для использования кеширования извлечённых признаков
- ``timeout`` - максимальное количество времени для составления пайплайна для классификации
- ``n_jobs`` - количество процессов для параллельного выполнения
- ``window_sizes`` - размеры окон для оконных генераторов
- ``logging_level`` - уровень логирования
- ``output_folder`` - путь к папке для сохранения результатов

Наборы данных для классификации должны храниться в каталоге ``data`` и
разделяться на наборы ``train`` и ``test``  с расширением ``.tsv``. Таким образом, имя папки
в каталоге ``data``  должно соответствовать названию набора данных, который будет
использоваться в эксперименте. В случае, если в локальной папке нет данных,
класс ``Data Loader`` попытается загрузить данные из `архива UCR`_.

Генераторы признаков, которые могут быть указаны в конфигурации:
``quantile``, ``wavelet``, ``recurrence`` и ``topological``.

Также можно объединить несколько генераторов признаков.
Для этого в конфигурации, где задаётся их список,
необходимо присвоить полю ``strategy`` следующее значение:

.. code-block:: python

    'ensemble: topological wavelet quantile'

Кеширование признаков
+++++++++++++++++++++

Чтобы ускорить эксперимент, можно кэшировать признаки, созданные генераторами.
Если у флага ``use_cache`` в конфигурации установлено значение ``True``,
то каждое пространство признаков, сгенерированное во время эксперимента,
кэшируется в соответствующую папку. Для этого вычисляется хэш на основе аргументов
функции извлечения признаков и атрибутов генератора. Затем полученное пространство признаков
записывается на диск с помощью библиотеки ``pickle``.

В следующий раз, когда будет запрашиваеться то же пространство объектов, хэш вычисляется снова и
соответствующее пространство объектов загружается из кэша, что намного быстрее, чем генерировать
его с нуля.

Структура проекта
-----------------

Последняя стабильная версия FEDOT.Industrial находится в ветке `main`_.

В репозиторий включены следующие каталоги:

- В папке ``api`` содержатся основные классы и скрипты интерфейса
- В папке ``core`` содержатся основные алгоритмы и модели
- В папке ``examples`` содержится несколько примеров использования, которые помогают разобраться, как начать работать с фреймворком
- Все интеграционные и юнит-тесты находятся в папке ``test``
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

.. _AutoML фреймворка FEDOT: https://gitlab.actcognitive.org/aimclub/FEDOT
.. _архива UCR: https://www.cs.ucr.edu/~eamonn/time_series_data/
.. _main: https://gitlab.actcognitive.org/aimclub/FEDOT-Industrial
.. _readthedocs: https://fedotindustrial.readthedocs.io/en/latest/