# Готовность benchmark-инфраструктуры для сравнительного набора по римановым признакам

## Что уже готово

- В проекте уже есть полноценный benchmark-слой для временных рядов: [benchmark/industrial](benchmark/industrial).
- Для задач классификации доступны:
  - загрузка данных через `in_memory_tsc` и `ucr/uea`-адаптеры;
  - конфигурация набора датасетов и моделей через `DatasetSpec`, `ModelSpec`, `BenchmarkSuiteConfig`;
  - сохранение результатов и публикационной упаковки в выходную директорию.
- Для генерации признаков уже есть инфраструктура генераторов признаков в [fedot_ind/core/kernel_learning/generators/adapters.py](fedot_ind/core/kernel_learning/generators/adapters.py). Именно она теперь используется в новом benchmark-адаптере `sklearn_classifier`.

## Что добавлено для текущего этапа

- Новый benchmark-адаптер `sklearn_classifier` в [benchmark/industrial/models/classification.py](benchmark/industrial/models/classification.py).
  - Он позволяет запускать обычные sklearn-классификаторы поверх существующих генераторов признаков.
  - В том числе можно использовать `riemann_extractor` как источник признаков.
- Пример запуска: [benchmark/industrial/examples/riemann_feature_suite.py](benchmark/industrial/examples/riemann_feature_suite.py).
- Минимальный тест на интеграцию: [tests/unit/benchmark/industrial/models/test_classification_adapters.py](tests/unit/benchmark/industrial/models/test_classification_adapters.py).

## Как пользоваться

1. Запустите пример:
   ```bash
   .venv/Scripts/python.exe benchmark/industrial/examples/riemann_feature_suite.py
   ```
2. Изучите результаты в каталоге `benchmark/results/industrial_demo/riemann_feature_suite`.
3. Для более серьёзного сравнения замените `in_memory_tsc` на набор реальных UCR-датасетов и добавьте несколько моделей/генераторов признаков.

## Что пока ещё не покрывает текущий benchmark-слой

- Измерение времени генерации признаков и памяти как отдельного этапа.
- Сравнение по нескольким seed/разбиениям для оценки устойчивости.
- Автоматическая выборка датасетов по длине/размерности/количеству классов.
- Отдельные отчёты по "переходам на fallback" и по стоимости вычислений.

## Следующий логический шаг

Для полноценного набора "сравнительный набор для римановых признаков" лучше сначала:

1. зафиксировать список UCR-датасетов по категориям короткие/длинные, 1D/MD, small/medium;
2. собрать конфигурацию из нескольких генераторов признаков (`statistical_summary`, `topological_extractor`, `riemann_extractor` с разными параметрами);
3. добавить модели `logistic_regression`, `svc`, `random_forest`, `knn` через `sklearn_classifier`;
4. расширить метрики и отчёты времени/памяти.
