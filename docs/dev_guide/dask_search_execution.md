# Dask: один сеанс на поиск

Жизненный цикл задаётся явно в конфигурации исполнения:

```python
from fedot_ind.core.optimizer.configuration import EvolutionConfig
from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend, EvaluationExecutionConfig
from fedot_ind.core.optimizer.evaluation.dask_config import DaskExecutionConfig, DaskLifecycle

config = EvolutionConfig(evaluation_execution=EvaluationExecutionConfig(
    backend=EvaluationBackend.DASK,
    candidate_timeout_seconds=120,
    total_timeout_seconds=600,
    dask=DaskExecutionConfig(lifecycle=DaskLifecycle.SEARCH),
))
```

Передайте `config` в `IndustrialEvoOptimizer(..., optimisation_params=config)`.
В JSON эквивалентная настройка: `evaluation_execution.dask.lifecycle = "search"`.
Прежнее значение по умолчанию — `"batch"`: отдельный кластер на пакет.

`optimise` владеет областью поиска. Кластер создаётся только перед первым
непустым пакетом и закрывается при выходе, включая исключение. Подготовка
кластера расходует бюджет оценки и общий срок поиска, а не получает добавочное
время. Окончательное обучение после поиска не входит в эту область.

Если вызывающий код явно передал уже открытый `DaskEvaluationSession` через
`dask_session_scope`, он имеет приоритет и закрывается только своим владельцем.
Случайный глобальный `distributed.Client` не используется. Отдельный вызов
диспетчера с режимом `search` требует явной области `dispatcher.search_scope()`.

Это управление ресурсами, не переиспользование оценок. Конфигурации кандидатов,
их параметры, начальные значения генераторов случайных чисел и правила отбора
не меняются. Последовательный режим не получает принудительное прерывание `fit`.
