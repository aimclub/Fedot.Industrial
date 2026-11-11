# P0: измерение генераторов и координатора

Измерения включаются явно через `PythonProfile` или `profile_call` из
`benchmark.industrial.experiments.evolution_parallelism.profiling`.
Профилировщик не заменяет методы генераторов, не меняет их параметры и RNG.
Он измеряет текущий поток; работники Dask и фоновый монитор требуют отдельных
областей измерения. Вложенный или внешний активный профилировщик не подменяется.

```python
from benchmark.industrial.experiments.evolution_parallelism.profiling import profile_call

result = profile_call(coordinator_step, output_dir, graph, label='mutation_step')
```

Для статистик можно использовать сохранённые массивы TRAIN без обучения:

```powershell
python -m benchmark.industrial.experiments.evolution_parallelism.profile_statistics
python -m benchmark.industrial.experiments.evolution_parallelism.profile_statistics --execute --inputs examples/artifacts/cloud_bundle/staged_training_20261029/input_snapshots/regression.npz --config examples/artifacts/cloud_bundle/staged_training_20261029/full_01/regression_42_staged/config.json --output-dir examples/artifacts/cloud_bundle/performance_p0/statistics
```

Без `--execute` нет вычислений и записи. Параметры берутся из сохранённой
конфигурации. Каждому опыту нужна новая папка. Сохраняются `calls.pstats`,
`calls.csv`, метаданные профиля и числовой выход генератора. Массивы TEST
не загружаются. Доступны только итоговый TRAIN и первое внутреннее разбиение.

`exclusive_seconds` учитывает собственное время функции, `cumulative_seconds`
включает вложенные вызовы и не складывается по строкам. Время CPU потока
отделено от CPU процесса, в котором может работать фоновый сборщик.
Профилирование само создаёт нагрузку: такие запуски не служат чистым замером
ускорения. Полное время отдельно включает подготовку, оценку, сохранение и
освобождение ресурсов; остаток координатора не следует приписывать одному Dask.
