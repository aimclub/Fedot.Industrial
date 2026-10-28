# Прикладные примеры выпуска 1.0

Три коротких сценария используют реальные наборы, включённые в установленный
`sktime`. Личные файлы и загрузка данных из сети не нужны. Это проверка применения
API, а не сравнение производительности, поиск лучшей модели или полный эксперимент.

Отдельный [пример эволюционного поиска](evolution/README.md) сравнивает
последовательное, процессное и Dask-исполнение на трёх задачах, с 15 поколениями
и встроенными отчётами Dask. Его измерения не относятся к коротким сценариям ниже.

| Сценарий | Данные и разделение | Реальное выполнение |
| --- | --- | --- |
| `motions_graph` | BasicMotions: официальный TRAIN/TEST, по 40 записей, 6 каналов, 100 отсчётов | Граф `quantile_extractor_torch -> logit`, `fit` и `predict` |
| `airline` | Airline: первые 132 месяца TRAIN, последние 12 TEST | `create_forecasting_runtime`, `lagged_ridge_forecaster`, `fit` и `predict` |
| `motions_neural` | То же официальное разделение BasicMotions | `fedot_ind.neural` CNN-кодировщик и линейный выход: 3 шага SGD, затем классификация TEST |

## Запуск

Из корня репозитория в окружении с зависимостями проекта и совместимым FEDOT:

```bash
python -m examples.release_1_0 --scenario motions_graph
python -m examples.release_1_0 --scenario motions_graph --execute --output-dir ./release_outputs/motions
python -m examples.release_1_0 --scenario airline --backend process --execute --output-dir ./release_outputs/airline
python -m examples.release_1_0 --scenario motions_neural --execute --output-dir ./release_outputs/neural
```

Проверенный профиль FEDOT описан в `.github/actions/fedot-compat/action.yml`.
Это примеры из исходного репозитория, а не файлы отдельного установленного пакета.
Для воспроизведения именно локальной проверки разработчика использовались:

```powershell
$env:PYTHONPATH = "$PWD/.local/worktrees/fedot-evo04-ci-smoke;$PWD"
$python = ".local/evo04-cpu-20261019/Scripts/python.exe"
& $python -m examples.release_1_0 --scenario motions_graph
& $python -m examples.release_1_0 --scenario motions_graph --execute --output-dir examples/release_1_0/_outputs/motions_graph
& $python -m examples.release_1_0 --scenario airline --execute --output-dir examples/release_1_0/_outputs/airline
& $python -m examples.release_1_0 --scenario motions_neural --execute --output-dir examples/release_1_0/_outputs/motions_neural
```

Без `--execute` команда только читает и проверяет JSON, печатает конфигурацию и
не загружает набор, не создаёт модель и не пишет файлы. Параметры находятся рядом
со скриптами в трёх одноимённых JSON. `--config PATH` заменяет весь документ;
неизвестные поля, неверные типы и выход за заданные малые пределы отклоняются.
Документ задаёт сценарий; `--scenario` выбирает только файл по умолчанию.

Для графа и прогноза `--backend process` явно включает `EvaluationPlan` с отдельным
процессом; по умолчанию используется `sequential`. Один кандидат и один исполнитель,
без AutoML и перебора. Лимит кандидата 120 секунд, общий лимит 150 секунд; в
последовательном режиме проверка времени происходит между вызовами, а не прерывает
обучение. Порождённый процесс принадлежит исполнителю и закрывается им.

Для `TensorData` проверен именно граф из таблицы. Штатный узел FEDOT `scaling`
в этом профиле ожидает табличный контейнер, а не Torch Tensor; добавлять его
в граф без соответствующего адаптера нельзя. Здесь нормализация явно выполняется
до построения графа, только по TRAIN. Успех примера не гарантирует работоспособность
произвольного сочетания узлов FEDOT и Industrial.

Нейросетевой пример работает последовательно. `--device cpu` задан по умолчанию;
`--device cuda` допустим только для него и при недоступной CUDA завершается ошибкой
без перехода на CPU. Для предсказания используются `plan_tensor_operation` и
`execute_tensor_operation` из экспериментального пакета профилирования
`benchmark.industrial.experiments.compute_profile.operation_compute`, не из
публичного API Industrial. Оценка памяти относится только к одному вызову
предсказания с объявленным запасом рабочих данных, не ко всему обучению или
свободной памяти устройства. CUDA в проверке примеров на CPU не подтверждена.

## Границы данных и результаты

Среднее и стандартное отклонение вычисляются только на TRAIN: отдельно для каждого
канала BasicMotions и для всей истории TRAIN Airline. Временная ось BasicMotions
сохраняется до статистического извлечения признаков; табличные данные не выдаются
за временные ряды.
Классы кодируются по TRAIN. TEST не передаётся обучению, подбору параметров или
выбору числа шагов. Мини-пакеты нейросети выбираются из TRAIN с фиксированным seed;
глобальные настройки Torch и производственные значения по умолчанию не меняются.
Это демонстрационный цикл, не новый механизм обучения NN-02; контрольные точки
и веса не сохраняются.

`--output-dir` содержит `result.json` с конфигурацией, источником, хешем содержимого,
версиями пакетов, параметрами нормализации и метриками; `predictions.csv` содержит
индекс TEST, истинное значение и предсказание. Для Airline дополнительно вызывается
существующий `render_forecast_comparison_pack`: таблицы, сводка и изображение.
Accuracy/F1 относятся к фиксированному TEST, MAE/RMSE Airline даны в исходных единицах.
Три шага нейросети не обещают полезного качества. Повторный запуск перезаписывает
результаты в выбранной папке. `_outputs/` исключён из Git; исходные данные и веса
в репозиторий не добавляются.

```powershell
& $python -m pytest tests/unit/examples/test_release_1_0.py -q
```
