# industrial-forecasting

Контракты и модели прогнозирования временных рядов. Базовая установка для
Python 3.10 и 3.11 требует только NumPy. Для тензорных моделей предусмотрена
группа зависимостей `neural`; FEDOT не требуется ни в одном варианте.

## Установка и пример

Команды из корня репозитория в новом виртуальном окружении:

```bash
python -m pip install ./packages/industrial-forecasting
python -I examples/forecasting_package.py --standalone
```

Пример выводит прогнозы и MAE для двух базовых моделей. Последние четыре
точки синтетического ряда используются только для оценки.
Вместо исходников можно установить wheel из вложения
`industrial-forecasting-distributions`. Команда сборки:

```bash
python -m build packages/industrial-forecasting --outdir dist/forecasting
```

Для проверки тензорных моделей установите `industrial-neural` и
`industrial-forecasting[neural]` той же версии. Для установки на CPU:

```bash
python -m pip install --upgrade pip
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0"
python -m pip install ./packages/industrial-neural "./packages/industrial-forecasting[neural]"
python -m pip check
python -I examples/forecasting_package.py --standalone --neural
```

## Публичные точки входа

- `ForecastScenario` хранит историю и известные к моменту прогноза признаки.
  Сетка задаётся через `TimeGrid` целыми позициями или явными числовыми
  либо календарными отметками NumPy. История и прогноз не пересекаются.
- `PointForecast` и `QuantileForecast` сохраняют отметки каждого шага и
  порядок рядов. `score_forecast` оценивает один точечный прогноз;
  `pinball_loss` возвращает потери отдельно для каждого уровня квантиля.
- `baselines.predict_baseline` строит отдельный прогноз для каждого ряда:
  последний уровень, сезонная повторяемость, среднее, дрейф, скользящее
  среднее или линейный тренд. Эти модели не обучаются совместно и не
  используют дополнительные признаки.
- `neural.models.forecast_series` использует лаговый или низкоранговый
  Ridge, SSA/MSSA, HAVOK или гибридный ансамбль. Вычислительное устройство
  задаётся в параметрах модели. `neural.tcn.TCNModule` предоставляет
  тензорную сеть TCN без обёртки для подготовки данных FEDOT.
- `regimes` описывает свойства ряда и предлагает варианты моделей.
  `tuning` строит поэтапный план и принимает внешнюю функцию оценки;
  `tuning.series` использует временные разбиения и переданную фабрику моделей.

Адаптеры FEDOT 1.0 и исполнение бенчмарков остаются в Industrial. Полный
перечень перемещённых реализаций и сохранённых адаптеров приведён в
[описании переноса](../../docs/dev_guide/forecasting_package.md).
