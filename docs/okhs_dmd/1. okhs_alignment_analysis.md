# OKHS-DMD Alignment Analysis And Reference Refactoring Design

## Обзор

- Наиболее сильное совпадение теории и реализации находится в математическом
  ядре [okhs.py](D:\data_old\WORK\Repo\Industiral\IndustrialTS\fedot_ind\core\operation\decomposition\matrix_decomposition\method_impl\okhs.py):
- Есть Gram-представление по траекториям, квадратуры Якоби, оператор Лиувилля и прогноз через Mittag-Leffler.
- Основные инженерные разрывы сосредоточены не в самой идее, а в слое example-скрипт содержит утечку train/test,
  numerical policy зашита в методы, а plotting смешан с inference.
- Верхнеуровневый замысел
  из [постановка_по_okhs_dmd.md](D:\data_old\WORK\Repo\Industiral\IndustrialTS\docs\okhs_dmd\постановка_по_okhs_dmd.md)
  заметно шире текущего кода:
    - AutoML-оркестрация,
    - benchmarking и интегрированный выбор `q` .

## Alignment Scorecard

- implemented: 1
- partial: 1
- heuristic: 1
- missing: 1

## Alignment Matrix

| Theme                                                  | Status      | What matches                                                                                                                                                                                                             | Missing                                                                                                                | Evidence                                                                                                                                                                                                                                                                                                                                                      |
|--------------------------------------------------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Representer theorem / RKHS / OKHS                      | implemented | Пример и core реально строят Gram-матрицу по траекториям и используют occupation-like интегральное представление. Это самый сильный участок соответствия постановке.                                                     | -                                                                                                                      | `fedot_ind/core/operation/decomposition/matrix_decomposition/method_impl/okhs.py:11-215` - строится Gram-матрица по траекториям через двойной интеграл и квадратуры Якоби.<br>`examples/rkhs_okhs/temp_file/okhs_advanced.py:140-164` - пример действительно использует `OKHSTransformer` как первую стадию пайплайна.                                        |
| Fractional Liouville + Mittag-Leffler + fractional DMD | partial     | В core реализованы квадратуры Якоби, матрица Лиувилля, generalized eigensolver и прогноз через Mittag-Leffler. Это уже не просто эвристика, а рабочая конечномерная аппроксимация заявленной математической конструкции. | typed validation for numerical policies<br>separate inference API without plotting side effects                        | `fedot_ind/core/operation/decomposition/matrix_decomposition/method_impl/okhs.py:218-348` - `FractionalLiouvilleOperator` реализует матрицу Лиувилля и обобщённую спектральную задачу.<br>`fedot_ind/core/operation/decomposition/matrix_decomposition/method_impl/okhs.py:351-725` - `FractionalDMD` реализует модальный прогноз с функцией Миттаг-Леффлера. |
| OKHSForecaster strategies                              | heuristic   | Документ заявляет recursive, multi-output и dmd-режимы, но в коде стратегии частично размазаны между numpy и torch обёртками. Семантика существует, однако vocabulary и API не унифицированы.                            | single typed strategy enum<br>consistent recursive/multi-output semantics                                              | `fedot_ind/core/models/kernel/okhs_forecasting.py:8-87` - numpy-обёртка использует строки `dmd` и `direct`.<br>`fedot_ind/core/models/kernel/okhs_forecasting_torch.py:15-133` - torch-обёртка использует `dmd` и `occupation`, что уже расходится с numpy-версией.                                                                                           |
| AutoML / RKBS / uncertainty / q-selection              | missing     | В постановке это существенная часть продукта, но в текущем коде нет единого AutoML-слоя для OKHS-DMD. Есть только отдельные элементы и эвристики без общей orchestration logic.                                          | unified automl entrypoint<br>uncertainty estimates<br>integrated q-selection policy<br>benchmarking/reporting workflow | `fedot_ind/core/operation/transformation/representation/kernel/kernels.py:255-360` - есть `DataDrivenQSelector`, но он не встроен в единый forecasting/API слой.<br>`docs/okhs_dmd/постановка_по_okhs_dmd.md:89-148` - AutoML, rolling forecast, uncertainty и benchmarking заявлены как целевой контур.                                                      |

## Что бьется с теорией

### 1. OKHS as a trajectory-level space

В [okhs.py](D:\data_old\WORK\Repo\Industiral\IndustrialTS\fedot_ind\core\operation\decomposition\matrix_decomposition\method_impl\okhs.py)
траектории действительно входят как объекты сравнения в интегральной Gram-конструкции,
а не редуцируются к отдельным точкам. Это хорошо согласуется с разделом 1.3 постановки -
код ближе к функциональным траекториям модели, чем к обычной оконной регрессии

### 2. Реализован дробный оператор Лиувилля

`FractionalLiouvilleOperator` и `FractionalDMD` реализуют основную математическую гипотезу -  
дробный оператор аппроксимируется в конечномерном пространстве, далее строится спектральное разложение
и эволюция через функцию Миттаг-Леффлера. Именно этот участок кода делает утверждение
из разделов 2.1-2.3 содержательным.

### 3. Есть полный e2e пример

[okhs_advanced.py](D:\data_old\WORK\Repo\Industiral\IndustrialTS\examples\rkhs_okhs\temp_file\okhs_advanced.py)
проходит полный маршрут
`OKHSTransformer -> FractionalLiouvilleOperator -> FractionalDMD`.

## Что реализовано частично

### 1. Стратегии прогнозирования пока только на бумаге

Документ описывает recursive, multi-output и DMD-подходов. Реально:

- [okhs_forecasting.py](D:\data_old\WORK\Repo\Industiral\IndustrialTS\fedot_ind\core\models\kernel\okhs_forecasting.py)
  знает `direct` и `dmd`;
- [okhs_forecasting_torch.py](D:\data_old\WORK\Repo\Industiral\IndustrialTS\fedot_ind\core\models\kernel\okhs_forecasting_torch.py)
  знает `occupation` и `dmd`;
- тип стратегии выражен строками, а не enum/ADT.

### 2. q-selection exists as a detached heuristic

В [kernels.py](D:\data_old\WORK\Repo\Industiral\IndustrialTS\fedot_ind\core\operation\transformation\representation\kernel\kernels.py)
есть `DataDrivenQSelector`, а в постановке выбор `q` — важная часть модели памяти.
Сейчас это отдельная эвристика, не интегрированная в единый forecasting workflow.

## Что есть в документе но чего нет в коде.

- Единый `UnifiedOKHSAutoML` / `OKHSEnhancedAutoML` слой.
- Полноценные доверительные интервалы.
- Rolling forecast с явной схемой переобучения.
- Системный benchmarking и сравнения с бейзлайнами.
- Реальная связка forecasting / classification / regression через одну OKHS/RKBS-платформу.
- Интерпретируемая и формализованная логика оркестрации для выбора `q`.

## Основные минусы

### Train/test leakage in example

- Category: `data_leakage`
- Severity: `high`
- Summary: в example-скрипте в качестве тестовой выбирается траектория, которая уже попала в обучающий набор.
-
Evidence: [okhs_advanced.py](D:\data_old\WORK\Repo\Industiral\IndustrialTS\examples\rkhs_okhs\temp_file\okhs_advanced.py#L317)
- Impact: метрика `Forecast MSE` оказывается оптимистичной и не отражает честное обобщение.
- Recommendation: сделать явный `TrajectorySplit` и держать test-only trajectory вне `train_trajectories`.

### Important invariant is encoded only in comments

- Category: `api_design`
- Severity: `high`
- Summary: требование `initial_segment_length >= n_train_traj` фигурирует как комментарий к конфигу, а не как typed
  validation.
-
Evidence: [okhs_advanced.py](D:\data_old\WORK\Repo\Industiral\IndustrialTS\examples\rkhs_okhs\temp_file\okhs_advanced.py#L237), [okhs.py](D:\data_old\WORK\Repo\Industiral\IndustrialTS\fedot_ind\core\operation\decomposition\matrix_decomposition\method_impl\okhs.py#L451)
- Impact: пользователь получает позднее исключение из линейной системы вместо ранней диагностики конфигурации.
- Recommendation: валидировать инвариант до запуска fit/predict и выражать его через typed config/result.

### Inference and plotting are mixed

- Category: `side_effects`
- Severity: `medium`
- Summary: `plot_predict` внутри модели одновременно считает прогноз, готовит диагностику, печатает в stdout и рисует
  графики.
-
Evidence: [okhs.py](D:\data_old\WORK\Repo\Industiral\IndustrialTS\fedot_ind\core\operation\decomposition\matrix_decomposition\method_impl\okhs.py#L566)
- Impact: модель хуже тестируется, сложнее переиспользуется в non-interactive pipeline и труднее поддаётся композиции.
- Recommendation: оставить в модели только `predict`, а plotting вынести в effect shell.

### Numerical policy is hidden

- Category: `numerical_stability`
- Severity: `medium`
- Summary: регуляризация и fallback solver присутствуют, но не оформлены как явные policy objects.
-
Evidence: [okhs.py](D:\data_old\WORK\Repo\Industiral\IndustrialTS\fedot_ind\core\operation\decomposition\matrix_decomposition\method_impl\okhs.py#L184), [okhs.py](D:\data_old\WORK\Repo\Industiral\IndustrialTS\fedot_ind\core\operation\decomposition\matrix_decomposition\method_impl\okhs.py#L323)
- Impact: сложно объяснить, воспроизвести и контролировать численное поведение.
- Recommendation: ввести `RegularizationPolicy` и `StabilityPolicy`.

### No dedicated OKHS/fDMD tests

- Category: `testability`
- Severity: `medium`
- Summary: в `tests/` нет выделенного набора unit/integration тестов именно для OKHS/fDMD ядра.
- Impact: математические регрессии и API drift почти не зафиксированы.
- Recommendation: добавить pure-core tests на invariants и thin integration tests на честный holdout scenario.

### Required package is not declared explicitly

- Category: `dependency_hygiene`
- Severity: `medium`
- Summary: `pymittagleffler` импортируется и в core, и в example, но не был найден в явных зависимостях.
-
Evidence: [okhs.py](D:\data_old\WORK\Repo\Industiral\IndustrialTS\fedot_ind\core\operation\decomposition\matrix_decomposition\method_impl\okhs.py#L7), [okhs_advanced.py](D:\data_old\WORK\Repo\Industiral\IndustrialTS\examples\rkhs_okhs\temp_file\okhs_advanced.py#L10)
- Impact: окружение может собраться неполно и упасть уже на runtime import.
- Recommendation: зафиксировать зависимость в `requirements.txt` и `pyproject.toml`.

## Потенциальные направления для развития

### 1. "Тонкий" адаптер для экспериментов

Превратить [okhs_advanced.py](D:\data_old\WORK\Repo\Industiral\IndustrialTS\examples\rkhs_okhs\temp_file\okhs_advanced.py)
в тонкий orchestration слой:

- pure generation/split/evaluation helpers;
- отдельные функции plotting;
- честный holdout;
- ранняя конфигурационная валидация.

### 2. Типизированное вычислительное ядро

Развивать [okhs.py](D:\data_old\WORK\Repo\Industiral\IndustrialTS\fedot_ind\core\operation\decomposition\matrix_decomposition\method_impl\okhs.py)
не как монолитный scientific script, а как:

- pure numerical core;
- explicit policies for regularization/stability;
- effect shell for plotting/logging;
- typed result paths вместо скрытых fallbacks.

### 3. Унифицированное API для forecasting

Свести numpy/torch обёртки к одному словарю:

- `OKHSMethod` enum вместо `direct/dmd/occupation` строк;
- единый contract для recursive/multi-output semantics;
- q-selection как явная policy, а не фоновая эвристика.

### 4. Product-level layer for the document promises

Следующий продуктовый слой должен включать:

- честный benchmark runner,
- uncertainty/interval estimation,
- rolling forecast evaluation,
- orchestration для выбора `q`,
- унификацию forecasting/regression/classification сценариев.

## План доработки

### Стадия 1. Стабилизировать пример эксперимента

Goal: сделать пример честным, воспроизводимым и безопасным для интерпретации.

- `split-example-runner`: превратить `okhs_advanced.py` в thin runner.
  Paths: `examples/rkhs_okhs/temp_file/okhs_advanced.py`
  Changes:
- вынести generation/split/evaluation/plotting в отдельные функции;
- ввести явный holdout split;
- валидировать конфиг до запуска fit.
  Tests: no-overlap split; early failure при недостаточном `initial_segment_length`.

### Стадия 2. Работа над вычислительным ядром

Цель: отделить "чистую" вычислительную логику от сайд эффектов и вынести численные решения в конфиг.

- `core-policy-split`: ввести `RegularizationPolicy` и `StabilityPolicy`.
  Paths: `fedot_ind/core/operation/decomposition/matrix_decomposition/method_impl/okhs.py`
  Changes: убрать скрытый jitter, сделать явный solver fallback, вынести plotting из модели.
  Tests: симметрия Gram-матрицы, shape-инварианты Liouville/DMD, чистый `predict` path.

### 3 Этап. Унифицирова forecasting APIs

Цель: выровнять numpy и torch wrappers вокруг одной типизированной модели стратегий.

- `unify-forecaster-methods`: нормализовать naming и подключить выбор `q` как отдельнулью "политику".
  Paths: `fedot_ind/core/models/kernel/okhs_forecasting.py`, `fedot_ind/core/models/kernel/okhs_forecasting_torch.py`, `fedot_ind/core/operation/transformation/representation/kernel/kernels.py`
  Changes: `OKHSMethod` enum, backward-compatible string aliases, явный q-policy input.
  Tests: совместимость старых entrypoint-ов и единая семантика enum-based config.

## Какой эффект от рефакторинга ожидается

- Typed domain model делает стратегию, split и numerical policies частью API, а не “знанием в комментариях”.
- Pure analysis/planning core можно тестировать без `matplotlib`, runtime solver-а и file IO.
- Effect boundaries становятся узкими: чтение артефактов, запись отчётов и визуализация не загрязняют вычислительное
  ядро.
- Такой подход позволяет делать инкрементальный рефакторинг без needless rewrite: отдельно example runner, отдельно
  numerical core, отдельно wrappers.

