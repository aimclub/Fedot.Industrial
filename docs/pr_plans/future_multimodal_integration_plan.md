# FUTURE multimodal integration plan

Дата анализа: 2026-06-15.

Источник материалов:

- `D:\data_old\WORK\obsidian\Papers_2025\Проекты для разработки\Исследовательские проекты\Industrial\Трек по мультимодальности\Описание НИР.pdf`
- `D:\data_old\WORK\obsidian\Papers_2025\Проекты для разработки\Исследовательские проекты\Industrial\Трек по мультимодальности\FUTURE-feature-multimodal_experiments.zip`

## Краткое резюме

FUTURE, Fusion-based Unified Time-series Representation Encoder, проверяет идею общего представления временных рядов через несколько модальностей: сырой ряд, статистические признаки, GAF-изображения, STFT-спектрограммы и, в коде трансформаций, MTF. Основной эксперимент был поставлен на многоклассовой классификации UCR/UEA: 6 датасетов, 38 конфигураций, 3 seed, всего 684 запуска. Основная метрика: `macro_f1`.

Сильная сторона метода в том, что это не только набор архитектур, но и уже проверенная экспериментальная постановка: есть единая подготовка модальностей, train/val/test нормализация без явного использования test statistics, реестр энкодеров, несколько fusion-семейств и агрегированные таблицы результатов. Особенно полезны torch-реализации GAF, STFT и MTF: по приложенной таблице они быстрее референсов при очень малом RMSE: GAF примерно `3.27x`, STFT примерно `16.37x`, MTF примерно `3.16x`.

По качеству сигнал положительный, но не без оговорок. По сравнению с лучшим single-view neural baseline fusion-семейства дают медианный прирост на 5 из 6 датасетов почти для всех семейств, а `gated` положителен на всех 6. Лучшие семейства по датасетам: `film` на Crop, `raw_centered_residual` на ElectricDevices, `bottleneck` на FordA, NonInvasiveFetalECGThorax1 и UWaveGestureLibraryAll, `gated` на StarLightCurves. Средний delta к single median по семействам: `bottleneck` около `+0.057`, `raw_centered_residual` около `+0.053`, `gated` около `+0.052`, `concat` около `+0.049`, `film` около `+0.047`.

Главный минус: это пока не доказательство превосходства над сильными classical baselines. `minirocket_ridge` в приложенных результатах выигрывает или почти выигрывает на FordA, NonInvasiveFetalECGThorax1, StarLightCurves и UWaveGestureLibraryAll. Поэтому интеграция должна идти как экспериментальный Industrial-native модуль и benchmark candidate, а не как замена текущих baseline-моделей.

Еще один риск: прототип живет как самостоятельный пакет `src/` с собственным training loop, загрузкой данных, агрегацией результатов и зависимостями `pyts`, `sktime`, `scipy`. В Industrial уже есть похожие точки расширения: `FeatureGeneratorProtocol`, `FeatureBundle`, `RepositoryFeatureGeneratorAdapter`, `BudgetedRepositoryFeatureGeneratorAdapter`, `build_generator_registry()`, torch backend для `quantile_extractor_torch` и `recurrence_extractor`, benchmark `ModelSpec` и analysis utilities. Поэтому перенос "как есть" создаст второй фреймворк внутри фреймворка. Правильный путь: забрать математически полезные и протестированные части, но посадить их на текущие контракты Industrial.

Перспективность интеграции: высокая, если идти поэтапно. Первый слой: image-transform operations и train-aware multimodal preprocessing. Второй слой: компактный MVP neural fusion с raw/stats/GAF/STFT. Третий слой: bottleneck-варианты и полноценная benchmark-воспроизводимость с MiniRocket как обязательным контролем.

## Черновики issue

### TRE-109: Add Torch Image Transformations For FUTURE Modalities

Актуализировано по текущему коду: 2026-06-15.

## Текущее состояние

В FUTURE-прототипе есть torch-реализации image-based модальностей:

- `GAF` в `src/representations/image_transformation/methods/gaf_transformation.py`;
- `MTF` в `src/representations/image_transformation/methods/mtf_transformation.py`;
- `STFTSpectrogram` в `src/representations/image_transformation/methods/stft_transformation.py`;
- общий `ImageTransformer` и mapping трансформаций.

В приложенном benchmark по трансформациям torch-версии быстрее референсов: GAF примерно `3.27x` против `pyts`, STFT примерно `16.37x` против `scipy`, MTF примерно `3.16x` против `pyts`; RMSE находится на уровне `1e-9...1e-14`.

В Industrial уже есть torch backend для статистических и recurrence-признаков: `fedot_ind/core/operation/transformation/torch_backend/statistical/` и `fedot_ind/core/operation/transformation/torch_backend/recurrence/`. GAF/MTF/STFT как reusable Industrial operations или kernel-learning generators в текущем коде не найдены. Старый spectrogram-код в `benchmark/industrial/legacy/forecasting_data.py` связан с конкретным legacy-сценарием и не является общей модальностью.

## Что сделать

Добавить Industrial-native torch image transformations для GAF, MTF и STFT без переноса внешнего пакета `src/` целиком.

Разместить код в согласованном месте рядом с существующими torch backend-модулями, например `fedot_ind/core/operation/transformation/torch_backend/image/`, либо в существующей ветке representation, если это лучше ложится на локальный стиль.

Для каждой трансформации зафиксировать:

- поддерживаемые формы входа `1D`, `2D`, `3D`;
- правила batch/channel flattening и восстановления формы;
- параметры из FUTURE-конфига: `image_size`, `method`, `overlapping`, `n_bins`, `strategy`, `window_size`, `hop_length`, `n_fft`, `window_type`, `center`, `power`;
- device policy через существующий `resolve_torch_device`;
- стабильную форму выхода для CNN-входа;
- понятные ошибки для коротких рядов и неподдержанных параметров.

Если трансформации подключаются в `kernel_learning`, добавить `OperationSpec`/registry entry только после того, как shape и finite-value contract покрыты тестами.

## Что уже не надо делать повторно

Не переносить заново `TorchQuantileExtractor` и torch recurrence: в Industrial они уже есть.

Не добавлять `pyts` и `sktime` в обязательные зависимости только ради reference checks. Они могут использоваться только в optional parity tests через `pytest.importorskip(...)`.

Не переносить `ImageTransformer` с опечатками в публичных параметрах вроде `tranfromation_type` и `transfromation_params`; публичный Industrial API должен иметь нормальные имена.

## Критерии готовности

GAF, MTF и STFT доступны как локальные Industrial torch transformations с предсказуемыми формами выхода.

На синтетических рядах и маленьком batch нет `NaN/inf`.

STFT корректно нормализует `n_fft`, `window_size` и `hop_length` для коротких рядов или дает понятную ошибку.

Опциональные parity tests подтверждают близость к `pyts`/`scipy`, когда эти библиотеки установлены.

## Проверки

Новые unit-тесты рядом с torch backend tests: shape, short series, constant series, batch input, multichannel input, finite values, CPU device.

Optional parity tests для GAF/MTF/STFT с `pytest.importorskip("pyts")` и `pytest.importorskip("scipy")`.

## Не входит в задачу

Не подключать neural fusion classifier и benchmark preset. Это отдельные задачи после появления стабильных модальностей.

### TRE-110: Formalize Train-Aware Multimodal Preparation Contract

Актуализировано по текущему коду: 2026-06-15.

## Текущее состояние

FUTURE-прототип содержит важную логику подготовки данных в `experiments/fusion_over_raw_experiment.py`:

- raw: per-sample z-normalization;
- stats: извлечение torch statistical features, impute `NaN/inf` по train mean, затем z-нормализация по train mean/std;
- GAF: per-sample minmax в `[-1, 1]`, затем image standardization по train statistics;
- STFT: `log1p` для спектрограмм, затем image standardization по train statistics;
- split: official train/test, затем stratified train/val `80/20` внутри official train.

В текущем Industrial `RepositoryFeatureGeneratorAdapter` и `FeatureBundle` работают с одной feature matrix. Это хорошо для kernel-learning generators, но не описывает multimodal payload вида `{"raw": tensor, "stats": tensor, "gaf": tensor, "stft": tensor}` и train-fitted normalization state.

В benchmark classification path сейчас данные приводятся к матрице через `_normalize_matrix`, поэтому прямой перенос FUTURE training loop будет обходить существующий benchmark contract.

## Что сделать

Выделить train-aware preparation слой для мультимодальных моделей без привязки к FUTURE-экспериментальному скрипту.

Минимальный контракт должен уметь:

- `fit` на train split и сохранить statistics для каждой модальности;
- `transform` для val/test/predict без пересчета statistics на этих данных;
- вернуть словарь модальностей с тензорами и metadata с формами, параметрами, device и normalization policy;
- явно проверять совпадение числа samples между модальностями;
- работать с существующим `fedot_ind.tools.loader.DataLoader` и benchmark `ClassificationDatasetRecord`;
- не ломать текущий matrix-based путь для обычных baseline-моделей.

Название и расположение можно выбрать по текущему стилю, например `fedot_ind/core/models/nn/multimodal/preprocessing.py` или отдельный слой рядом с model adapter.

## Что уже не надо делать повторно

Не писать новый UCR/UEA loader. Использовать текущий `fedot_ind.tools.loader.DataLoader` и `benchmark/industrial/datasets/local_io.py`.

Не тащить `PreparedDataset` из прототипа как публичный API один к одному. Его можно использовать как источник правил, но финальный контракт должен быть Industrial-native.

Не использовать test split для подбора normalization statistics.

## Критерии готовности

Один и тот же fitted preprocessor дает стабильные формы и значения для train/val/test transform.

В metadata видны shapes модальностей, normalization policy и параметры image/stat transforms.

На коротких, константных и multichannel рядах поведение покрыто тестами.

## Проверки

Unit-тесты на no-leakage: statistics считаются на train и не меняются при transform test.

Unit-тесты на согласованность batch size и числа samples.

Smoke-тест с маленьким synthetic classification dataset.

## Зависимости

Зависит от TRE-109 для GAF/MTF/STFT, если image-модальности включены в проверку.

### TRE-111: Port FUTURE Encoder Registry And MVP Fusion Classifier

Актуализировано по текущему коду: 2026-06-15.

## Текущее состояние

В FUTURE-прототипе есть аккуратный neural layer:

- `RawTimeSeriesEncoder`, `StatisticalEncoder`, `GAFEncoder`, `STFTEncoder`, `MTFEncoder`;
- `ENCODER_REGISTRY`, который строит encoder по имени модальности;
- flexible classifiers для `concat`, `gated`, `film`, `raw_centered_residual` и bottleneck-вариантов;
- классификационная голова `ClassificationHead`.

В Industrial уже есть общий neural stack в `fedot_ind/core/models/nn/`, `BaseNeuralModel`, network modules и device helpers. При этом мультимодального classifier adapter для словаря модальностей в текущем benchmark path нет.

По результатам FUTURE простые fusion-семейства не бесполезны: `gated` дает положительный delta к single median на 6 из 6 датасетов, `film`, `concat` и `raw_centered_residual` на 5 из 6. Это хороший MVP до bottleneck-слоя.

## Что сделать

Перенести encoder registry и MVP fusion classifier в Industrial-стиле.

MVP-состав:

- модальности: `raw`, `stats`, `gaf`, `stft`;
- fusion methods: `concat`, `gated`, `raw_centered_residual`, `film`;
- `d_model`, dropout, hidden dims и encoder kwargs задаются через params/model config;
- forward принимает mapping модальностей и может вернуть auxiliary diagnostics.

Diagnostics должны включать хотя бы:

- список активных модальностей;
- embedding dimension;
- число параметров;
- для `gated`: веса gate;
- для `raw_centered_residual`: статистику `alpha`;
- для `film`: нормы или summary по `gamma`/`beta`.

## Что уже не надо делать повторно

Не переносить training loop из `src/models/common/training.py` как основной Industrial training runtime. Использовать существующий стиль `BaseNeuralModel`/benchmark adapter либо добавить тонкий adapter, который не дублирует весь benchmark stack.

Не включать bottleneck-варианты в MVP. Они требуют отдельной задачи из-за риска переобучения и более сложной диагностики.

Не подключать MTF в MVP fusion, если к этому моменту нет подтвержденного Industrial use case. MTF может остаться как transformation из TRE-109.

## Критерии готовности

Модель строится из списка модальностей и fusion method без ручного if-дерева в пользовательском коде.

Forward-pass работает на CPU для synthetic batch по каждой MVP-комбинации.

Ошибки по неизвестной модальности, отсутствующему входу и несовместимому `d_model/num_heads` понятны.

Auxiliary diagnostics доступны без изменения logits.

## Проверки

Unit-тесты на encoder registry.

Unit-тесты на forward shape для `concat`, `gated`, `raw_centered_residual`, `film`.

Unit-тесты на missing modality и duplicate modality.

## Зависимости

Зависит от TRE-110 для train-aware preprocessing.

### TRE-112: Add Bottleneck Fusion Variants After MVP Stabilization

Актуализировано по текущему коду: 2026-06-15.

## Текущее состояние

В FUTURE-прототипе есть bottleneck-семейство:

- `ordinary_bottleneck`;
- `raw_residual_bottleneck`;
- `context_only_residual_bottleneck`;
- `raw_conditioned_context_bottleneck`.

По приложенным результатам bottleneck-семейство выглядит наиболее перспективным среди fusion-family aggregations: средний delta к single median около `+0.057`, медианный около `+0.039`, top family на 3 из 6 датасетов. Внутри bottleneck-вариантов лучший метод зависит от датасета: `context_only_residual_bottleneck` лучший на Crop, StarLightCurves и UWaveGestureLibraryAll, `ordinary_bottleneck` лучший на FordA и NonInvasiveFetalECGThorax1, `raw_residual_bottleneck` лучший на ElectricDevices. `raw_conditioned_context_bottleneck` в этих CSV ни разу не top-1 среди bottleneck-вариантов.

Риск: bottleneck сложнее, имеет больше параметров, attention/latent pooling могут переобучаться на малых датасетах и труднее диагностируются.

## Что сделать

После MVP из TRE-111 добавить bottleneck-варианты как отдельный расширенный режим.

Первая очередь:

- `ordinary_bottleneck`;
- `raw_residual_bottleneck`;
- `context_only_residual_bottleneck`.

`raw_conditioned_context_bottleneck` добавить только если будет отдельное обоснование или ablation, потому что в приложенных результатах он не был лидером.

Для bottleneck-моделей обязательно сохранить diagnostics:

- attention summary;
- pooling mode;
- `num_latents`, `num_heads`, `num_layers`;
- alpha/residual strength, если есть residual path;
- parameter count и train duration.

## Что уже не надо делать повторно

Не переписывать encoder registry и preprocessing. Использовать TRE-110 и TRE-111.

Не включать все bottleneck-варианты в default preset до появления smoke-тестов и сравнения с MiniRocket.

## Критерии готовности

Bottleneck models собираются тем же публичным config path, что и MVP fusion models.

Forward-pass и короткое обучение на synthetic dataset проходят на CPU.

Diagnostics позволяют понять, не схлопнулся ли residual/gating/attention путь.

## Проверки

Unit-тесты на shape, missing modalities, invalid `num_heads`, invalid pooling.

Smoke-тест обучения на маленьком synthetic dataset с `epochs=1..2`.

Benchmark smoke через новый adapter из TRE-113.

## Зависимости

Зависит от TRE-111.

### TRE-113: Add Industrial Benchmark Adapters For FUTURE And MiniRocket Ridge

Актуализировано по текущему коду: 2026-06-15.

## Текущее состояние

В FUTURE-эксперименте есть внешний baseline `minirocket_ridge`, и он важен для честной оценки. В приложенных результатах `minirocket_ridge` выигрывает или почти выигрывает на нескольких датасетах: NonInvasiveFetalECGThorax1, StarLightCurves, UWaveGestureLibraryAll и FordA.

В Industrial уже есть `MiniRocketExtractor` в `fedot_ind/core/models/nn/network_impl/feature_extraction/mini_rocket.py` и регистрация `minirocket_extractor` в `model_repository.py`. Но в текущем `benchmark/industrial/models/classification.py` нет benchmark adapter для `MiniRocket + RidgeClassifierCV`.

Текущий classification benchmark поддерживает `majority_class`, `nearest_centroid`, `kernel_ensemble_classifier`, `pdl_classifier` и optional `fedot_industrial_classifier`. FUTURE multimodal classifier там также не подключен.

## Что сделать

Добавить два benchmark adapter-а:

- `MiniRocketRidgeClassifierAdapter`, который использует существующий Industrial `MiniRocketExtractor` и `RidgeClassifierCV` или существующий эквивалентный pipeline;
- `FutureMultimodalClassifierAdapter`, который использует preprocessing из TRE-110 и neural models из TRE-111/TRE-112.

Для FUTURE adapter-а параметры должны включать:

- список модальностей;
- fusion method;
- seed;
- train/val split policy;
- epochs, patience, learning rate, batch sizes;
- torch device;
- output diagnostics flag.

Adapter должен сохранять результат в существующий benchmark record/artifact flow, а не писать отдельные `results/fusion_over_raw` файлы по схеме прототипа.

## Что уже не надо делать повторно

Не переносить `experiments/fusion_over_raw_experiment.py` как второй benchmark runner.

Не удалять и не обходить `ModelSpec`, `RunStatus`, incremental artifacts и текущую result analysis инфраструктуру.

Не сравнивать FUTURE только с raw neural baseline. MiniRocket должен быть обязательным контрольным baseline для честного вывода.

## Критерии готовности

`build_classification_model()` умеет создать MiniRocketRidge и FUTURE adapters по `ModelSpec`.

MiniRocketRidge smoke-тест проходит на маленьком synthetic или локальном UCR subset.

FUTURE adapter умеет обучиться на маленьком dataset с одной MVP-конфигурацией и вернуть predictions в benchmark format.

Failures optional neural dependencies не ломают весь benchmark suite, а дают понятный `RunStatus`.

## Проверки

Unit-тесты для `build_classification_model`.

Smoke-тест `run_local_benchmark_preset('ucr', ...)` с MiniRocketRidge.

Smoke-тест FUTURE adapter на in-memory TSC record.

## Зависимости

FUTURE adapter зависит от TRE-110 и TRE-111.

### TRE-114: Reproduce fusion_over_raw As An Industrial Benchmark Preset

Актуализировано по текущему коду: 2026-06-15.

## Текущее состояние

FUTURE-прототип хранит экспериментальную постановку в `experiments/configs/fusion_over_raw.json`: 6 UCR/UEA датасетов, seeds `42`, `3407`, `2025`, train/val `80/20`, `macro_f1` как основная метрика, 5 single-view baselines, MiniRocketRidge и 33 fusion-конфигурации.

В Industrial уже есть preset layer: `benchmark/industrial/experiments/presets.py`, `preset_defaults.json`, `ModelSpec`, `DatasetSpec`, `RunSpec`, incremental artifacts и result analysis helpers. Сейчас classification defaults включают только `majority_class` и `nearest_centroid`.

## Что сделать

Добавить Industrial preset или manifest для FUTURE fusion-over-raw эксперимента.

Минимальный воспроизводимый этап:

- datasets из FUTURE: FordA, ElectricDevices, Crop, StarLightCurves, NonInvasiveFetalECGThorax1, UWaveGestureLibraryAll;
- seeds: `42`, `3407`, `2025`;
- metric: `f1_macro` как primary или явно mapped `macro_f1`;
- baselines: raw neural, raw_larger neural, stats, GAF, STFT, MiniRocketRidge;
- MVP fusion methods из TRE-111;
- bottleneck methods только после TRE-112.

Переиспользовать `benchmark/industrial/evaluation/result_analysis.py` и текущие visualization/artifact contracts. Если нужны таблицы из FUTURE, перенести только смысл:

- family vs best single;
- top-3 family by dataset;
- feature-combination sensitivity;
- bottleneck variants comparison.

## Что уже не надо делать повторно

Не переносить `experiments/agregate_results.py` и visualization scripts целиком.

Не хранить checkpoint paths и CSV layout из прототипа как новый стандарт.

Не запускать полный 684-run grid как обязательный unit/integration test.

## Критерии готовности

Есть preset/config, который описывает FUTURE experiment через текущие Industrial benchmark сущности.

Малый smoke preset запускает 1 датасет, 1 seed, 1 baseline и 1 fusion model.

Result analysis строит таблицу прироста к best single и comparison с MiniRocket.

## Проверки

Unit-тесты на загрузку preset defaults.

Smoke-тест на in-memory или маленьком локальном UCR dataset.

Тест, что `f1_macro` правильно выбирается как primary metric.

## Зависимости

Зависит от TRE-113.

### TRE-115: Add FUTURE Dependency And Documentation Gate

Актуализировано по текущему коду: 2026-06-15.

## Текущее состояние

FUTURE-прототип требует `torch`, `pyts`, `scipy`, `sktime`, `pandas`, `matplotlib`, `seaborn`. В Industrial `torch` уже фактически используется через `fastai` и локальные torch-модули, `scipy` приходит через другие зависимости, но `pyts` и `sktime` не заявлены в `pyproject.toml`/`requirements.txt` как обязательные зависимости.

Основная ценность FUTURE для Industrial не требует обязательного `pyts`/`sktime`: torch image transforms можно держать self-contained, а reference-библиотеки использовать только для parity checks.

Документация по текущим TRE-issue обычно фиксирует, что уже есть в коде, что не надо делать повторно и какие проверки должны пройти. Для FUTURE нужен такой же короткий integration note, чтобы следующий исполнитель не начал с копирования внешнего `src/`.

## Что сделать

Подготовить dependency decision note для FUTURE-интеграции.

Зафиксировать:

- какие зависимости уже есть или транзитивно доступны;
- какие зависимости должны остаться optional;
- нужны ли `pyts` и `sktime` только для тестов;
- как запускать parity tests без обязательной установки optional packages;
- где хранить FUTURE docs и benchmark reproduction instructions.

Добавить короткую страницу в `docs/` с итогами анализа метода, staged integration path и ссылками на новые TRE-issue.

## Что уже не надо делать повторно

Не добавлять `pyts` и `sktime` в базовые install requirements без отдельного решения.

Не включать полный zip-прототип или его `uv.lock` в репозиторий.

Не создавать notebook-first tutorial до появления стабильного Industrial adapter.

## Критерии готовности

Есть decision note по зависимостям и optional tests.

Есть короткая документация по FUTURE integration path.

Базовый import Industrial не требует `pyts`/`sktime`.

## Проверки

Базовые unit-тесты проходят без optional reference dependencies.

Optional parity tests пропускаются через `pytest.importorskip(...)`, если reference package отсутствует.

## Зависимости

Связано с TRE-109, TRE-113 и TRE-114.
