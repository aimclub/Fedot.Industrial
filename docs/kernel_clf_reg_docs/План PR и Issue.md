## ## Целевая архитектура модуля

Предлагаемая структура:

fedot_ind/core/kernel_learning/
generators/
base.py
statistical.py
wavelet.py
fourier.py
eigen.py
topology.py
shapelet.py
embedding.py
okhs.py
kernels/
builder.py
normalization.py
psd.py
approximations.py
selection/
objectives.py
sparse_mkl.py
classwise_mkl.py
horizon_mkl.py
estimators/
classifier.py
regressor.py
forecaster.py
integration/
fedot_strategy_adapter.py
initial_assumption_builder.py
diagnostics/
reports.py
explainability.py

Основные классы:

FeatureGeneratorProtocol
KernelGeneratorProtocol
KernelBundle
KernelSelectionReport
SparseMKLSelector
KernelEnsembleClassifier
KernelEnsembleRegressor
KernelEnsembleForecaster
FedotKernelStrategyAdapter

## План реализации: Issues и PRs

### Issue 1. Ввести единый kernel/feature-generator contract

**Проблема.** Сейчас генераторы встроены в `KERNEL_BASELINE_FEATURE_GENERATORS` как FEDOT pipelines, но нет единого
контракта, который гарантирует fit/transform/kernel/diagnostics.

**Почему важно.** Без контракта невозможно безопасно добавлять другие генераторы topology, shapelets, TS2Vec, MOMENT и
другие генераторы.

**Варианты.**

| Вариант                                                          | Трудозатраты | Риск    | Отдача  | Поддержка |
|------------------------------------------------------------------|--------------|---------|---------|-----------|
| Использовать текущие PipelineBuilder objects                     | низкие       | высокий | средняя | низкая    |
| Добавить thin adapters вокруг текущих pipelines                  | средние      | низкий  | высокая | средняя   |
| Полностью заменить FEDOT nodes на отдельную sklearn-like систему | высокие      | средний | высокая | высокая   |

**Рекомендация:** thin adapters. FEDOT nodes сохранить, но обернуть.

**PRs.**

- PR 1.1: добавить `FeatureGeneratorProtocol` / `KernelGeneratorProtocol`.
- PR 1.2: добавить `FeatureBundle` и `KernelBundle`.
- PR 1.3: адаптеры для `quantile`, `wavelet`, `fourier`, `eigen`.
- PR 1.4: registry с lazy initialization и параметрами генераторов.
- PR 1.5: unit tests на contract: fit/transform shape, deterministic output, no target leakage.

### Issue 2. Исправить построение kernel matrices

**Проблема.** Сейчас distance matrices передаются как kernels.

**Почему важно.** MKL должен работать с валидными similarity/Gram matrices.

**Рекомендация:** distance-to-kernel + PSD validation; PSD projection как configurable fallback.

**PRs.**

- PR 2.1: `KernelMatrixBuilder` с RBF, Laplacian, cosine, linear, polynomial.
- PR 2.2: train-train и test-train kernel construction.
- PR 2.3: centering, trace normalization, Frobenius normalization.
- PR 2.4: PSD diagnostics: min eigenvalue, condition number, correction flag.
- PR 2.5: property tests: symmetry, shape, train/test consistency, PSD tolerance.

---

### Issue 3. Переписать `KernelEnsembler` в полноценный estimator

**Проблема.** Текущий `KernelEnsembler` возвращает tuple с pipelines и data, а не обученную модель.

**Почему важно.** Это ограничивает переиспользование и повышает связность с `IndustrialStrategy`.

**Рекомендация:** Разделить selector и strategy builder. Сделать task-specific estimators

**PRs.**

- PR 3.1: `KernelFeatureSelector.fit(X, y)` возвращает weights и selected generators.
- PR 3.2: `KernelEnsembleClassifier.fit/predict/predict_proba`.
- PR 3.3: совместимость с `IndustrialStrategy`: старый путь временно оставить через adapter.
- PR 3.4: исправить `kernel_strategy` → `kernel_strategy` с backward compatibility.
- PR 3.5: убрать stateful accumulation `feature_matrix_train` между вызовами.

---

### Issue 4. Реализовать sparse/adaptive MKL objective с регуляризацией сложности

**Проблема.** В коде нет целевой функции из главы 3 с $C(K)$.

**Почему важно.** Это центральная научная новизна главы 3.

**Рекомендация:** Реализовать собственный optimizer для α, а библиотеку MKLpy использовать как baseline.

**PRs.**

- PR 4.1: `KernelAlignmentObjective`.
- PR 4.2: `kernel_complexity(K)` по $C(K)$.
- PR 4.3: redundancy penalty между ядрами.
- PR 4.4: sparse simplex optimizer: projected gradient / scipy constrained optimization.
- PR 4.5: class-specific и global weights.
- PR 4.6: tests на synthetic kernels: должен выбирать информативное ядро и отбрасывать noise kernel.

### Issue 5. Вернуть и стабилизировать топологический генератор

**Проблема.** `topological_extractor` заявлен, но не активен.

**Почему важно.** Это один из четырех базовых генераторов главы 2 и важная часть научной преемственности.

**Рекомендация:** optional topology с hard budget, затем multi-fidelity.

**PRs.**

- PR 5.1: adapter для `topological_extractor`.
- PR 5.2: параметры TDA: embedding dimension, delay, max epsilon, homology dimensions.
- PR 5.3: кэширование persistence/topological features.
- PR 5.4: fallback при слишком большом N/L.
- PR 5.5: tests на простых синус/шум/ступенчатых рядах.

### Issue 6. Добавить shapelet/CNN-kernel генератор

**Проблема.** В текущем составе нет shapelet-представлений, хотя современная литература показывает связь shapelets и CNN
kernels.

**Почему важно.** Shapelet generator даст интерпретируемые локальные паттерны, полезные для промышленной диагностики.

**Рекомендация:** начать с random/soft shapelet transform как генератора признаков. Позже добавить learnable soft
shapelet / expert routing

**PRs.**

- PR 6.1: `ShapeletFeatureGenerator`.
- PR 6.2: shapelet distance kernel.
- PR 6.3: class-wise shapelet diagnostics.
- PR 6.4: benchmark на UCR classification subset.

### Issue 7. Добавить self-supervised/foundation embedding generators

**Проблема.** Текущий подход основан только на hand-crafted feature spaces.

**Почему важно.** Работы TS2Vec, CoST, TimesURL, MOMENT, TimesFM, Moirai показывают, что универсальные представления
временных рядов стали стандартным направлением A*/top-tier исследований.

**Рекомендация:** optional adapters без обязательных тяжелых зависимостей.

**PRs.**

- PR 7.1: `EmbeddingGeneratorProtocol`.
- PR 7.2: adapter для TS2Vec-like embeddings.
- PR 7.3: adapter для MOMENT embeddings как optional dependency.
- PR 7.4: foundation model output-as-feature и embedding-as-kernel.
- PR 7.5: tests with mocked embedding model, чтобы CI не тянул тяжелые веса.

### Issue 8. Реализовать `KernelEnsembleRegressor`

**Проблема.** В `KernelEnsembler` нет регрессии. В `UnifiedOKHSAutoML._fit_regression`
используется `RKBSCompositeClassifier` с L2 penalty: `kernel_automl.py:165-176`, что выглядит концептуально некорректно,
если класс действительно является classifier.

**Рекомендация:** `KernelRidge` и `SVR(kernel="precomputed")` как first-class heads.

**PRs.**

- PR 8.1: `KernelEnsembleRegressor`.
- PR 8.2: regression target kernel KyK_yKy​.
- PR 8.3: MAE/RMSE/R2 validation.
- PR 8.4: multi-output regression support.
- PR 8.5: tests на synthetic trend/seasonal/noisy datasets.

### Issue 9. Реализовать `KernelEnsembleForecaster`

**Проблема.** Forecasting-идея главы 3 не реализована в `kernel_ensemble`; OKHS существует отдельно.

**Рекомендация:** kernel-gated forecasting с OKHS/SSA/ARIMA/FEDOT heads.

**PRs.**

- PR 9.1: rolling-origin splitter.
- PR 9.2: horizon-specific target kernels.
- PR 9.3: spectral clustering / kernel k-means gating.
- PR 9.4: heads: SSA+AR, ARIMA fallback, OKHSForecaster.
- PR 9.5: aggregation strategy: selected head, weighted heads, conformal intervals.
- PR 9.6: tests на synthetic seasonal/trend/chaotic/noise series.

### Issue 10. Интеграция с FEDOT search space

**Проблема.** Сейчас kernel strategy создает pipelines, затем tuner донастраивает
head: `industrial_strategy.py:194-211`. Это полезно, но не дает FEDOT видеть kernel-selection diagnostics как часть
search space.

**Почему важно.** Нужно, чтобы FEDOT мог использовать выбранные генераторы как prior/initial assumption, а не только как
отдельную strategy.

**Рекомендация:** selected generators как initial assumptions; node-level integration позже.

**PRs.**

- PR 10.1: `KernelSelectionReport` → FEDOT initial assumptions.
- PR 10.2: search-space constraints на основе selected generators.
- PR 10.3: generator weights как meta-features для optimizer.
- PR 10.4: integration tests с `IndustrialStrategy`.

---

### Issue 11. Тестирование и валидация

**Проблема.** Нет достаточного тестового покрытия `KernelEnsembler`.

**Почему важно.** Новые генераторы и ядра быстро приведут к скрытым ошибкам shape/PSD/leakage.

**Рекомендация:** contract/property tests в CI; benchmarks отдельно.

**PRs.**

- PR 11.1: unit tests for generator registry.
- PR 11.2: unit tests for kernel matrix builder.
- PR 11.3: property tests for PSD/symmetry/normalization.
- PR 11.4: multiclass mapping tests.
- PR 11.5: regression/forecasting estimator tests.
- PR 11.6: leakage tests for forecasting split.

### Issue 12. Производительность и масштабирование

**Проблема.** Kernel matrices имеют стоимость $O(mn^2)$, где m — число генераторов. Topology/shapelets/foundation
embeddings добавят стоимость.

**Почему важно.** Без approximation/caching модуль будет плохо масштабироваться.

**Варианты.**

**Рекомендация:** cache + budget сразу; Nyström/random features/leverage sampling для больших N.

**PRs.**

- PR 12.1: feature/kernel cache by dataset hash + generator params.
- PR 12.2: memory diagnostics.
- PR 12.3: generator budget policy.
- PR 12.4: Nyström approximation for kernels.
- PR 12.5: benchmark report: wall-clock, memory, kernel quality.