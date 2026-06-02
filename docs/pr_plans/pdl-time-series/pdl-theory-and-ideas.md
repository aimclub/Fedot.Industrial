---
title: "PDL — теория и идеи применения"
aliases:
  - Pairwise Difference Learning
  - Target-relation augmentation
  - Аугментация таргета
tags:
  - pdl
  - time-series
  - machine-learning
  - classification
  - regression
  - research
status: approved-for-planning
source:
  - PDL_learning.pdf
  - Fedot.Industrial/refactor_monad_usage
created: 2026-06-01
---

# PDL — теория и идеи применения

Связанные заметки:

- [[Проекты для разработки/Исследовательские проекты/Industrial/Трек по классификации и регрессии/PDL/pdl-development-issues]]
- [[pdl_pr_plan|PDL — план PR]]

## 1. Короткая формулировка идеи

**Pairwise Difference Learning** можно трактовать как частный случай более широкой идеи:

> **Target-relation augmentation** — построение дополнительных supervised-задач не из самих объектов, а из отношений
> между их таргетами.

Вместо стандартной постановки:

$$
x_i \mapsto y_i
$$

строится задача над парами:

$$
(x_i, x_j) \mapsto R(y_i, y_j)
$$

где $R$ — отношение между таргетами: разность, совпадение класса, относительное изменение, ранговый порядок, близость
классов, similarity score или другой supervised-сигнал.

Главная инженерная ценность PDL: исходный набор из $N$ объектов превращается в набор из $O(N^2)$ пар или $O(N \cdot A)$
пар при использовании ограниченного множества якорей $A$.

Главная исследовательская ценность PDL: модель учится не абсолютному отображению $x \mapsto y$, а относительному правилу
переноса таргета от одного объекта к другому.
йййййййййй

## 2. PDL для регрессии

### 2.1. Базовая математическая постановка

Пусть есть обучающая выборка:

$$
D = \{(x_i, y_i)\}_{i=1}^{N}, \quad y_i \in \mathbb{R}
$$

Искомая функция:

$$
y = f(x)
$$

Ключевое тождество PDL:

$$
f(x) = f(x') + \Delta(x, x')
$$

где:

$$
\Delta(x, x') = f(x) - f(x')
$$

Вместо прямого обучения $f(x)$ обучается **модель разностей**:

$$
\tilde{\Delta}(x_i, x_j) \approx y_i - y_j
$$

Парный обучающий датасет:

$$
D_{pair} = \{(\phi(x_i, x_j), y_i - y_j) \mid i,j = 1,\ldots,N\}
$$

где $\phi(x_i, x_j)$ — функция построения признаков пары.

Для нового объекта $x_q$ прогноз строится через anchors:

$$
\hat{y}_q = \frac{1}{A}\sum_{a \in \mathcal{A}} \left(y_a + \tilde{\Delta}(x_q, x_a)\right)
$$

где $\mathcal{A}$ — множество якорных обучающих объектов.

### 2.2. Почему это работает

PDL-регрессия сочетает два механизма:

1. **Увеличение числа семплов для обучения**: вместо $N$ наблюдений появляется до $N^2$ пар. Особенно полезно для
   сценариев где у нас мало семплов и много классов.
2. **Усреднение по якорям**: каждый якорь даёт частный прогноз (похоже/не похоже на этот конкретный класс), а итоговая
   оценка строится как агрегат.

Если ошибки по якорям не полностью коррелированы, усреднение снижает дисперсию ошибки. На практике независимость ошибок
не выполняется, но эффект ансамблирования всё равно может быть полезен.

### 2.3. Как преобразовывать таргет (пример для регрессии)

Базовый вариант:

$$
\delta_{ij} = y_i - y_j
$$

Варианты развития:

#### Absolute delta

$$
\delta_{ij} = |y_i - y_j|
$$

Использовать для обычной регрессии с аддитивной ошибкой.

#### Log-delta

$$
\delta_{ij}^{log} = \log(y_i + \varepsilon) - \log(y_j + \varepsilon)
$$

Прогноз:

$$
\hat{y}_q = \exp\left(\operatorname{Agg}_{a \in \mathcal{A}}[\log(y_a + \varepsilon) + \hat{\delta}^{log}_{qa}]\right) -
\varepsilon
$$

Использовать для положительных таргетов, где относительная ошибка важнее абсолютной: спрос, энергия, вибрация,
деградация.

#### Relative delta

$$
\delta_{ij}^{rel} = \frac{y_i - y_j}{|y_j| + \varepsilon}
$$

Подходит для задач которые чувствителен к масштабу.

#### Rank-sign target

$$
r_{ij} = \operatorname{sign}(y_i - y_j)
$$

Подходит, когда абсолютное значение не так важно, но порядок объектов надёжнее. Условно - "есть пересечение трешхолда
или нет"

#### Quantized delta

$$
r_{ij} = \operatorname{bin}(y_i - y_j)
$$

Это переводит регрессию разности в классификацию интервалов.

#### Multi-output delta

Для векторного таргета:

$$
y_i \in \mathbb{R}^{H}
$$

$$
\delta_{ij} = y_i^{(1:H)} - y_j^{(1:H)}
$$

Это применимо для multi-horizon forecasting.

---

## 3. PDL для классификации

### 3.1. Базовая постановка

Пусть:

$$
D = \{(x_i, y_i)\}_{i=1}^{N}, \quad y_i \in \{1, \ldots, K\}
$$

Классификационная версия PDL заменяет multiclass-задачу одной бинарной задачей на парах:

$$
r_{ij} = \mathbb{1}[y_i = y_j]
$$

То есть модель учится отвечать на вопрос:

> Принадлежат ли два объекта одному классу?

Парный датасет:

$$
D_{pair} = \{(\phi(x_i, x_j), r_{ij}) \mid i,j = 1,\ldots,N\}
$$

Обучается бинарный классификатор:

$$
\gamma(x_i, x_j) \approx P(y_i = y_j \mid x_i, x_j)
$$

### 3.2. Важная деталь семантики таргета

В статье удобно использовать:

$$
1 = same, \quad 0 = different
$$

В текущей реализации Fedot.Industrial используется обратная внутренняя семантика:

$$
0 = same, \quad 1 = different
$$

Это допустимо, если весь pipeline согласован. Но для развития модуля нужно явно фиксировать контракт
через `pair_target_semantics`, иначе легко сломать агрегацию вероятностей и calibration.

### 3.3. Признаковое описание "пары"

Базовая формула:

$$
\phi(x_i, x_j) = [x_i, x_j, x_i - x_j]
$$

Варианты:

$$
\phi_{concat\_diff}(x_i, x_j) = [x_i, x_j, x_i - x_j]
$$

$$
\phi_{concat\_absdiff}(x_i, x_j) = [x_i, x_j, |x_i - x_j|]
$$

$$
\phi_{diff\_only}(x_i, x_j) = x_i - x_j
$$

Для симметричных задач безопаснее использовать `absdiff` или симметризованный inference.

### 3.4. Symmetry

Отношение "один класс" симметрично:

$$
\gamma(x_i, x_j) = \gamma(x_j, x_i)
$$

Если пара фичей несимметрична из-за concat-порядка, полезно использовать:

$$
\gamma_{sym}(x_i, x_j) = \frac{\gamma(x_i, x_j) + \gamma(x_j, x_i)}{2}
$$

Это снижает риск, что модель выучит артефакты порядка пары.

## 4. Агрегация вероятностей для классификации

### 4.1. Простая агрегация по сходству

Для интересующего семпла $x_q$ считаются similarity scores до якоря:

$$
s_{qa} = \gamma(x_q, x_a)
$$

Для каждого класса $c$:

$$
p(c \mid x_q) \propto \frac{1}{|\mathcal{A}_c|}\sum_{a \in \mathcal{A}_c} s_{qa}
$$

После этого вероятности нормируются по классам.

Плюсы:

- простая реализация;
- понятная интерпретация;

Минусы:

- плохо использует negative evidence;
- не учитывает априорное распределение класса;
- чувствителен к числу и качеству якорей.

### 4.2. Апостериорная агрегация

Пусть anchor $a$ имеет класс $y_a$.

Событие:

$$
E_a = [y_q = y_a]
$$

Модель даёт:

$$
P(E_a \mid x_q, x_a) = \gamma_{sym}(x_q, x_a)
$$

Для каждого anchor строится распределение

$$
p_{post,a}(y) =
\begin{cases}
\gamma_{sym}(x_q, x_a), & y = y_a \\
\frac{p(y)(1 - \gamma_{sym}(x_q, x_a))}{1 - p(y_a)}, & y \neq y_a
\end{cases}
$$

Итоговая вероятность:

$$
p_{post}(y \mid x_q) = \frac{1}{A}\sum_{a \in \mathcal{A}} p_{post,a}(y)
$$

Финальная метка:

$$
\hat{y}_q = \arg\max_{y \in \mathcal{Y}} p_{post}(y \mid x_q)
$$

### 4.3. Взвешенное агрегирование

Если anchors имеют веса $w_a$:

$$
\sum_{a \in \mathcal{A}} w_a = 1
$$

то:

$$
p_{post}(y \mid x_q) = \sum_{a \in \mathcal{A}} w_a \cdot p_{post,a}(y)
$$

Веса можно получать из:

- качества якорей на валидации;
- исходного распределения классов;
- расстояния до class prototype;

## 5. Неопределённость в PDL

PDL естественно даёт ensemble-like представление: каждый якорь формирует частный случай вероятностного распределения

Общая неопределенность прогноза:

$$
TU(x_q) = H(p_{post}(y \mid x_q))
$$

По якорям:

$$
AU(x_q) = \frac{1}{A}\sum_{a \in \mathcal{A}} H(p_{post,a}(y \mid x_q))
$$

Ожидаемая:

$$
EU(x_q) = TU(x_q) - AU(x_q)
$$

Для регрессии:

$$
\hat{y}_{q,a} = y_a + \tilde{\Delta}(x_q, x_a)
$$

Среднее:

$$
\hat{y}_q = \operatorname{mean}_{a \in \mathcal{A}} \hat{y}_{q,a}
$$

Дисперсия:

$$
u_q = \operatorname{Var}_{a \in \mathcal{A}}(\hat{y}_{q,a})
$$

## 6. Почему PDL особенно интересен для маленьких датасетов с большим числом классов

Гипотеза:

> PDL заменяет абсолютную задачу $x \mapsto y$ на относительную задачу $(x_i, x_j) \mapsto R(y_i, y_j)$, увеличивая
> число обучающих ограничений и переводя многоклассовую постановку в бинарную/metric-like learning.

Для $N$ объектов и $A$ anchors:

$$
N \rightarrow N \cdot A
$$

Для полного режима:

$$
N \rightarrow N^2
$$

Для классификации с $K$ классами:

$$
K\text{-way classification} \rightarrow binary\ same/different\ classification
$$

Это может быть проще, чем напрямую строить $K$-классовые разделяющие поверхности.

### 6.1. Главный риск - имбаланс пар

Число "положительных пар":

$$
N_{pos} = \sum_{k=1}^{K} n_k^2
$$

Число различающихся пар:

$$
N_{neg} = N^2 - N_{pos}
$$

Если классов много, а объектов на класс мало, то положительных пар становится мало. Модель может научиться почти всегда
говорить `different`, что разрушает полезность

Инженерный вывод:

- нужен семплер который "докидывает нужное число позитив пар"
- нужны якоря которые это учитывают;
- нужны веса для классов/семплов;

## 7. PDL для временных рядов

### 7.1. Обозначения

Временной ряд:

$$
X_i \in \mathbb{R}^{C \times T}
$$

где:

- $C$ — число каналов;
- $T$ — длина ряда.

Для одномерного ряда:

$$
X \in \mathbb{R}^{N \times C \times T}
$$

Цель: не терять временную и канальную структуру при построении пар фичей

### 7.2. Наивный бейзлайн

Можно просто сделать флаттен:

$$
\operatorname{flat}(X_i) \in \mathbb{R}^{C \cdot T}
$$

Тогда пары фичей:

$$
\phi_{raw}(X_i,
X_j) = [\operatorname{flat}(X_i), \operatorname{flat}(X_j), \operatorname{flat}(X_i)-\operatorname{flat}(X_j)]
$$

Это допустимый бейзлайн но не целевая реализация.

Недостатки:

- теряется явная временная ось;
- теряется структура каналов;
- сдвиг по фазе делает похожие ряды далёкими;
- spectral/shape схожесть не учитывается.

### 7.3. Embedding-based PDL

Пусть есть некое преобразование (encoder) временного ряда:

$$
E(X_i) = z_i
$$

Тогда:

$$
\phi_{embedding}(X_i, X_j) = [z_i, z_j, z_i - z_j, |z_i - z_j|]
$$

Encoder может быть (смотрим реализации в индастриале):

- статистический feature extractor;
- spectral feature extractor;
- topological feature extractor;
- ROCKET/MiniRocket-like transform;
- pretrained TS encoder;

### 7.4. Distance-based PDL

Можно добавлять признаки на основе "расстояний":

$$
\phi_{distance}(X_i, X_j) = [
d_{euclidean},
d_{corr},
d_{spectral},
d_{derivative},
d_{shapelet},
d_{dtw-like}
]
$$

Полезные признаки для временных рядов:

- корреляционная близость;
- distance между производными;
- spectral entropy/distance;
- расстояние после z-normalization;
- расстояние после alignment/windowing.

### 7.5. Гибридное представление

Рекомендуемый вариант по "умолчанию":

$$
\phi_{hybrid}(X_i, X_j) = [
E(X_i),
E(X_j),
E(X_i)-E(X_j),
|E(X_i)-E(X_j)|,
D(X_i, X_j)
]
$$

где $D$ — набор специфичных для временных рядов "расстояний".

---

## 8. Идеи как можно "расширить таргеты" для классификации временных рядов

### 8.1. Class-similarity targets

Бинаризация слишком груба для многоклассовых задач. Можно ввести матрицу сходства классов:

$$
S \in [0,1]^{K \times K}
$$

И обучать:

$$
r_{ij} = S_{y_i,y_j}
$$

Источники $S$:

- экспертная иерархия классов;
- confusion matrix бейзлайн-модели;
- расстояние между прототипами классов;
- domain ontology;

Это полезно, если классы не равноудалены друг от друга.

### 8.2. Severity/ranking targets

Если классы имеют порядок:

$$
severity(y_i) \in \mathbb{R}
$$

то можно обучать:

$$
r_{ij} = \operatorname{sign}(severity(y_i) - severity(y_j))
$$

Применимо для:

- стадий деградации;
- уровней дефекта;
- severity-based fault classification.

### 8.3. Supervised contrastive learning

Идея: в embedding space объекты одного класса притягиваются, разных классов — отталкиваются.

PDL и supervised contrastive learning близки концептуально:

- PDL учит модель предсказывать "разницу в классах";
- contrastive learning учит representation, где relation target геометрически выражен.

Практичный вариант для Fedot.Industrial:

1. Извлечь time-series features.
2. Обучить projection/metric model.
3. Использовать PDL classifier поверх embeddings.
4. Агрегировать через anchors/prototypes.

## 9. Идеи аугментации таргета для регрессии и прогнозирования

### 9.1. Delta regression

Текущий базовый PDL:

$$
\delta_{ij} = y_i - y_j
$$

Предикт:

$$
\hat{y}_q = \operatorname{Agg}_{a \in \mathcal{A}}[y_a + \hat{\delta}_{qa}]
$$

### 9.2. Log-ratio regression

Для положительных таргетов:

$$
\delta_{ij}^{log} = \log(y_i + \varepsilon) - \log(y_j + \varepsilon)
$$

### 9.3. Pairwise ranking + calibration

Обучается:

$$
r_{ij} = \operatorname{sign}(y_i - y_j)
$$

Затем скалярное значение восстанавливается через:

- anchor voting;
- isotonic calibration;
- Bradley-Terry-like model;
- quantile aggregation.

### 9.4. Multi-task pair targets

Для пары можно обучать несколько таргетов (многомерная регрессия):

$$
R(y_i, y_j) = [
y_i - y_j,
|y_i - y_j|,
\operatorname{sign}(y_i-y_j),
\mathbb{1}[|y_i-y_j| < \tau]
]
$$

Такой multi-task target заставляет модель учить "амплитуд", "направление" и "сходство".

### 9.5. Forecasting vector-delta

Для горизонта $H$:

$$
y_i = [y_i^{(1)}, \ldots, y_i^{(H)}]
$$

Pair target:

$$
\delta_{ij}^{(h)} = y_i^{(h)} - y_j^{(h)}, \quad h = 1,\ldots,H
$$

Прогноз:

$$
\hat{y}_q^{(h)} = \operatorname{Agg}_{a \in \mathcal{A}}[y_a^{(h)} + \hat{\delta}_{qa}^{(h)}]
$$

---

## 10. Целевые архитектурные абстракции

PDL-модуль лучше развивать как библиотеку стратегий.

```python
@dataclass(frozen=True)
class PairwiseLearningConfig:
    task: Literal["classification", "regression", "forecasting"]
    pair_feature_mode: str
    pair_target_mode: str
    anchor_policy: str
    aggregation_policy: str
    pair_sampling_policy: str
    max_pairs: int
    max_pair_memory_mb: int | None
    anchors_per_class: int
    random_state: int
    nan_policy: Literal["zero", "raise", "indicator", "impute"]
```

Основные интерфейсы:

```python
class PairFeatureBuilder(Protocol):
    def build(self, left: np.ndarray, anchors: np.ndarray) -> np.ndarray: ...

class PairTargetBuilder(Protocol):
    def build(self, y_left: np.ndarray, y_anchor: np.ndarray) -> np.ndarray: ...

class AnchorSelector(Protocol):
    def select(self, X: np.ndarray, y: np.ndarray) -> np.ndarray: ...

class PairSampler(Protocol):
    def sample(self, y: np.ndarray, anchors: np.ndarray) -> PairIndexBatch: ...

class PairAggregator(Protocol):
    def aggregate(self, pair_predictions: np.ndarray, anchors: AnchorSet) -> np.ndarray: ...

class UncertaintyEstimator(Protocol):
    def estimate(self, anchor_predictions: np.ndarray) -> dict[str, np.ndarray]: ...
```

Предлагаемая структура:

```text
fedot_ind/core/models/pdl/
  __init__.py
  config.py
  pair_features.py
  pair_targets.py
  anchors.py
  samplers.py
  aggregators.py
  uncertainty.py
  estimators.py
  ts_adapters.py
  diagnostics.py
  legacy.py
```

---

## 11. Риски и ограничения

### 11.1. Пары не являются независимыми samples

Увеличение $N \rightarrow N^2$ не эквивалентно появлению $N^2$ независимых наблюдений. Это supervised constraints, а не
новые независимые объекты.

### 11.2. Pair imbalance

Для большого числа классов негативные примеры обычно доминируют. Нужен balanced sampler.

### 11.3. Anchor quality

Плохие якоря дают шумное распределение. Нужны стратегии взвешивания и выбора якорей

### 11.4. Time-series flattening

Flatten-only подход не должен быть финальным решением. Он полезен только как baseline.

### 11.5. Memory blow-up

Память оценивается как:

$$
Memory \approx n_{pairs} \cdot d_{pair} \cdot bytes(dtype)
$$

Для `concat_diff`:

$$
d_{pair} = 3F
$$

Поэтому ограничивать нужно не только число пар, но и `n_pairs × pair_dim × dtype`.

## 12. Рекомендуемая траектория развития

1. Стабилизировать контракты pair targets, preprocessing и diagnostics.
2. Вынести pair construction, target construction, anchor selection и aggregation в стратегии.
3. Добавить posterior aggregation и uncertainty.
4. Добавить balanced pair sampling для small-N/high-K.
5. Добавить weighted anchors и prototype anchors.
6. Добавить time-series-aware pair features.
7. Добавить расширенные regression target modes.
8. Добавить memory-aware pair generation.
9. Закрепить всё benchmark-протоколом.

Итоговая продуктовая формулировка:

> PDL-модуль должен стать не одной моделью, а библиотекой "преобразования постановки задачи/таргета" для классификации,
> регрессии и forecasting на временных рядах.

---

## 13. Справочные источники

- Mohamed Karim Belaid, Maximilian Rabus, Eyke Hüllermeier. **Pairwise Difference Learning for Classification**, arXiv:
  2406.20031, 2024.
- Tynes et al. Работы по pairwise difference regression для химических задач.
- Wetzel et al. Twin neural network подходы для semi-supervised regression через target differences.
- Zhang et al. **mixup: Beyond Empirical Risk Minimization**, arXiv:1710.09412.
- Khosla et al. **Supervised Contrastive Learning**, arXiv:2004.11362.
- Wen et al. **Time Series Data Augmentation for Deep Learning: A Survey**, arXiv:2002/2007 family.
- Dempster et al. **MiniRocket**, arXiv:2012.08791.
