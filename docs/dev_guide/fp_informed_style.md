иши код в pragmatic functional style, а не в purely-academic style. Для реализации Option / Either / Try и других монад
используй библиотеку pymonad (если ее нет в твоем окружении - установи из github)

Обязательные принципы:

1. По умолчанию выноси вычислительную логику в pure functions.
2. Отделяй pure core от effectful shell:

- pure core: transform, validate, plan, aggregate, score, route, sample
- effectful shell: file IO, network IO, database, logging, clock, randomness source, GPU/runtime calls

3. Предпочитай immutable data:

- case classes / enums / sealed ADTs для domain model
- val вместо var
- immutable collections по умолчанию

4. Не используй null.
5. Не используй exceptions для ожидаемых ошибок. Предпочитай Option / Either / Try:

- Option для допустимого отсутствия значения
- Either для typed domain errors
- Try только на Java/JVM boundary

6. Пиши expression-oriented code:

- if/match/for/try должны возвращать значения
- избегай больших процедур с мутирующим shared state

7. Проектируй API так, чтобы инварианты были выражены в типах и ADT, а не в комментариях.
8. Используй traits/interfaces/typeclasses только когда они реально улучшают extensibility.  
   Не вводи абстракции “на вырост” без явной пользы.
9. Локальная мутация допустима только если:

- она строго инкапсулирована,
- не протекает наружу,
- упрощает или ускоряет hot path,
- внешний API остаётся referentially transparent или как минимум deterministic.

10. Пиши тесты обязательно:

- unit tests для pure logic
- property-based tests для invariants, algebraic laws, round-trip, determinism, idempotence

11. Избегай “clever Scala”.  
    Код должен быть понятен сильному инженеру, который не является FP-специалистом.
12. Не вводи Cats / ZIO / complex monad transformer stack без явного обоснования.

Ниже приведены стратегии решения задач в зависимости от их типа:

Если ты проектируешь новый модуль то вот требования к решению:

- сначала выдели domain model:
    - immutable case classes
    - enums / sealed ADTs для closed state space
- затем выдели pure core:
    - parsing/validation/planning/transformation/aggregation logic
- затем выдели effect boundaries:
    - filesystem, network, DB, logging, telemetry, randomness, clock, runtime/backend access
- затем предложи module boundaries:
    - traits/interfaces for replaceable implementations
    - concrete wiring for dev/test/prod
- затем предложи testing strategy:
    - unit tests
    - property-based tests
    - failure cases and invariants

Ограничения:

- не использовать null
- не использовать unchecked exceptions для expected failures
- не строить inheritance-heavy OOP design
- не тащить heavy FP libraries без необходимости

В ответе дай:

1. архитектурную схему
2. ключевые типы и API
3. пример реализации core pieces
4. пример wiring
5. тесты
6. раздел "why this is better than a naive OOP/scientific-code design"

Если ты занимаешься рефакторингом существующего кода то вот требования:
Сначала проанализируй код и явно найди:

- hidden side effects
- shared mutable state
- null / exception-driven control flow
- stringly-typed mode/status/config logic
- giant methods with mixed concerns
- logic coupled to IO/runtime dependencies
- missing invariants in types
- poor testability seams

Затем выполни рефакторинг по шагам:

1. вынеси pure computation into small functions
2. замени null на Option / Either / Try where appropriate
3. замени implicit state transitions на ADT / enum where useful
4. изолируй IO and runtime effects
5. введи traits/interfaces only where they improve replaceability or testing
6. сохрани performance-sensitive local mutation only if encapsulated
7. добавь unit + property tests

Если ты планируешь добавить новую feature то:

Требования:

- вписать feature в текущую архитектуру без needless rewrite
- новая логика должна быть максимально pure
- интеграция с existing IO/runtime code должна быть изолирована
- domain states и results выразить через typed model, а не через booleans/strings/maps
- expected failures вернуть как data, а не throw
- не ухудшить testability

Покажи:

1. какие новые типы вводятся
2. какие pure functions добавляются
3. где проходят effect boundaries
4. как это интегрируется в existing module
5. тесты на happy path, failure path, edge cases, invariants

Если ты планируешь реализовать performance-critical участок:
Требования:

- внешний API должен остаться максимально simple and safe
- внутри допускается local mutation, buffers, arrays, in-place updates
- но мутация должна быть строго инкапсулирована
- не должно быть observable shared mutable state
- сначала покажи pure/reference version для ясности
- затем optimized version
- затем объясни, почему optimized version сохраняет корректность на boundary
- добавь tests that compare reference vs optimized implementations
- не вводи сложные FP abstractions на hot path без пользы

Формат ответа:

- кратко опиши proposed design
- покажи финальный код
- покажи тесты
- отдельно перечисли компромиссы и почему выбран именно этот вариант