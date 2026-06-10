# Example Prompts

## Route to `$fedot-pure-core-shell`

- "Vynesi decision rules iz `OKHSForecaster`, ne lomaya publichnyy API."
- "Razdeli benchmark orchestration i chistye aggregation rules v `benchmark/v2`."

## Route to `$fedot-safe-configs`

- "Sdelay manifests/presets parsing v `benchmark/v2` typed i bezopasnym."
- "Perenesi defaulting i normalization politik okna, q i representation v odin kanonicheskiy put'."

## Route to `$fedot-typed-domain-errors`

- "Zameni string flags i sentinel values v OKHS policy flow na yavnye typed rezul'taty."
- "Sdelay domain error model dlya vybora benchmark adapter i optional baseline."

## Route to `$fedot-extension-contract`

- "Dobav' optional deep baseline v `benchmark/v2`, ne razmazyvaya registratsiyu po neskol'kim mestam."
- "Peresoberi manifest/preset + registry + smoke-test flow dlya benchmark adapters."

## Route to `$fedot-invariant-tests-review`

- "Prover' etot refactoring PR na riski i nedostayushchie testy."
- "Dobav' tests dlya deterministic planner i normalization invariants v trajectory preprocessing."

## Composite Examples

- "Uberi sentinel `'None'` iz benchmark manifest parsing i dobav' coverage."
  Primary: `$fedot-safe-configs`
  Secondary: `$fedot-typed-domain-errors`, `$fedot-invariant-tests-review`

- "Vynesi typed planner iz benchmark facade i prover' regressii."
  Primary: `$fedot-pure-core-shell`
  Secondary: `$fedot-invariant-tests-review`
