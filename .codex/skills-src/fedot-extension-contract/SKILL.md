name: fedot-extension-contract
description: Design, validate, and refactor IndustrialTS integration contracts for external models, benchmark adapters, manifests, presets, registry-like discovery, parameter rules, and runtime adapters. Use when adding new baselines or optional backends, changing manifest shape, updating benchmark/v2 adapter flows, or reviewing PRs that touch manifest-driven runtime integration in benchmark/v2 or fedot_ind model wrappers.
---

# FEDOT Extension Contract

## Overview

Use this skill when the change concerns how IndustrialTS discovers, validates, registers, and executes external models or benchmark adapters. In this repository the main contract surface is not one `fedot/extensions` package; it is the typed benchmark and model-wrapper layer around `benchmark/v2`, optional baselines, manifests, presets, and adapter-facing runtime APIs.

Keep the integration surface explicit and typed so that new integrations follow one canonical flow: define the spec, validate or materialize it, smoke test or skip gracefully, then expose runtime behavior.

## Quick Start

1. Read [contract-map.md](references/contract-map.md) to anchor the canonical entities and current files.
2. Read [implementation-flow.md](references/implementation-flow.md) to preserve the integration lifecycle.
3. Read [test-and-review.md](references/test-and-review.md) before changing registry logic, parameter resolution, or runtime behavior.

## Workflow

### 1. Preserve the canonical contract

- Use repository-native typed specs such as `BenchmarkSuiteConfig`, `DatasetSpec`, `ModelSpec`, `RunSpec`, manifest payloads, preset records, and adapter metadata as the entry contract.
- Prefer one validated config path before runtime adapter instantiation.
- Use explicit result or error records for expected contract failures.

### 2. Keep one integration path

- The default path should remain `author preset or manifest -> validate/materialize -> instantiate adapters -> smoke test or skip optional dependencies -> runtime use`.
- Avoid introducing alternative side channels that bypass validation.
- Keep manifest or preset discovery separate from runtime instantiation.

### 3. Isolate pure rule layers

- Keep manifest validation, parameter resolution, and adapter selection in deterministic helpers.
- Keep module imports, registry mutation, and estimator instantiation at the shell boundary.
- Make runtime adapter code depend on validated specs rather than raw dictionaries.

### 4. Model failures explicitly

- Return structured contract errors for invalid manifests, duplicate model ids, unsupported adapter signatures, unavailable optional dependencies, or failed smoke tests.
- Keep duplicate detection, parameter validation, availability checks, and empty capability checks as explicit rules.
- Do not hide contract failures behind generic `Exception` messages when the failure is expected.

### 5. Protect integration ergonomics

- Keep the authoring path simple enough for new adapters or optional baselines.
- Avoid over-designing the contract when one new field or helper function would be enough.
- Prefer additive evolution of manifests and presets when backward compatibility matters.

## Strong Signals to Use This Skill

- A PR touches benchmark model adapters, preset or manifest loaders, runtime registry-like helpers, or optional external baseline integrations.
- A change adds or modifies adapter-facing manifest or preset schemas.
- An external model or benchmark adapter should be available without editing several unrelated internal registries.
- Runtime integration depends on validated capabilities, task support, hyperparameter schema, or manifest-driven operation resolution.

## Output Expectations

- Keep the integration contract explicit and discoverable.
- Preserve the canonical registration and smoke-test flow.
- Add or update mirrored tests under `tests/unit/models` or the closest benchmark-facing mirrored path.
- Keep runtime behavior downstream of a validated manifest or preset.

## References

- Read [contract-map.md](references/contract-map.md) for current entities and files.
- Read [implementation-flow.md](references/implementation-flow.md) for the canonical lifecycle.
- Read [test-and-review.md](references/test-and-review.md) for adapter-specific coverage and review rules.
