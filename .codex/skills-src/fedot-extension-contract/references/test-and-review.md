# Test and Review Guidance

## Core Test Targets

Prefer tests in `tests/unit/models` or the nearest benchmark-facing mirrored path for:

- valid manifest or preset acceptance
- duplicate rejection
- lookup behavior after registration
- manifest or preset discovery
- smoke-test behavior for valid, unavailable, and broken adapters
- parameter resolution from specs
- method or task routing from registered adapters

## Minimum Coverage for a Contract Change

When the contract or registry changes, keep coverage for:

- one happy path registration or config materialization
- one invalid manifest or preset path
- one duplicate or conflicting path
- one runtime-facing lookup or smoke-test path

## Review Questions

Ask:

1. Can a new adapter still be integrated through one obvious path?
2. Did the change preserve validated data structures as the source of truth?
3. Are contract failures explicit and typed enough to test?
4. Did runtime behavior start depending on raw manifest blobs or unvalidated adapter metadata again?
5. Are contract tests mirrored and focused on behavior rather than internals?

## Typical Findings

Common review findings in this area:

- runtime code re-parses fields that should already be validated
- duplicate detection is missing or split across layers
- smoke tests are weakened or bypassed
- manifest or preset evolution breaks old callers without need
