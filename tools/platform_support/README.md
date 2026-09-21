# Platform Support

Developer-only metadata verification, not part of the library runtime. Run from
the checkout with Python 3.10 or 3.11, `packaging`, and `tomli` (Python 3.10 only).

```
python -m tools.platform_support check --json
python -m tools.platform_support export --check
python -m tools.platform_support export --profile legacy
python -m tools.platform_support export --profile tensor
python -m tools.platform_support environment --profile legacy --json
python -m tools.platform_support environment --profile tensor --json
```

`--root PATH` selects a fixture or checkout; the default is the checkout containing
this package. Compatibility policy always comes from the package-local versioned
`compatibility.json`. Requirements are exported in the exact declared order and
spelling, including markers. `export --check` never writes.

Environment inspection reads distribution metadata only. Optional extras are
listed but not inferred. Select exactly one FEDOT profile for each environment:
``legacy`` uses the established InputData API, while ``tensor`` verifies the
TensorData integration target. The corresponding extras are mutually exclusive.
Missing packages, incompatible versions, unsupported Python, and unverified
direct sources produce a nonzero exit. FEDOT source verification requires PEP
610 repository and commit metadata, not a version match. Python 3.12 is
audit-only. Known constraints document each profile; dependency resolution is
checked separately.
