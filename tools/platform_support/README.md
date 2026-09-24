# Platform support checks

These developer tools validate the single supported FEDOT integration contract.
Run them from the repository checkout with Python 3.10 or 3.11:

```text
python -m tools.platform_support check --json
python -m tools.platform_support export --check
python -m tools.platform_support export
python -m tools.platform_support environment --json
```

`compatibility.json` is the versioned policy. It declares one `current` FEDOT
profile pinned to a full commit SHA, the Python support matrix, and known
dependency constraints. FEDOT must appear once as an unconditional base
dependency and must not be repeated in an optional dependency group.

`requirements.txt` is generated from the base dependencies in their declared
order. `export --check` is read-only. Environment inspection uses distribution
metadata and PEP 610 source information; an installed version number alone does
not prove the expected FEDOT revision. Python 3.12 remains audit-only.
