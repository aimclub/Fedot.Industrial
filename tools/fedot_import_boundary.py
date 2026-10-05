"""Reject removed FEDOT integration paths in executable Industrial code."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


FORBIDDEN_MODULES = frozenset({
    "fedot.core.data.array_utilities",
    "fedot.core.data.cv_folds",
    "fedot.core.data.data",
    "fedot.core.data.data_split",
    "fedot.core.data.multi_modal",
    "fedot_ind.core.repository.initializer_industrial_models",
    "fedot_ind.integration.fedot.legacy",
    "fedot_ind.integration.fedot.legacy_repository",
})
FORBIDDEN_CALLS = frozenset({
    "setup_repository",
    "setup_default_repository",
})
FORBIDDEN_REPOSITORY_MUTATIONS = frozenset({
    "assign_repo",
    "__repository_dict__",
})
SCAN_ROOTS = ("fedot_ind", "benchmark", "examples")


@dataclass(frozen=True, order=True)
class BoundaryViolation:
    path: str
    line: int
    kind: str
    symbol: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.kind}: {self.symbol}"


def scan_forbidden_usage(
        root: Path,
        scan_roots: Iterable[str] = SCAN_ROOTS,
) -> tuple[BoundaryViolation, ...]:
    violations: list[BoundaryViolation] = []
    for relative_root in scan_roots:
        directory = root / relative_root
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            violations.extend(_file_violations(relative, tree))
    return tuple(sorted(violations))


def _file_violations(path: str, tree: ast.AST) -> list[BoundaryViolation]:
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in FORBIDDEN_MODULES:
            violations.append(BoundaryViolation(path, node.lineno, "forbidden import", node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_MODULES:
                    violations.append(BoundaryViolation(path, node.lineno, "forbidden import", alias.name))
        elif isinstance(node, ast.Call):
            call_name = _attribute_name(node.func)
            if call_name in FORBIDDEN_CALLS:
                violations.append(BoundaryViolation(path, node.lineno, "removed repository call", call_name))
            if call_name in FORBIDDEN_REPOSITORY_MUTATIONS and not (
                    isinstance(node.func, ast.Attribute) and call_name == "__repository_dict__"):
                violations.append(BoundaryViolation(path, node.lineno, "repository mutation", call_name))
        elif isinstance(node, ast.Attribute) and node.attr == "__repository_dict__":
            violations.append(BoundaryViolation(path, node.lineno, "repository mutation", node.attr))
    return violations


def _attribute_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    violations = scan_forbidden_usage(args.root)
    for violation in violations:
        print(violation.render())
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main())
