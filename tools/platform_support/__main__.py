"""Minimal developer CLI for metadata checks, export, and environment reports."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .environment import current_environment
from .loading import default_root, load_compatibility, load_project, load_requirements
from .schema import Issue
from .verification import render_requirements, verify_project


def _report(issues: list[Issue], **details) -> dict:
    return {"ok": not issues, **details, "issues": [asdict(issue) for issue in issues]}


def _emit(result: dict, as_json: bool) -> int:
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print("OK" if result["ok"] else "FAILED")
        if "python" in result:
            python = result["python"]
            print(f"Python {python['version']}: {python['status']} ({python['reason']})")
        for dependency in result.get("dependencies", []):
            print(f"{dependency['name']}: {dependency['status']} {dependency.get('version', '')}".rstrip())
        for issue in result["issues"]:
            print(f"{issue['field']}: [{issue['code']}] {issue['message']}")
    return 0 if result["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.platform_support")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "export", "environment"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--root", type=Path, default=default_root())
        if command == "export":
            subparser.add_argument("--check", action="store_true")
        else:
            subparser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    project = load_project(args.root)
    issues = list(project.issues)
    if args.command == "export":
        if project.ok:
            expected = render_requirements(project.value)
            if args.check:
                existing = load_requirements(args.root)
                issues.extend(existing.issues)
                if existing.ok and existing.value != expected:
                    issues.append(Issue("requirements.txt", "export", "Requirements are stale; regenerate explicitly."))
            else:
                try:
                    with (args.root / "requirements.txt").open("w", encoding="utf-8", newline="\n") as stream:
                        stream.write(expected)
                except (OSError, UnicodeError) as error:
                    issues.append(Issue("requirements.txt", "write", str(error)))
        return _emit(_report(issues), False)
    compatibility = load_compatibility()
    issues.extend(compatibility.issues)
    if args.command == "check":
        requirements = load_requirements(args.root)
        issues.extend(requirements.issues)
        if project.ok and compatibility.ok and requirements.ok:
            issues.extend(verify_project(project.value, compatibility.value, requirements.value))
        return _emit(_report(issues), args.json)
    if issues:
        return _emit(_report(issues), args.json)
    issues.extend(verify_project(project.value, compatibility.value, render_requirements(project.value)))
    if issues:
        return _emit(_report(issues), args.json)
    return _emit(current_environment(project.value, compatibility.value), args.json)


if __name__ == "__main__":
    raise SystemExit(main())
