"""Explicit developer command, never part of ordinary library import."""
import argparse
import json

from .patches import apply_compatibility_patches, audit_compatibility_worktree


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check or explicitly patch a pinned FEDOT worktree")
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args(argv)
    if args.audit and args.execute:
        parser.error("--audit does not apply patches")
    result = (audit_compatibility_worktree(args.worktree) if args.audit
              else apply_compatibility_patches(args.worktree, execute=args.execute))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
