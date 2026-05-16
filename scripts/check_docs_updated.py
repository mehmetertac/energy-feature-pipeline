#!/usr/bin/env python3
"""Pre-push guard: if package code changed on branch vs upstream, README or docs must change too."""

from __future__ import annotations

import subprocess
import sys


def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return None


def main() -> int:
    upstream = _git_output(["rev-parse", "--abbrev-ref", "@{upstream}"])
    if not upstream:
        print("check_docs_updated: no @{upstream}; skipping.")
        return 0

    merge_base = _git_output(["merge-base", "@{upstream}", "HEAD"])
    if not merge_base:
        print("check_docs_updated: could not resolve merge-base; skipping.")
        return 0

    changed = _git_output(["diff", "--name-only", f"{merge_base}...HEAD"])
    if not changed:
        return 0

    paths = [p for p in changed.splitlines() if p.strip()]
    pkg_changed = any(p.startswith("energy_features/") for p in paths)
    if not pkg_changed:
        return 0

    docs_ok = any(p.startswith("docs/") or p == "README.md" or p == "AGENTS.md" for p in paths)
    if docs_ok:
        return 0

    print(
        "Documentation rule: commits touching energy_features/ must also update "
        "README.md and/or docs/ (same push).\n"
        "Changed files:\n  " + "\n  ".join(paths),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
