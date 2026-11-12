"""Require a pull request to raise ROBERT's version above master."""

import ast
import runpy
import subprocess
import sys
from pathlib import Path

from packaging.version import Version


def version_at_revision(repository_root, revision):
    """Read a version from Git, including the legacy setup.py location."""
    for path, variable_name in (
        ("robert/_version.py", "ROBERT_VERSION"),
        ("setup.py", "version"),
    ):
        result = subprocess.run(
            ["git", "show", f"{revision}:{path}"],
            cwd=repository_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            continue
        for statement in ast.parse(result.stdout).body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == variable_name
                for target in statement.targets
            ):
                value = ast.literal_eval(statement.value)
                if isinstance(value, str):
                    return Version(value)
        raise ValueError(f"Cannot read {variable_name} at {revision}:{path}")
    raise ValueError(f"Cannot find a ROBERT version at revision {revision}")


def main(repository_root=None, base_revision="origin/master"):
    repository_root = (
        Path(repository_root) if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    version_file = repository_root / "robert" / "_version.py"
    try:
        pr_version = Version(runpy.run_path(str(version_file))["ROBERT_VERSION"])
        master_version = version_at_revision(repository_root, base_revision)
    except (KeyError, OSError, SyntaxError, ValueError) as error:
        print(f"Cannot verify the ROBERT version for this PR: {error}", file=sys.stderr)
        return 1

    if pr_version <= master_version:
        print(
            f"ROBERT_VERSION must increase before merging into master "
            f"({master_version} -> {pr_version}). Update robert/_version.py.",
            file=sys.stderr,
        )
        return 1

    print(f"ROBERT_VERSION increases from {master_version} to {pr_version}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
