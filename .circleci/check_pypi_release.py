"""Require a new ROBERT version before publishing to PyPI."""

import json
import runpy
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from packaging.version import Version


PROJECT_NAME = "robert"


def main(repository_root=None):
    repository_root = (
        Path(repository_root) if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    version_file = repository_root / "robert" / "_version.py"
    try:
        source_version = Version(runpy.run_path(str(version_file))["ROBERT_VERSION"])
        request = Request(
            f"https://pypi.org/pypi/{PROJECT_NAME}/json",
            headers={"Accept": "application/json", "User-Agent": "robert-release-ci"},
        )
        with urlopen(request, timeout=30) as response:
            published_versions = {
                Version(value) for value in json.load(response)["releases"]
            }
    except (URLError, KeyError, OSError, ValueError) as error:
        print(f"Cannot verify the ROBERT version against PyPI: {error}", file=sys.stderr)
        return 1

    if source_version in published_versions:
        print(
            f"ROBERT {source_version} is already on PyPI. Increase ROBERT_VERSION "
            "in robert/_version.py before publishing.",
            file=sys.stderr,
        )
        return 1

    latest_published_version = max(published_versions, default=None)
    if latest_published_version is not None and source_version < latest_published_version:
        print(
            f"ROBERT {source_version} is older than the latest PyPI release "
            f"({latest_published_version}). Increase ROBERT_VERSION before publishing.",
            file=sys.stderr,
        )
        return 1

    print(f"ROBERT {source_version} is new; publication may proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
