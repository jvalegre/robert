"""Exercise the release checks against a real temporary Git history."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CI_DIRECTORY = REPOSITORY_ROOT / ".circleci"


def load_ci_module(monkeypatch, filename):
    monkeypatch.syspath_prepend(str(CI_DIRECTORY))
    script_path = CI_DIRECTORY / filename
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_git(repository_root, *args):
    return subprocess.check_output(
        ["git", *args], cwd=repository_root, text=True
    ).strip()


def create_repository(tmp_path):
    run_git(tmp_path, "init", "-q", "-b", "master")
    (tmp_path / "setup.py").write_text('version = "2.2.1"\n', encoding="utf-8")
    run_git(tmp_path, "add", "setup.py")
    run_git(
        tmp_path,
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=release@example.com",
        "commit",
        "-q",
        "-m",
        "Base version",
    )
    base_revision = run_git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "robert").mkdir()
    return base_revision


def write_version(repository_root, version):
    (repository_root / "robert" / "_version.py").write_text(
        f'ROBERT_VERSION = "{version}"\n', encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("final_version", "expected_result"),
    [("2.2.1", 1), ("2.2.0", 1), ("2.2.2", 0)],
)
def test_pr_requires_final_version_above_master(
    tmp_path, monkeypatch, final_version, expected_result
):
    base_revision = create_repository(tmp_path)
    write_version(tmp_path, final_version)
    checker = load_ci_module(monkeypatch, "check_pr_version.py")

    assert checker.main(tmp_path, base_revision) == expected_result


def test_pr_uses_master_base_after_multiple_branch_versions(tmp_path, monkeypatch):
    base_revision = create_repository(tmp_path)
    for version in ("2.2.2", "2.3.0"):
        write_version(tmp_path, version)
        run_git(tmp_path, "add", "robert/_version.py")
        run_git(
            tmp_path,
            "-c",
            "user.name=Release Test",
            "-c",
            "user.email=release@example.com",
            "commit",
            "-q",
            "-m",
            f"Version {version}",
        )
    (tmp_path / "README.md").write_text("Final change\n", encoding="utf-8")
    run_git(tmp_path, "add", "README.md")
    run_git(
        tmp_path,
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=release@example.com",
        "commit",
        "-q",
        "-m",
        "Final change",
    )
    checker = load_ci_module(monkeypatch, "check_pr_version.py")

    assert checker.main(tmp_path, base_revision) == 0


def test_pypi_release_uses_published_versions_without_git_baseline(
    tmp_path, monkeypatch
):
    (tmp_path / "robert").mkdir()
    write_version(tmp_path, "2.3.0")
    checker = load_ci_module(monkeypatch, "check_pypi_release.py")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"releases": {"2.2.1": []}}).encode("utf-8")

    monkeypatch.setattr(checker, "urlopen", lambda *_args, **_kwargs: Response())

    assert checker.main(tmp_path) == 0
