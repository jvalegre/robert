from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

from robert.report_utils import get_repro_ml_stack_versions


def test_get_repro_ml_stack_versions_skips_missing_packages():
    def fake_version(pkg):
        if pkg == "numpy":
            return "2.3.4"
        raise PackageNotFoundError(pkg)

    with patch("importlib.metadata.version", side_effect=fake_version):
        versions = get_repro_ml_stack_versions()

    assert versions == [("numpy", "numpy", "2.3.4")]
