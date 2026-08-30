"""Foundation smoke tests; application behavior is intentionally not present yet."""

from theme_leadership_os import __version__


def test_package_version_is_present() -> None:
    assert __version__ == "0.1.0"
