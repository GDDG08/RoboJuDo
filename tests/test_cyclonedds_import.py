import importlib

import pytest


def test_cyclonedds_import():
    pytest.importorskip("cyclonedds")
    assert importlib.import_module("cyclonedds") is not None
