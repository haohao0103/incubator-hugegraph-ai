"""Conftest for mocking hugegraph_llm dependencies in test environment."""

import sys
import types
import pytest


# ── Mock hugegraph_llm.utils.log (avoids pyhugegraph dependency) ──

class _MockLogger:
    """No-op logger for testing."""
    def debug(self, *a, **kw): pass
    def info(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
    def error(self, *a, **kw): pass


def _install_mocks():
    """Install mock modules before test collection."""
    log_mod = types.ModuleType("hugegraph_llm.utils.log")
    log_mod.log = _MockLogger()

    utils_mod = types.ModuleType("hugegraph_llm.utils")
    utils_mod.log = log_mod.log

    sys.modules["hugegraph_llm.utils.log"] = log_mod
    sys.modules["hugegraph_llm.utils"] = utils_mod


# Install at import time (before any test imports)
_install_mocks()
