"""Every module must import without the heavy ML deps installed.

This guards the lazy-import design in agent.engine: importing the package (and
running the tool loop) must not require torch/transformers.
"""
import importlib

import pytest

MODULES = [
    "agent",
    "agent.engine",
    "agent.ollama_engine",
    "agent.finetune",
    "agent.webui",
    "agent.loop",
    "agent.cli",
    "agent.memory",
    "agent.tools",
    "agent.tools.calculator",
    "agent.tools.files",
    "agent.tools.shell",
]


@pytest.mark.parametrize("mod", MODULES)
def test_module_imports(mod):
    importlib.import_module(mod)
