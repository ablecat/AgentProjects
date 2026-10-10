from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from unittest import mock
from urllib.parse import urlsplit


def load_models_module():
    package = types.ModuleType("httpie")
    package.__path__ = []
    compat = types.ModuleType("httpie.compat")
    compat.urlsplit = urlsplit
    compat.str = str
    stubs = {"httpie": package, "httpie.compat": compat}
    source = Path(__file__).parents[1] / "httpie" / "models.py"
    spec = importlib.util.spec_from_file_location("httpie.models", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen HTTPie module")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module
