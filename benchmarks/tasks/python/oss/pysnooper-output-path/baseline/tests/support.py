from __future__ import annotations

import abc
import importlib.util
import io
import os
from pathlib import Path
import sys
import types
from unittest import mock


def load_pysnooper_module():
    package = types.ModuleType("pysnooper")
    package.__path__ = []

    class WritableStream(metaclass=abc.ABCMeta):
        pass

    WritableStream.register(io.StringIO)
    utils = types.ModuleType("pysnooper.utils")
    utils.WritableStream = WritableStream
    pycompat = types.ModuleType("pysnooper.pycompat")
    pycompat.PathLike = os.PathLike
    tracer = types.ModuleType("pysnooper.tracer")
    tracer.Tracer = object
    decorator = types.ModuleType("decorator")
    decorator.decorator = lambda function: function
    stubs = {
        "pysnooper": package,
        "pysnooper.utils": utils,
        "pysnooper.pycompat": pycompat,
        "pysnooper.tracer": tracer,
        "decorator": decorator,
    }
    source = Path(__file__).parents[1] / "pysnooper" / "pysnooper.py"
    spec = importlib.util.spec_from_file_location("pysnooper.pysnooper", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen PySnooper module")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module
