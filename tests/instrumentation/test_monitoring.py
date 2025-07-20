#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2025 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
import importlib

from unittest.mock import MagicMock
from unittest.mock import call

import pytest

from pynguin.instrumentation.monitoring import LineCoverageMonitoringInstrumentation
from pynguin.instrumentation.monitoring import MonitoringInstrumentationTransformer
from pynguin.instrumentation.tracer import InstrumentationExecutionTracer


@pytest.fixture
def simple_module():
    simple = importlib.import_module("tests.fixtures.instrumentation.simple")
    return importlib.reload(simple)


@pytest.fixture
def tracer_mock():
    tracer = MagicMock()
    tracer.register_line.side_effect = range(100)
    return tracer


def test_line_coverage(simple_module, tracer_mock):
    instrumentation_tracer = InstrumentationExecutionTracer(tracer_mock)
    adapter = LineCoverageMonitoringInstrumentation(instrumentation_tracer)
    transformer = MonitoringInstrumentationTransformer(instrumentation_tracer, [adapter])
    transformer.instrument_module(simple_module.simple_function.__code__)
    simple_module.simple_function(1)
    assert len(tracer_mock.register_line.mock_calls) == 2
    assert tracer_mock.track_line_visit.mock_calls == [call(1)]
