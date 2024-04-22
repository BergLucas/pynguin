#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
from unittest.mock import MagicMock

import pytest

from pynguin.ga.stoppingcondition import MaxStatementExecutionsStoppingCondition


@pytest.fixture()
def stopping_condition():
    return MaxStatementExecutionsStoppingCondition(10000)


def test_current_value(stopping_condition):
    assert stopping_condition.current_value() == 0


def test_current_value_reset(stopping_condition):
    result = MagicMock()
    stopping_condition.remote_observer.before_statement_execution(None, None, None)
    stopping_condition.remote_observer.after_test_case_execution(None, None, result)
    stopping_condition.after_remote_test_case_execution(None, result)
    stopping_condition.reset()
    assert stopping_condition.current_value() == 0


def test_before_search_start(stopping_condition):
    result = MagicMock()
    stopping_condition.remote_observer.before_statement_execution(None, None, None)
    stopping_condition.remote_observer.after_test_case_execution(None, None, result)
    stopping_condition.after_remote_test_case_execution(None, result)
    stopping_condition.before_search_start(None)
    assert stopping_condition.current_value() == 0


def test_set_get_limit(stopping_condition):
    stopping_condition.set_limit(42)
    assert stopping_condition.limit() == 42


def test_is_not_fulfilled(stopping_condition):
    assert not stopping_condition.is_fulfilled()


def test_is_fulfilled(stopping_condition):
    result = MagicMock()
    stopping_condition.set_limit(3)
    stopping_condition.remote_observer.before_statement_execution(None, None, None)
    stopping_condition.remote_observer.before_statement_execution(None, None, None)
    stopping_condition.remote_observer.before_statement_execution(None, None, None)
    stopping_condition.remote_observer.after_test_case_execution(None, None, result)
    stopping_condition.after_remote_test_case_execution(None, result)
    assert stopping_condition.is_fulfilled()
