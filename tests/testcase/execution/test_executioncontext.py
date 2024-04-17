#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
from unittest.mock import MagicMock

import pytest

import pynguin.testcase.variablereference as vr
import pynguin.utils.generic.genericaccessibleobject as gao

from pynguin.testcase.execution import ExecutionContext
from pynguin.testcase.execution import ModuleProvider
from pynguin.testcase.statement_to_ast import BUILTIN_TRANSFORMER_FUNCTIONS
from pynguin.testcase.statement_to_ast import StatementToAstTransformer


def test_get_reference_value():
    statement_transformer = StatementToAstTransformer(BUILTIN_TRANSFORMER_FUNCTIONS)
    ctx = ExecutionContext(statement_transformer, ModuleProvider())
    ref = vr.VariableReference(MagicMock(), int)
    with pytest.raises(ValueError):  # noqa: PT011
        ctx.get_reference_value(ref)


def test_get_reference_value_2():
    statement_transformer = StatementToAstTransformer(BUILTIN_TRANSFORMER_FUNCTIONS)
    ctx = ExecutionContext(statement_transformer, ModuleProvider())
    module_mock = MagicMock(foo=MagicMock(bar=5))
    ref = vr.FieldReference(
        vr.StaticModuleFieldReference(gao.GenericStaticModuleField("sys", "foo", int)),
        gao.GenericField(MagicMock, "bar", int),
    )
    ctx._global_namespace = {ctx._module_aliases.get_name("sys"): module_mock}
    assert ctx.get_reference_value(ref) == 5


def test_get_reference_value_3(test_case_mock):
    statement_transformer = StatementToAstTransformer(BUILTIN_TRANSFORMER_FUNCTIONS)
    ctx = ExecutionContext(statement_transformer, ModuleProvider())
    var_mock = MagicMock(foo=MagicMock(bar=5))
    var = vr.VariableReference(test_case_mock, int)
    ref = vr.FieldReference(
        vr.FieldReference(var, gao.GenericField(MagicMock, "foo", int)),
        gao.GenericField(MagicMock, "bar", int),
    )
    ctx._local_namespace = {ctx._variable_names.get_name(var): var_mock}
    assert ctx.get_reference_value(ref) == 5
