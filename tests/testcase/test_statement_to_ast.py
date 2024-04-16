#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
import ast
import inspect

from unittest.mock import MagicMock

import pytest

import pynguin.testcase.statement as stmt
import pynguin.testcase.variablereference as vr
import pynguin.utils.generic.genericaccessibleobject as gao

from pynguin.analyses.typesystem import InferredSignature
from pynguin.testcase.statement_to_ast import transform_assignment_statement
from pynguin.testcase.statement_to_ast import transform_class_primitive_statement
from pynguin.testcase.statement_to_ast import transform_constructor_statement
from pynguin.testcase.statement_to_ast import transform_dict_statement
from pynguin.testcase.statement_to_ast import transform_enum_statement
from pynguin.testcase.statement_to_ast import transform_field_statement
from pynguin.testcase.statement_to_ast import transform_function_statement
from pynguin.testcase.statement_to_ast import transform_list_statement
from pynguin.testcase.statement_to_ast import transform_method_statement
from pynguin.testcase.statement_to_ast import transform_primitive_statement
from pynguin.testcase.statement_to_ast import transform_set_statement
from pynguin.testcase.statement_to_ast import transform_tuple_statement
from pynguin.utils.generic.genericaccessibleobject import GenericConstructor
from pynguin.utils.generic.genericaccessibleobject import GenericFunction
from pynguin.utils.generic.genericaccessibleobject import GenericMethod
from pynguin.utils.namingscope import NamingScope


@pytest.fixture()
def var_names() -> NamingScope:
    return NamingScope()


@pytest.fixture()
def module_aliases() -> NamingScope:
    return NamingScope(prefix="module")


def __create_source_from_ast(module_body: ast.stmt) -> str:
    return ast.unparse(
        ast.fix_missing_locations(ast.Module(body=[module_body], type_ignores=[]))
    )


def test_statement_to_ast_int(module_aliases, var_names, default_test_case):
    int_stmt = stmt.IntPrimitiveStatement(default_test_case, 5)
    ast_node = transform_primitive_statement(int_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == "var_0 = 5"


def test_statement_to_ast_float(module_aliases, var_names, default_test_case):
    float_stmt = stmt.FloatPrimitiveStatement(default_test_case, 5.5)
    ast_node = transform_primitive_statement(
        float_stmt, module_aliases, var_names, True
    )
    assert __create_source_from_ast(ast_node) == "var_0 = 5.5"


def test_statement_to_ast_str(module_aliases, var_names, default_test_case):
    str_stmt = stmt.StringPrimitiveStatement(default_test_case, "TestMe")
    ast_node = transform_primitive_statement(str_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == "var_0 = 'TestMe'"


def test_statement_to_ast_bytes(module_aliases, var_names, default_test_case):
    bytes_stmt = stmt.BytesPrimitiveStatement(default_test_case, b"TestMe")
    ast_node = transform_primitive_statement(
        bytes_stmt, module_aliases, var_names, True
    )
    assert __create_source_from_ast(ast_node) == "var_0 = b'TestMe'"


def test_statement_to_ast_bool(module_aliases, var_names, default_test_case):
    bool_stmt = stmt.BooleanPrimitiveStatement(default_test_case, True)  # noqa: FBT003
    ast_node = transform_primitive_statement(bool_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == "var_0 = True"


def test_statement_to_ast_class(module_aliases, var_names, default_test_case):
    class_stmt = stmt.ClassPrimitiveStatement(default_test_case, 0)
    ast_node = transform_class_primitive_statement(
        class_stmt, module_aliases, var_names, True
    )
    assert __create_source_from_ast(ast_node) == "var_0 = module_0.int"


def test_statement_to_ast_none(module_aliases, var_names, default_test_case):
    none_stmt = stmt.NoneStatement(default_test_case)
    ast_node = transform_primitive_statement(none_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == "var_0 = None"


def test_statement_to_ast_enum(module_aliases, var_names, default_test_case):
    enum_stmt = stmt.EnumPrimitiveStatement(
        default_test_case,
        MagicMock(
            owner=default_test_case.test_cluster.type_system.to_type_info(MagicMock),
            names=["BAR"],
        ),
        0,
    )
    ast_node = transform_enum_statement(enum_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == "var_0 = module_0.MagicMock.BAR"


def test_statement_to_ast_assignment(module_aliases, var_names, default_test_case):
    string = stmt.StringPrimitiveStatement(default_test_case, "foo")
    field = gao.GenericField(
        default_test_case.test_cluster.type_system.to_type_info(str),
        "foo",
        default_test_case.test_cluster.type_system.convert_type_hint(None),
    )
    int_0 = stmt.IntPrimitiveStatement(default_test_case, 42)
    assign_stmt = stmt.AssignmentStatement(
        default_test_case, vr.FieldReference(string.ret_val, field), int_0.ret_val
    )
    ast_node = transform_assignment_statement(
        assign_stmt, module_aliases, var_names, True
    )
    assert __create_source_from_ast(ast_node) == "var_0.foo = var_1"


def test_statement_to_ast_field(module_aliases, var_names, default_test_case):
    string = stmt.StringPrimitiveStatement(default_test_case, "foo")
    field = gao.GenericField(
        default_test_case.test_cluster.type_system.to_type_info(str),
        "foo",
        default_test_case.test_cluster.type_system.convert_type_hint(None),
    )
    field_stmt = stmt.FieldStatement(default_test_case, field, string.ret_val)
    ast_node = transform_field_statement(field_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == "var_0 = var_1.foo"


def all_param_types_signature(type_system):
    return InferredSignature(
        signature=inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name="a",
                    kind=inspect.Parameter.POSITIONAL_ONLY,
                    annotation=float,
                ),
                inspect.Parameter(
                    name="b",
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=float,
                ),
                inspect.Parameter(
                    name="c",
                    kind=inspect.Parameter.VAR_POSITIONAL,
                    annotation=float,
                ),
                inspect.Parameter(
                    name="d",
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    annotation=float,
                ),
                inspect.Parameter(
                    name="e",
                    kind=inspect.Parameter.VAR_KEYWORD,
                    annotation=float,
                ),
            ]
        ),
        original_return_type=type_system.convert_type_hint(float),
        original_parameters={
            "a": type_system.convert_type_hint(float),
            "b": type_system.convert_type_hint(float),
            "c": type_system.convert_type_hint(float),
            "d": type_system.convert_type_hint(float),
            "e": type_system.convert_type_hint(float),
        },
        type_system=type_system,
    )


def default_args_signature(type_system):
    return InferredSignature(
        signature=inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name="a",
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=float,
                    default=0.5,
                ),
                inspect.Parameter(
                    name="b",
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=float,
                    default=0.42,
                ),
                inspect.Parameter(
                    name="c",
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    annotation=float,
                    default=0.42,
                ),
            ]
        ),
        original_return_type=type_system.convert_type_hint(float),
        original_parameters={
            "a": type_system.convert_type_hint(float),
            "b": type_system.convert_type_hint(float),
            "c": type_system.convert_type_hint(float),
        },
        type_system=type_system,
    )


def no_default_args_signature(type_system):
    return InferredSignature(
        signature=inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name="a",
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=float,
                ),
                inspect.Parameter(
                    name="b",
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=float,
                ),
                inspect.Parameter(
                    name="c",
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    annotation=float,
                    default=0.42,
                ),
            ]
        ),
        original_return_type=type_system.convert_type_hint(float),
        original_parameters={
            "a": type_system.convert_type_hint(float),
            "b": type_system.convert_type_hint(float),
            "c": type_system.convert_type_hint(float),
        },
        type_system=type_system,
    )


@pytest.fixture()
def all_types_constructor(type_system):
    return GenericConstructor(
        owner=type_system.to_type_info(MagicMock),
        inferred_signature=all_param_types_signature(type_system),
    )


@pytest.fixture()
def all_types_method(type_system):
    return GenericMethod(
        owner=MagicMock(),
        inferred_signature=all_param_types_signature(type_system),
        method=MagicMock(__name__="method"),
    )


@pytest.fixture()
def all_types_function(type_system):
    return GenericFunction(
        function=MagicMock(__name__="function"),
        inferred_signature=all_param_types_signature(type_system),
    )


@pytest.fixture()
def default_args_function(type_system):
    return GenericFunction(
        function=MagicMock(__name__="function"),
        inferred_signature=default_args_signature(type_system),
    )


@pytest.fixture()
def no_default_args_function(type_system):
    return GenericFunction(
        function=MagicMock(__name__="function"),
        inferred_signature=no_default_args_signature(type_system),
    )


@pytest.mark.parametrize(
    "args,expected",
    [
        ({}, "var_0 = module_0.MagicMock()"),
        (
            {"a"},
            "var_1 = module_0.MagicMock(var_0)",
        ),
        (
            {"b"},
            "var_1 = module_0.MagicMock(var_0)",
        ),
        (
            {"c"},
            "var_1 = module_0.MagicMock(*var_0)",
        ),
        (
            {"d"},
            "var_1 = module_0.MagicMock(d=var_0)",
        ),
        (
            {"e"},
            "var_1 = module_0.MagicMock(**var_0)",
        ),
        (
            {
                "a",
                "b",
                "c",
                "d",
                "e",
            },
            "var_5 = module_0.MagicMock(var_0, var_1, *var_2, d=var_3, **var_4)",
        ),
    ],
)
def test_statement_to_ast_constructor_args(
    module_aliases, var_names, default_test_case, all_types_constructor, args, expected
):
    args_stmts = {
        a: stmt.IntPrimitiveStatement(default_test_case, 3).ret_val for a in args
    }
    constr_stmt = stmt.ConstructorStatement(
        default_test_case, all_types_constructor, args_stmts
    )
    ast_node = transform_constructor_statement(
        constr_stmt, module_aliases, var_names, True
    )
    assert __create_source_from_ast(ast_node) == expected


def test_statement_to_ast_constructor_no_store(
    module_aliases, var_names, test_case_mock, all_types_constructor
):
    constr_stmt = stmt.ConstructorStatement(test_case_mock, all_types_constructor, {})
    ast_node = transform_constructor_statement(
        constr_stmt, module_aliases, var_names, False
    )
    assert __create_source_from_ast(ast_node) == "module_0.MagicMock()"


@pytest.mark.parametrize(
    "args,expected",
    [
        ({}, "var_1 = var_0.method()"),
        (
            {"a": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val},
            "var_2 = var_1.method(var_0)",
        ),
        (
            {"b": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val},
            "var_2 = var_1.method(var_0)",
        ),
        (
            {"c": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val},
            "var_2 = var_1.method(*var_0)",
        ),
        (
            {"d": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val},
            "var_2 = var_1.method(d=var_0)",
        ),
        (
            {"e": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val},
            "var_2 = var_1.method(**var_0)",
        ),
        (
            {
                "a": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val,
                "b": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val,
                "c": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val,
                "d": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val,
                "e": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val,
            },
            "var_6 = var_5.method(var_0, var_1, *var_2, d=var_3, **var_4)",
        ),
    ],
)
def test_statement_to_ast_method_args(
    module_aliases, var_names, test_case_mock, all_types_method, args, expected
):
    method_stmt = stmt.MethodStatement(
        test_case_mock,
        all_types_method,
        stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val,
        args,
    )
    ast_node = transform_method_statement(method_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == expected


def test_statement_to_ast_method_no_store(
    module_aliases, var_names, test_case_mock, all_types_method
):
    method_stmt = stmt.MethodStatement(
        test_case_mock,
        all_types_method,
        stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val,
        {},
    )
    ast_node = transform_method_statement(method_stmt, module_aliases, var_names, False)
    assert __create_source_from_ast(ast_node) == "var_0.method()"


@pytest.mark.parametrize(
    "args,expected",
    [
        ({}, "var_0 = module_0.function()"),
        (
            {"a": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val},
            "var_1 = module_0.function(var_0)",
        ),
        (
            {"b": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val},
            "var_1 = module_0.function(var_0)",
        ),
        (
            {"c": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val},
            "var_1 = module_0.function(*var_0)",
        ),
        (
            {"d": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val},
            "var_1 = module_0.function(d=var_0)",
        ),
        (
            {"e": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val},
            "var_1 = module_0.function(**var_0)",
        ),
        (
            {
                "a": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val,
                "b": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val,
                "c": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val,
                "d": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val,
                "e": stmt.IntPrimitiveStatement(MagicMock(), 3).ret_val,
            },
            "var_5 = module_0.function(var_0, var_1, *var_2, d=var_3, **var_4)",
        ),
    ],
)
def test_statement_to_ast_function_args(
    module_aliases, var_names, test_case_mock, all_types_function, args, expected
):
    func_stmt = stmt.FunctionStatement(test_case_mock, all_types_function, args)
    ast_node = transform_function_statement(func_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == expected


@pytest.mark.parametrize(
    "args,expected",
    [
        ({}, "var_0 = module_0.function()"),
        (
            {"a": stmt.FloatPrimitiveStatement(MagicMock(), 3.0).ret_val},
            "var_1 = module_0.function(var_0)",
        ),
        (
            {"b": stmt.FloatPrimitiveStatement(MagicMock(), 3.0).ret_val},
            "var_1 = module_0.function(b=var_0)",
        ),
        (
            {
                "a": stmt.FloatPrimitiveStatement(MagicMock(), 3.0).ret_val,
                "b": stmt.FloatPrimitiveStatement(MagicMock(), 3.0).ret_val,
            },
            "var_2 = module_0.function(var_0, var_1)",
        ),
        (
            {"c": stmt.FloatPrimitiveStatement(MagicMock(), 3.0).ret_val},
            "var_1 = module_0.function(c=var_0)",
        ),
        (
            {
                "a": stmt.FloatPrimitiveStatement(MagicMock(), 3.0).ret_val,
                "c": stmt.FloatPrimitiveStatement(MagicMock(), 3.0).ret_val,
            },
            "var_2 = module_0.function(var_0, c=var_1)",
        ),
        (
            {
                "a": stmt.FloatPrimitiveStatement(MagicMock(), 3.0).ret_val,
                "b": stmt.FloatPrimitiveStatement(MagicMock(), 3.0).ret_val,
                "c": stmt.FloatPrimitiveStatement(MagicMock(), 3.0).ret_val,
            },
            "var_3 = module_0.function(var_0, var_1, c=var_2)",
        ),
    ],
)
def test_statement_to_ast_function_default_args(
    module_aliases, var_names, test_case_mock, default_args_function, args, expected
):
    func_stmt = stmt.FunctionStatement(test_case_mock, default_args_function, args)
    ast_node = transform_function_statement(func_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == expected


@pytest.mark.parametrize(
    "args,expected",
    [
        (
            {
                "a": stmt.FloatPrimitiveStatement(MagicMock(), 3.0).ret_val,
                "b": stmt.FloatPrimitiveStatement(MagicMock(), 3.0).ret_val,
            },
            "var_2 = module_0.function(var_0, var_1)",
        ),
        (
            {
                "a": stmt.FloatPrimitiveStatement(MagicMock(), 3.0).ret_val,
                "b": stmt.FloatPrimitiveStatement(MagicMock(), 3.0).ret_val,
                "c": stmt.FloatPrimitiveStatement(MagicMock(), 3.0).ret_val,
            },
            "var_3 = module_0.function(var_0, var_1, c=var_2)",
        ),
    ],
)
def test_statement_to_ast_function_no_default_args(
    module_aliases, var_names, test_case_mock, no_default_args_function, args, expected
):
    func_stmt = stmt.FunctionStatement(test_case_mock, no_default_args_function, args)
    ast_node = transform_function_statement(func_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == expected


def test_statement_to_ast_function_no_store(
    module_aliases, var_names, test_case_mock, all_types_function
):
    func_stmt = stmt.FunctionStatement(test_case_mock, all_types_function, {})
    ast_node = transform_function_statement(func_stmt, module_aliases, var_names, False)
    assert __create_source_from_ast(ast_node) == "module_0.function()"


def test_statement_to_ast_list_single(module_aliases, var_names, default_test_case):
    list_stmt = stmt.ListStatement(
        default_test_case,
        default_test_case.test_cluster.type_system.convert_type_hint(list[int]),
        [stmt.IntPrimitiveStatement(default_test_case, 5).ret_val],
    )
    ast_node = transform_list_statement(list_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == "var_0 = [var_1]"


def test_statement_to_ast_list_empty(module_aliases, var_names, default_test_case):
    list_stmt = stmt.ListStatement(
        default_test_case,
        default_test_case.test_cluster.type_system.convert_type_hint(list[int]),
        [],
    )
    ast_node = transform_list_statement(list_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == "var_0 = []"


def test_statement_to_ast_set_single(module_aliases, var_names, default_test_case):
    set_stmt = stmt.SetStatement(
        default_test_case,
        default_test_case.test_cluster.type_system.convert_type_hint(set[int]),
        [stmt.IntPrimitiveStatement(default_test_case, 5).ret_val],
    )
    ast_node = transform_set_statement(set_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == "var_1 = {var_0}"


def test_statement_to_ast_set_empty(module_aliases, var_names, default_test_case):
    set_stmt = stmt.SetStatement(
        default_test_case,
        default_test_case.test_cluster.type_system.convert_type_hint(set[int]),
        [],
    )
    ast_node = transform_set_statement(set_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == "var_0 = set()"


def test_statement_to_ast_tuple_single(module_aliases, var_names, default_test_case):
    tuple_stmt = stmt.TupleStatement(
        default_test_case,
        default_test_case.test_cluster.type_system.convert_type_hint(tuple[int]),
        [stmt.IntPrimitiveStatement(default_test_case, 5).ret_val],
    )
    ast_node = transform_tuple_statement(tuple_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == "var_0 = (var_1,)"


def test_statement_to_ast_tuple_empty(module_aliases, var_names, default_test_case):
    tuple_stmt = stmt.TupleStatement(
        default_test_case,
        default_test_case.test_cluster.type_system.convert_type_hint(tuple),
        [],
    )
    ast_node = transform_tuple_statement(tuple_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == "var_0 = ()"


def test_statement_to_ast_dict_single(module_aliases, var_names, default_test_case):
    dict_stmt = stmt.DictStatement(
        default_test_case,
        default_test_case.test_cluster.type_system.convert_type_hint(dict[int, int]),
        [
            (
                stmt.IntPrimitiveStatement(default_test_case, 5).ret_val,
                stmt.IntPrimitiveStatement(default_test_case, 5).ret_val,
            )
        ],
    )
    ast_node = transform_dict_statement(dict_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == "var_0 = {var_1: var_2}"


def test_statement_to_ast_dict_empty(module_aliases, var_names, default_test_case):
    dict_stmt = stmt.DictStatement(
        default_test_case,
        default_test_case.test_cluster.type_system.convert_type_hint(dict[int, int]),
        [],
    )
    ast_node = transform_dict_statement(dict_stmt, module_aliases, var_names, True)
    assert __create_source_from_ast(ast_node) == "var_0 = {}"
