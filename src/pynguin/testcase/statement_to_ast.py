#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides a visitor that transforms statements to AST."""
from __future__ import annotations

import ast

from abc import abstractmethod
from inspect import Parameter
from typing import TYPE_CHECKING
from typing import Protocol
from typing import TypeVar
from typing import cast

import pynguin.utils.ast_util as au

from pynguin.testcase.statement import AssignmentStatement
from pynguin.testcase.statement import BooleanPrimitiveStatement
from pynguin.testcase.statement import BytesPrimitiveStatement
from pynguin.testcase.statement import ClassPrimitiveStatement
from pynguin.testcase.statement import ComplexPrimitiveStatement
from pynguin.testcase.statement import ConstructorStatement
from pynguin.testcase.statement import DictStatement
from pynguin.testcase.statement import EnumPrimitiveStatement
from pynguin.testcase.statement import FieldStatement
from pynguin.testcase.statement import FloatPrimitiveStatement
from pynguin.testcase.statement import FunctionStatement
from pynguin.testcase.statement import IntPrimitiveStatement
from pynguin.testcase.statement import ListStatement
from pynguin.testcase.statement import MethodStatement
from pynguin.testcase.statement import NoneStatement
from pynguin.testcase.statement import ParametrizedStatement
from pynguin.testcase.statement import SetStatement
from pynguin.testcase.statement import Statement
from pynguin.testcase.statement import StringPrimitiveStatement
from pynguin.testcase.statement import TupleStatement
from pynguin.utils.generic.genericaccessibleobject import (
    GenericCallableAccessibleObject,
)


if TYPE_CHECKING:
    import pynguin.utils.namingscope as ns


S_contra = TypeVar("S_contra", bound=Statement, contravariant=True)


class StatementToAstTransformer:
    """Transforms internal statements to AST nodes."""

    def __init__(
        self,
        transformer_functions: dict[type, StatementToAstTransformerFunction],
    ) -> None:
        """Create a new statement to AST manager.

        Args:
            transformer_functions: A dictionary that maps statement types to
                transformer functions.
        """
        self._transformer_functions = transformer_functions

    def transform(
        self,
        stmt: Statement,
        module_aliases: ns.AbstractNamingScope,
        variable_names: ns.AbstractNamingScope,
        store_call_return: bool = True,  # noqa: FBT001, FBT002
    ) -> ast.stmt:
        """Transforms a statement to an AST node.

        Args:
            stmt: The statement to transform.
            module_aliases: A naming scope for module alias names.
            variable_names: A naming scope for variable names.
            store_call_return: Should the result of a call be stored in a variable?
                For example, if we know that a call raises an exception, then we don't
                need to assign the result to a variable, as it will never be assigned
                anyway.

        Returns:
            The AST node.
        """
        try:
            transformer_function = self._transformer_functions[type(stmt)]
        except KeyError as e:
            raise NotImplementedError(f"Unknown statement type: {type(stmt)}") from e

        return transformer_function(
            stmt,
            module_aliases,
            variable_names,
            store_call_return,
        )


def create_args(
    stmt: ParametrizedStatement,
    module_aliases: ns.AbstractNamingScope,
    variable_names: ns.AbstractNamingScope,
) -> tuple[list[ast.expr], list[ast.keyword]]:
    """Creates the AST nodes for arguments.

    Creates the positional arguments, i.e., POSITIONAL_ONLY,
    POSITIONAL_OR_KEYWORD and VAR_POSITIONAL as well as the keyword arguments,
    i.e., KEYWORD_ONLY or VAR_KEYWORD.

    Args:
        stmt: The parameterised statement.
        module_aliases: A naming scope for module alias names.
        variable_names: A naming scope for variable names.

    Returns:
        Two lists of AST statements, one for args and one for kwargs
    """
    args: list[ast.expr] = []
    kwargs = []

    gen_callable: GenericCallableAccessibleObject = cast(
        GenericCallableAccessibleObject, stmt.accessible_object()
    )

    left_of_current: list[str] = []

    parameters = gen_callable.inferred_signature.signature.parameters

    for param_name, param in parameters.items():
        if param_name in stmt.args:
            # The variable that is passed in as an argument
            var = au.create_full_name(
                variable_names,
                module_aliases,
                stmt.args[param_name],
                load=True,
            )
            match param.kind:
                case Parameter.POSITIONAL_ONLY:
                    args.append(var)
                case Parameter.POSITIONAL_OR_KEYWORD:
                    # If a POSITIONAL_OR_KEYWORD parameter left of the current param
                    # has a default, and we did not pass a value, we must pass the
                    # current value by keyword, otherwise by position.
                    if any(
                        parameters[left].default is not Parameter.empty
                        and left not in stmt.args
                        for left in left_of_current
                    ):
                        kwargs.append(
                            ast.keyword(
                                arg=param_name,
                                value=var,
                            )
                        )
                    else:
                        args.append(var)
                case Parameter.KEYWORD_ONLY:
                    kwargs.append(
                        ast.keyword(
                            arg=param_name,
                            value=var,
                        )
                    )
                case Parameter.VAR_POSITIONAL:
                    # Append *args, if necessary.
                    args.append(
                        ast.Starred(
                            value=var,
                            ctx=ast.Load(),
                        )
                    )
                case Parameter.VAR_KEYWORD:
                    # Append **kwargs, if necessary.
                    kwargs.append(
                        ast.keyword(
                            arg=None,
                            value=var,
                        )
                    )
        left_of_current.append(param_name)
    return args, kwargs


def create_module_alias(
    module_name,
    module_aliases: ns.AbstractNamingScope,
) -> ast.Name:
    """Create a name node for a module alias.

    Args:
        module_name: The name of the module.
        module_aliases: A naming scope for module alias names.

    Returns:
        An AST statement.
    """
    return ast.Name(id=module_aliases.get_name(module_name), ctx=ast.Load())


def create_statement(
    value: ast.expr,
    targets: list[ast.Name | ast.Attribute] | None,
) -> ast.stmt:
    """Create an assignment statement.

    Args:
        value: The value to assign.
        targets: The targets to assign to.

    Returns:
        An AST statement.
    """
    if targets is None:
        return ast.Expr(value=value)

    return ast.Assign(
        targets=targets,
        value=value,
    )


class StatementToAstTransformerFunction(Protocol[S_contra]):
    """Protocol for converting our internal statements to AST nodes."""

    @abstractmethod
    def __call__(
        self,
        stmt: S_contra,
        module_aliases: ns.AbstractNamingScope,
        variable_names: ns.AbstractNamingScope,
        store_call_return: bool,  # noqa: FBT001
    ) -> ast.stmt:
        """Transforms our internal statements to Python AST nodes.

        Args:
            stmt: The statement to convert.
            module_aliases: A naming scope for module alias names.
            variable_names: A naming scope for variable names.
            store_call_return: Should the result of a call be stored in a variable?

        Returns:
            The AST node.
        """


def transform_primitive_statement(
    stmt: (
        IntPrimitiveStatement
        | FloatPrimitiveStatement
        | ComplexPrimitiveStatement
        | StringPrimitiveStatement
        | BytesPrimitiveStatement
        | BooleanPrimitiveStatement
        | NoneStatement
    ),
    module_aliases: ns.AbstractNamingScope,
    variable_names: ns.AbstractNamingScope,
    store_call_return: bool,  # noqa: FBT001
) -> ast.stmt:
    """Transforms a primitive statement to an AST node.

    Args:
        stmt: The primitive statement to transform.
        module_aliases: A naming scope for module alias names.
        variable_names: A naming scope for variable names.
        store_call_return: Should the result of a call be stored in a variable?

    Returns:
        The AST node.
    """
    if store_call_return:
        targets = [
            au.create_full_name(
                variable_names, module_aliases, stmt.ret_val, load=False
            )
        ]
    else:
        targets = None

    return create_statement(
        value=ast.Constant(value=stmt.value),
        targets=targets,
    )


def transform_enum_statement(
    stmt: EnumPrimitiveStatement,
    module_aliases: ns.AbstractNamingScope,
    variable_names: ns.AbstractNamingScope,
    store_call_return: bool,  # noqa: FBT001
) -> ast.stmt:
    """Transforms an enum statement to an AST node.

    Args:
        stmt: The enum statement to transform.
        module_aliases: A naming scope for module alias names.
        variable_names: A naming scope for variable names.
        store_call_return: Should the result of a call be stored in a variable?

    Returns:
        The AST node.
    """
    accessible_object = stmt.accessible_object()
    owner = accessible_object.owner
    assert owner

    if store_call_return:
        targets = [
            au.create_full_name(
                variable_names, module_aliases, stmt.ret_val, load=False
            )
        ]
    else:
        targets = None

    return create_statement(
        value=ast.Attribute(
            value=ast.Attribute(
                value=create_module_alias(
                    accessible_object.exporter_module, module_aliases
                ),
                attr=owner.name,
                ctx=ast.Load(),
            ),
            attr=stmt.value_name,
            ctx=ast.Load(),
        ),
        targets=targets,
    )


def transform_class_primitive_statement(
    stmt: ClassPrimitiveStatement,
    module_aliases: ns.AbstractNamingScope,
    variable_names: ns.AbstractNamingScope,
    store_call_return: bool,  # noqa: FBT001
) -> ast.stmt:
    """Transforms a class primitive statement to an AST node.

    Args:
        stmt: The class primitive statement to transform.
        module_aliases: A naming scope for module alias names.
        variable_names: A naming scope for variable names.
        store_call_return: Should the result of a call be stored in a variable?

    Returns:
        The AST node.
    """
    cls = stmt.type_info

    if store_call_return:
        targets = [
            au.create_full_name(
                variable_names, module_aliases, stmt.ret_val, load=False
            )
        ]
    else:
        targets = None

    return create_statement(
        # TODO(fk) think about nested classes, also for enums.
        value=ast.Attribute(
            value=create_module_alias(cls.module, module_aliases),
            attr=cls.name,
            ctx=ast.Load(),
        ),
        targets=targets,
    )


def transform_constructor_statement(
    stmt: ConstructorStatement,
    module_aliases: ns.AbstractNamingScope,
    variable_names: ns.AbstractNamingScope,
    store_call_return: bool,  # noqa: FBT001
) -> ast.stmt:
    """Transforms a constructor statement to an AST node.

    Args:
        stmt: The constructor statement to transform.
        module_aliases: A naming scope for module alias names.
        variable_names: A naming scope for variable names.
        store_call_return: Should the result of a call be stored in a variable?

    Returns:
        The AST node.
    """
    accessible_object = stmt.accessible_object()
    owner = accessible_object.owner
    assert owner
    args, kwargs = create_args(stmt, module_aliases, variable_names)
    call = ast.Call(
        func=ast.Attribute(
            attr=owner.name,
            ctx=ast.Load(),
            value=create_module_alias(
                accessible_object.exporter_module, module_aliases
            ),
        ),
        args=args,
        keywords=kwargs,
    )

    if store_call_return:
        targets = [
            au.create_full_name(
                variable_names,
                module_aliases,
                stmt.ret_val,
                load=False,
            )
        ]
    else:
        targets = None

    return create_statement(
        value=call,
        targets=targets,
    )


def transform_method_statement(
    stmt: MethodStatement,
    module_aliases: ns.AbstractNamingScope,
    variable_names: ns.AbstractNamingScope,
    store_call_return: bool,  # noqa: FBT001
) -> ast.stmt:
    """Transforms a method statement to an AST node.

    Args:
        stmt: The method statement to transform.
        module_aliases: A naming scope for module alias names.
        variable_names: A naming scope for variable names.
        store_call_return: Should the result of a call be stored in a variable?

    Returns:
        The AST node.
    """
    args, kwargs = create_args(stmt, module_aliases, variable_names)
    call = ast.Call(
        func=ast.Attribute(
            attr=stmt.accessible_object().callable.__name__,
            ctx=ast.Load(),
            value=au.create_full_name(
                variable_names, module_aliases, stmt.callee, load=True
            ),
        ),
        args=args,
        keywords=kwargs,
    )

    if store_call_return:
        targets = [
            au.create_full_name(
                variable_names,
                module_aliases,
                stmt.ret_val,
                load=False,
            )
        ]
    else:
        targets = None

    return create_statement(
        value=call,
        targets=targets,
    )


def transform_function_statement(
    stmt: FunctionStatement,
    module_aliases: ns.AbstractNamingScope,
    variable_names: ns.AbstractNamingScope,
    store_call_return: bool,  # noqa: FBT001
) -> ast.stmt:
    """Transforms a function statement to an AST node.

    Args:
        stmt: The function statement to transform.
        module_aliases: A naming scope for module alias names.
        variable_names: A naming scope for variable names.
        store_call_return: Should the result of a call be stored in a variable?

    Returns:
        The AST node.
    """
    args, kwargs = create_args(stmt, module_aliases, variable_names)
    call = ast.Call(
        func=ast.Attribute(
            attr=stmt.accessible_object().callable.__name__,
            ctx=ast.Load(),
            value=create_module_alias(
                stmt.accessible_object().callable.__module__, module_aliases
            ),
        ),
        args=args,
        keywords=kwargs,
    )

    if store_call_return:
        targets = [
            au.create_full_name(
                variable_names,
                module_aliases,
                stmt.ret_val,
                load=False,
            )
        ]
    else:
        targets = None

    return create_statement(
        value=call,
        targets=targets,
    )


def transform_field_statement(
    stmt: FieldStatement,
    module_aliases: ns.AbstractNamingScope,
    variable_names: ns.AbstractNamingScope,
    store_call_return: bool,  # noqa: FBT001
) -> ast.stmt:
    """Transforms a field statement to an AST node.

    Args:
        stmt: The field statement to transform.
        module_aliases: A naming scope for module alias names.
        variable_names: A naming scope for variable names.
        store_call_return: Should the result of a call be stored in a variable?

    Returns:
        The AST node.
    """
    targets: list[ast.Name | ast.Attribute] | None
    if store_call_return:
        targets = [
            ast.Name(
                id=variable_names.get_name(stmt.ret_val),
                ctx=ast.Store(),
            )
        ]
    else:
        targets = None

    return create_statement(
        value=ast.Attribute(
            attr=stmt.field.field,
            ctx=ast.Load(),
            value=au.create_full_name(
                variable_names, module_aliases, stmt.source, load=True
            ),
        ),
        targets=targets,
    )


def transform_assignment_statement(
    stmt: AssignmentStatement,
    module_aliases: ns.AbstractNamingScope,
    variable_names: ns.AbstractNamingScope,
    store_call_return: bool,  # noqa: FBT001
) -> ast.stmt:
    """Transforms an assignment statement to an AST node.

    Args:
        stmt: The assignment statement to transform.
        module_aliases: A naming scope for module alias names.
        variable_names: A naming scope for variable names.
        store_call_return: Should the result of a call be stored in a variable?

    Returns:
        The AST node.
    """
    if store_call_return:
        targets = [
            au.create_full_name(variable_names, module_aliases, stmt.lhs, load=False)
        ]
    else:
        targets = None

    return create_statement(
        value=au.create_full_name(variable_names, module_aliases, stmt.rhs, load=True),
        targets=targets,
    )


def transform_list_statement(
    stmt: ListStatement,
    module_aliases: ns.AbstractNamingScope,
    variable_names: ns.AbstractNamingScope,
    store_call_return: bool,  # noqa: FBT001
) -> ast.stmt:
    """Transforms a list statement to an AST node.

    Args:
        stmt: The list statement to transform.
        module_aliases: A naming scope for module alias names.
        variable_names: A naming scope for variable names.
        store_call_return: Should the result of a call be stored in a variable?

    Returns:
        The AST node.
    """
    if store_call_return:
        targets = [
            au.create_full_name(
                variable_names, module_aliases, stmt.ret_val, load=False
            )
        ]
    else:
        targets = None

    return create_statement(
        value=ast.List(
            elts=[
                au.create_full_name(variable_names, module_aliases, x, load=True)
                for x in stmt.elements
            ],
            ctx=ast.Load(),
        ),
        targets=targets,
    )


def transform_set_statement(
    stmt: SetStatement,
    module_aliases: ns.AbstractNamingScope,
    variable_names: ns.AbstractNamingScope,
    store_call_return: bool,  # noqa: FBT001
) -> ast.stmt:
    """Transforms a set statement to an AST node.

    Args:
        stmt: The set statement to transform.
        module_aliases: A naming scope for module alias names.
        variable_names: A naming scope for variable names.
        store_call_return: Should the result of a call be stored in a variable?

    Returns:
        The AST node.
    """
    # There is no literal for empty sets, so we have to write "set()"
    inner: ast.Call | ast.Set
    if len(stmt.elements) == 0:
        inner = ast.Call(func=ast.Name(id="set", ctx=ast.Load()), args=[], keywords=[])
    else:
        inner = ast.Set(
            elts=[
                au.create_full_name(variable_names, module_aliases, x, load=True)
                for x in stmt.elements
            ],
            ctx=ast.Load(),
        )

    if store_call_return:
        targets = [
            au.create_full_name(
                variable_names, module_aliases, stmt.ret_val, load=False
            )
        ]
    else:
        targets = None

    return create_statement(
        value=inner,
        targets=targets,
    )


def transform_tuple_statement(
    stmt: TupleStatement,
    module_aliases: ns.AbstractNamingScope,
    variable_names: ns.AbstractNamingScope,
    store_call_return: bool,  # noqa: FBT001
) -> ast.stmt:
    """Transforms a tuple statement to an AST node.

    Args:
        stmt: The tuple statement to transform.
        module_aliases: A naming scope for module alias names.
        variable_names: A naming scope for variable names.
        store_call_return: Should the result of a call be stored in a variable?

    Returns:
        The AST node.
    """
    if store_call_return:
        targets = [
            au.create_full_name(
                variable_names, module_aliases, stmt.ret_val, load=False
            )
        ]
    else:
        targets = None

    return create_statement(
        value=ast.Tuple(
            elts=[
                au.create_full_name(variable_names, module_aliases, x, load=True)
                for x in stmt.elements
            ],
            ctx=ast.Load(),
        ),
        targets=targets,
    )


def transform_dict_statement(
    stmt: DictStatement,
    module_aliases: ns.AbstractNamingScope,
    variable_names: ns.AbstractNamingScope,
    store_call_return: bool,  # noqa: FBT001
) -> ast.stmt:
    """Transforms a dictionary statement to an AST node.

    Args:
        stmt: The dictionary statement to transform.
        module_aliases: A naming scope for module alias names.
        variable_names: A naming scope for variable names.
        store_call_return: Should the result of a call be stored in a variable?

    Returns:
        The AST node.
    """
    if store_call_return:
        targets = [
            au.create_full_name(
                variable_names, module_aliases, stmt.ret_val, load=False
            )
        ]
    else:
        targets = None

    return create_statement(
        value=ast.Dict(
            keys=[
                au.create_full_name(variable_names, module_aliases, x[0], load=True)
                for x in stmt.elements
            ],
            values=[
                au.create_full_name(variable_names, module_aliases, x[1], load=True)
                for x in stmt.elements
            ],
        ),
        targets=targets,
    )


BUILTIN_TRANSFORMER_FUNCTIONS: dict[type, StatementToAstTransformerFunction] = {
    IntPrimitiveStatement: transform_primitive_statement,
    FloatPrimitiveStatement: transform_primitive_statement,
    ComplexPrimitiveStatement: transform_primitive_statement,
    StringPrimitiveStatement: transform_primitive_statement,
    BytesPrimitiveStatement: transform_primitive_statement,
    BooleanPrimitiveStatement: transform_primitive_statement,
    NoneStatement: transform_primitive_statement,
    EnumPrimitiveStatement: transform_enum_statement,
    ClassPrimitiveStatement: transform_class_primitive_statement,
    ConstructorStatement: transform_constructor_statement,
    MethodStatement: transform_method_statement,
    FunctionStatement: transform_function_statement,
    FieldStatement: transform_field_statement,
    AssignmentStatement: transform_assignment_statement,
    ListStatement: transform_list_statement,
    SetStatement: transform_set_statement,
    TupleStatement: transform_tuple_statement,
    DictStatement: transform_dict_statement,
}
