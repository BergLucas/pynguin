#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""A plugin for generating int tensors."""

from __future__ import annotations

import ast
import math

from typing import TYPE_CHECKING

import pynguin.utils.ast_util as au

from pynguin import configuration as config
from pynguin.ga.postprocess import UnusedPrimitiveOrCollectionStatementRemoverFunction
from pynguin.ga.postprocess import remove_collection_or_primitive
from pynguin.testcase.statement import Statement
from pynguin.testcase.statement import VariableCreatingStatement
from pynguin.testcase.statement_to_ast import StatementToAstTransformerFunction
from pynguin.testcase.statement_to_ast import create_statement
from pynguin.testcase.testfactory import SupportedTypes
from pynguin.testcase.testfactory import TestFactory
from pynguin.testcase.testfactory import VariableGenerator
from pynguin.testcase.variablereference import VariableReference
from pynguin.utils import randomness


if TYPE_CHECKING:
    from argparse import ArgumentParser
    from argparse import Namespace

    import pynguin.utils.namingscope as ns

    from pynguin.analyses.module import TestCluster
    from pynguin.analyses.typesystem import Instance
    from pynguin.analyses.typesystem import ProperType
    from pynguin.analyses.typesystem import TupleType
    from pynguin.testcase.testcase import TestCase

NAME = "int_tensor_fuzzer"

int_tensor_weight: float = 0
int_tensor_concrete_weight: float = 0
int_tensor_min_ndim: int = 0
int_tensor_max_ndim: int = 0
int_tensor_min_ndim_length: int = 0
int_tensor_max_ndim_length: int = 0
int_tensor_min_number: int = 0
int_tensor_max_number: int = 0

int_tensor_types: set[type] = set()


def parser_hook(parser: ArgumentParser) -> None:  # noqa: D103
    parser.add_argument(
        "--int_tensor_weight",
        type=float,
        default=100.0,
        help="""Weight to use an int tensor as parameter type during test generation.
        Expects values > 0""",
    )
    parser.add_argument(
        "--int_tensor_concrete_weight",
        type=float,
        default=100.0,
        help="""Weight to convert an abstract type to a int tensor."""
        """Expects values > 0""",
    )
    parser.add_argument(
        "--int_tensor_min_ndim",
        type=int,
        default=1,
        help="Minimum number of dimensions for an int tensor.",
    )
    parser.add_argument(
        "--int_tensor_max_ndim",
        type=int,
        default=3,
        help="Maximum number of dimensions for an int tensor.",
    )
    parser.add_argument(
        "--int_tensor_min_ndim_length",
        type=int,
        default=1,
        help="Minimum length of a dimension for an int tensor.",
    )
    parser.add_argument(
        "--int_tensor_max_ndim_length",
        type=int,
        default=10,
        help="Maximum length of a dimension for an int tensor.",
    )
    parser.add_argument(
        "--int_tensor_min_number",
        type=int,
        default=-100,
        help="Minimum value for an int tensor.",
    )
    parser.add_argument(
        "--int_tensor_max_number",
        type=int,
        default=100,
        help="Maximum value for an int tensor.",
    )


def _create_int_tensor_type(ndim: int) -> type:
    """Create a type for an int tensor with a given number of dimensions.

    Args:
        ndim: The number of dimensions

    Returns:
        A type for an int tensor
    """
    if ndim == 1:
        return list[int]
    return list[_create_int_tensor_type(ndim - 1)]  # type: ignore[misc]


def configuration_hook(plugin_config: Namespace) -> None:  # noqa: D103
    global int_tensor_weight  # noqa: PLW0603
    global int_tensor_concrete_weight  # noqa: PLW0603
    global int_tensor_min_ndim  # noqa: PLW0603
    global int_tensor_max_ndim  # noqa: PLW0603
    global int_tensor_min_ndim_length  # noqa: PLW0603
    global int_tensor_max_ndim_length  # noqa: PLW0603
    global int_tensor_min_number  # noqa: PLW0603
    global int_tensor_max_number  # noqa: PLW0603

    int_tensor_weight = plugin_config.int_tensor_weight
    int_tensor_concrete_weight = plugin_config.int_tensor_concrete_weight
    int_tensor_min_ndim = plugin_config.int_tensor_min_ndim
    int_tensor_max_ndim = plugin_config.int_tensor_max_ndim
    int_tensor_min_ndim_length = plugin_config.int_tensor_min_ndim_length
    int_tensor_max_ndim_length = plugin_config.int_tensor_max_ndim_length
    int_tensor_min_number = plugin_config.int_tensor_min_number
    int_tensor_max_number = plugin_config.int_tensor_max_number


def types_hook() -> list[type]:  # noqa: D103
    int_tensor_types.update(
        _create_int_tensor_type(ndim)
        for ndim in range(int_tensor_min_ndim, int_tensor_max_ndim + 1)
    )
    return list(int_tensor_types)


def test_cluster_hook(test_cluster: TestCluster) -> None:  # noqa: D103
    for int_tensor_type in int_tensor_types:
        typ = test_cluster.type_system.convert_type_hint(int_tensor_type)
        test_cluster.set_concrete_weight(
            typ, int_tensor_concrete_weight / len(int_tensor_types)
        )


def ast_transformer_hook(  # noqa: D103
    transformer_functions: dict[type, StatementToAstTransformerFunction]
) -> None:
    transformer_functions[IntTensorStatement] = transform_int_tensor_statement


def statement_remover_hook(  # noqa: D103
    remover_functions: dict[type, UnusedPrimitiveOrCollectionStatementRemoverFunction]
) -> None:
    remover_functions[IntTensorStatement] = remove_collection_or_primitive


def variable_generator_hook(  # noqa: D103
    generators: dict[VariableGenerator, float]
) -> None:
    generators[IntTensorVariableGenerator()] = int_tensor_weight


def _create_ast_tensor(
    tensor: list[int], shape: tuple[int, ...], index: int, length: int
) -> ast.List:
    """Create an AST representation of a tensor from a list of integers and a shape.

    Args:
        tensor: The tensor as a list of integers
        shape: The shape of the tensor
        index: The current index in the tensor
        length: The total length of the tensor

    Returns:
        An AST representation of the tensor
    """
    first_dim = shape[0]
    other_dims = shape[1:]

    elts: list[ast.Constant] | list[ast.List]
    if other_dims:
        split_length = length // first_dim
        elts = [
            _create_ast_tensor(
                tensor, other_dims, index + i * split_length, split_length
            )
            for i in range(first_dim)
        ]
    else:
        assert first_dim == length
        elts = [ast.Constant(value=tensor[index + i]) for i in range(first_dim)]

    return ast.List(
        elts=elts,
        ctx=ast.Load(),
    )


def transform_int_tensor_statement(
    stmt: IntTensorStatement,
    module_aliases: ns.AbstractNamingScope,
    variable_names: ns.AbstractNamingScope,
    store_call_return: bool,  # noqa: FBT001
) -> ast.stmt:
    """Transforms an int tensor statement to an AST node.

    Args:
        stmt: The int tensor statement to transform.
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
        targets = []

    return create_statement(
        value=_create_ast_tensor(stmt.tensor, stmt.shape, 0, len(stmt.tensor)),
        targets=targets,
    )


class _IntTensorSupportedTypes(SupportedTypes):
    """Supported types for int tensors."""

    def visit_instance(self, left: Instance) -> bool:
        return left.type.raw_type in int_tensor_types

    def visit_tuple_type(self, left: TupleType) -> bool:
        return False


int_tensor_supported_types = _IntTensorSupportedTypes()


class IntTensorVariableGenerator(VariableGenerator):
    """An int tensor variable generator."""

    @property
    def supported_types(self) -> SupportedTypes:  # noqa: D102
        return int_tensor_supported_types

    def generate_variable(  # noqa: D102
        self,
        test_factory: TestFactory,
        test_case: TestCase,
        parameter_type: ProperType,
        position: int,
        recursion_depth: int,
        *,
        allow_none: bool,
    ) -> VariableReference | None:
        nb_dims = randomness.next_int(int_tensor_min_ndim, int_tensor_max_ndim + 1)
        shape = tuple(
            randomness.next_int(
                int_tensor_min_ndim_length, int_tensor_max_ndim_length + 1
            )
            for _ in range(nb_dims)
        )
        tensor = [
            randomness.next_int(int_tensor_min_number, int_tensor_max_number + 1)
            for _ in range(math.prod(shape))
        ]
        statement = IntTensorStatement(test_case, parameter_type, tensor, shape)
        ret = test_case.add_variable_creating_statement(statement, position)
        ret.distance = recursion_depth
        return ret


class IntTensorStatement(VariableCreatingStatement):
    """Represents a int tensor."""

    def __init__(
        self,
        test_case: TestCase,
        type_: ProperType,
        tensor: list[int],
        shape: tuple[int, ...],
    ):
        """Initializes the collection statement.

        Args:
            test_case: The test case the statement belongs to
            type_: The type of the elements in the collection
            tensor: A tensor
            shape: The shape of the tensor
        """
        super().__init__(
            test_case,
            VariableReference(test_case, type_),
        )
        assert math.prod(shape) == len(tensor), "Shape does not match tensor size"
        self._tensor = tensor
        self._shape = shape

    @property
    def tensor(self) -> list[int]:
        """The tensor.

        Returns:
            A tensor
        """
        return self._tensor

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the tensor.

        Returns:
            A tuple of integers
        """
        return self._shape

    def accessible_object(self) -> None:  # noqa: D102
        return None

    def mutate(self) -> bool:  # noqa: D102
        length = len(self._tensor)

        changed = False
        if (
            randomness.next_float()
            < config.configuration.search_algorithm.test_change_probability
            and length > 0
        ):
            p_per_element = 1.0 / length
            for i, _ in enumerate(self._tensor):
                if randomness.next_float() < p_per_element:
                    self._tensor[i] = randomness.next_int()
                    changed = True

        return changed

    def structural_eq(  # noqa: D102
        self, other: Statement, memo: dict[VariableReference, VariableReference]
    ) -> bool:
        if not isinstance(other, IntTensorStatement):
            return False
        return (
            self._tensor == other._tensor  # noqa: SLF001
            and self._shape == other._shape  # noqa: SLF001
        )

    def structural_hash(self, memo: dict[VariableReference, int]) -> int:  # noqa: D102
        return hash((tuple(self._tensor), self._shape))

    def get_variable_references(self) -> set[VariableReference]:  # noqa: D102
        return {self.ret_val}

    def replace(  # noqa: D102
        self, old: VariableReference, new: VariableReference
    ) -> None:
        if self.ret_val == old:
            self.ret_val = new

    def clone(  # noqa: D102
        self,
        test_case: TestCase,
        memo: dict[VariableReference, VariableReference],
    ) -> Statement:
        return IntTensorStatement(
            test_case, self.ret_val.type, list(self._tensor), self._shape
        )
