#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides a plugin to generate Pandas dataframes as test data."""
from __future__ import annotations

import ast
import io

from copy import deepcopy
from typing import TYPE_CHECKING

import pynguin.utils.ast_util as au

from pynguin.analyses.typesystem import Instance
from pynguin.ga.postprocess import UnusedPrimitiveOrCollectionStatementRemoverFunction
from pynguin.ga.postprocess import remove_collection_or_primitive
from pynguin.plugins.grammar_fuzzer.csv import create_csv_grammar
from pynguin.plugins.grammar_fuzzer.fuzzer import GrammarDerivationTree
from pynguin.plugins.grammar_fuzzer.fuzzer import GrammarFuzzer
from pynguin.testcase import variablereference as vr
from pynguin.testcase.statement import VariableCreatingStatement
from pynguin.testcase.statement_to_ast import StatementToAstTransformerFunction
from pynguin.testcase.statement_to_ast import create_module_alias
from pynguin.testcase.statement_to_ast import create_statement
from pynguin.testcase.testfactory import SupportedTypes
from pynguin.testcase.testfactory import TestFactory
from pynguin.testcase.testfactory import VariableGenerator
from pynguin.utils import randomness


if TYPE_CHECKING:
    from argparse import ArgumentParser
    from argparse import Namespace
    from types import ModuleType

    import pynguin.utils.namingscope as ns

    from pynguin.analyses.module import TestCluster
    from pynguin.analyses.typesystem import ProperType
    from pynguin.analyses.typesystem import TupleType
    from pynguin.testcase.statement import Statement
    from pynguin.testcase.statement_to_ast import StatementToAstTransformerFunction
    from pynguin.testcase.testcase import TestCase
    from pynguin.testcase.variablereference import VariableReference

NAME = "pandas_dataframe_fuzzer"

pandas_dataframe_weight: float = 0.0
pandas_dataframe_concrete_weight: float = 0.0
pandas_dataframe_min_columns_number: int = 0
pandas_dataframe_max_columns_number: int = 0
pandas_dataframe_min_field_length: int = 0
pandas_dataframe_max_field_length: int = 0
pandas_dataframe_min_rows_number: int = 0
pandas_dataframe_max_rows_number: int = 0
pandas_dataframe_number_column_probability: float = 0.0
pandas_dataframe_no_header: bool = False
pandas_dataframe_min_non_terminal: int = 0
pandas_dataframe_max_non_terminal: int = 0

pd: ModuleType


def parser_hook(parser: ArgumentParser) -> None:  # noqa: D103
    parser.add_argument(
        "--pandas_dataframe_weight",
        type=float,
        default=0.0,
        help="""Weight to use a Pandas dataframe object as parameter type."""
        """Expects values > 0""",
    )
    parser.add_argument(
        "--pandas_dataframe_concrete_weight",
        type=float,
        default=100.0,
        help="""Weight to convert an abstract type to a Pandas dataframe object."""
        """Expects values > 0""",
    )
    parser.add_argument(
        "--pandas_dataframe_min_columns_number",
        type=int,
        default=1,
        help="Minimum number of columns in the generated Pandas dataframe",
    )
    parser.add_argument(
        "--pandas_dataframe_max_columns_number",
        type=int,
        default=5,
        help="Maximum number of columns in the generated Pandas dataframe",
    )
    parser.add_argument(
        "--pandas_dataframe_min_field_length",
        type=int,
        default=1,
        help="Minimum length of a field in the generated Pandas dataframe",
    )
    parser.add_argument(
        "--pandas_dataframe_max_field_length",
        type=int,
        default=10,
        help="Maximum length of a field in the generated Pandas dataframe",
    )
    parser.add_argument(
        "--pandas_dataframe_min_rows_number",
        type=int,
        default=1,
        help="Minimum number of rows in the generated Pandas dataframe",
    )
    parser.add_argument(
        "--pandas_dataframe_max_rows_number",
        type=int,
        default=5,
        help="Maximum number of rows in the generated Pandas dataframe",
    )
    parser.add_argument(
        "--pandas_dataframe_number_column_probability",
        type=float,
        default=0.75,
        help="Probability that a column has the number type",
    )
    parser.add_argument(
        "--pandas_dataframe_no_header",
        action="store_true",
        help="Remove header from the Pandas dataframe",
    )
    parser.add_argument(
        "--pandas_dataframe_min_non_terminal",
        type=int,
        default=10,
        help="Minimum number of non-terminal symbols in the grammar",
    )
    parser.add_argument(
        "--pandas_dataframe_max_non_terminal",
        type=int,
        default=50,
        help="Maximum number of non-terminal symbols in the grammar",
    )


def configuration_hook(plugin_config: Namespace) -> None:  # noqa: D103
    global pandas_dataframe_weight  # noqa: PLW0603
    global pandas_dataframe_concrete_weight  # noqa: PLW0603
    global pandas_dataframe_min_columns_number  # noqa: PLW0603
    global pandas_dataframe_max_columns_number  # noqa: PLW0603
    global pandas_dataframe_min_field_length  # noqa: PLW0603
    global pandas_dataframe_max_field_length  # noqa: PLW0603
    global pandas_dataframe_min_rows_number  # noqa: PLW0603
    global pandas_dataframe_max_rows_number  # noqa: PLW0603
    global pandas_dataframe_number_column_probability  # noqa: PLW0603
    global pandas_dataframe_no_header  # noqa: PLW0603
    global pandas_dataframe_min_non_terminal  # noqa: PLW0603
    global pandas_dataframe_max_non_terminal  # noqa: PLW0603

    pandas_dataframe_weight = plugin_config.pandas_dataframe_weight
    pandas_dataframe_concrete_weight = plugin_config.pandas_dataframe_concrete_weight
    pandas_dataframe_min_columns_number = (
        plugin_config.pandas_dataframe_min_columns_number
    )
    pandas_dataframe_max_columns_number = (
        plugin_config.pandas_dataframe_max_columns_number
    )
    pandas_dataframe_min_field_length = plugin_config.pandas_dataframe_min_field_length
    pandas_dataframe_max_field_length = plugin_config.pandas_dataframe_max_field_length
    pandas_dataframe_min_rows_number = plugin_config.pandas_dataframe_min_rows_number
    pandas_dataframe_max_rows_number = plugin_config.pandas_dataframe_max_rows_number
    pandas_dataframe_number_column_probability = (
        plugin_config.pandas_dataframe_number_column_probability
    )
    pandas_dataframe_no_header = plugin_config.pandas_dataframe_no_header
    pandas_dataframe_min_non_terminal = plugin_config.pandas_dataframe_min_non_terminal
    pandas_dataframe_max_non_terminal = plugin_config.pandas_dataframe_max_non_terminal


def types_hook() -> list[tuple[type, ModuleType]]:  # noqa: D103
    global pd  # noqa: PLW0603
    import pandas as pd  # noqa: PLC0415

    return [(pd.DataFrame, pd)]


def test_cluster_hook(test_cluster: TestCluster) -> None:  # noqa: D103
    type_info = test_cluster.type_system.to_type_info(pd.DataFrame)
    typ = test_cluster.type_system.make_instance(type_info)
    test_cluster.set_concrete_weight(typ, pandas_dataframe_concrete_weight)


def ast_transformer_hook(  # noqa: D103
    transformer_functions: dict[type, StatementToAstTransformerFunction]
) -> None:
    transformer_functions[PandasDataframeStatement] = (
        transform_pandas_dataframe_statement
    )


def statement_remover_hook(  # noqa: D103
    remover_functions: dict[type, UnusedPrimitiveOrCollectionStatementRemoverFunction]
) -> None:
    remover_functions[PandasDataframeStatement] = remove_collection_or_primitive


def variable_generator_hook(  # noqa: D103
    generators: dict[VariableGenerator, float]
) -> None:
    generators[PandasVariableGenerator()] = pandas_dataframe_weight


def transform_pandas_dataframe_statement(
    stmt: PandasDataframeStatement,
    module_aliases: ns.AbstractNamingScope,
    variable_names: ns.AbstractNamingScope,
    store_call_return: bool,  # noqa: FBT001
) -> ast.stmt:
    """Transform a Pandas dataframe statement to an AST node.

    Args:
        stmt: The statement to transform.
        module_aliases: A naming scope for module alias names.
        variable_names: A naming scope for variable names.
        store_call_return: Should the result of a call be stored in a variable?

    Returns:
        The AST node.
    """
    io_module_name = io.__name__
    string_io_attr = io.StringIO.__name__

    string_io_call = ast.Call(
        func=ast.Attribute(
            attr=string_io_attr,
            ctx=ast.Load(),
            value=create_module_alias(io_module_name, module_aliases),
        ),
        args=[ast.Constant(value=stmt.csv_string)],
        keywords=[],
    )

    module_name = pd.__name__
    attr = pd.read_csv.__name__

    call = ast.Call(
        func=ast.Attribute(
            attr=attr,
            ctx=ast.Load(),
            value=create_module_alias(module_name, module_aliases),
        ),
        args=[string_io_call],
        keywords=[],
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


class _PandasDataframeSupportedTypes(SupportedTypes):
    """Supported types for Pandas dataframes."""

    def visit_instance(self, left: Instance) -> bool:
        try:
            return issubclass(left.type.raw_type, pd.DataFrame)
        except TypeError:
            return False

    def visit_tuple_type(self, left: TupleType) -> bool:
        return False


pandas_dataframe_supported_types = _PandasDataframeSupportedTypes()


class PandasVariableGenerator(VariableGenerator):
    """A Pandas dataframes variable generator."""

    @property
    def supported_types(self) -> SupportedTypes:  # noqa: D102
        return pandas_dataframe_supported_types

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
        columns_number = randomness.next_int(
            pandas_dataframe_min_columns_number, pandas_dataframe_max_columns_number
        )

        csv_grammar = create_csv_grammar(
            columns_number=columns_number,
            min_field_length=pandas_dataframe_min_field_length,
            max_field_length=pandas_dataframe_max_field_length,
            min_rows_number=pandas_dataframe_min_rows_number,
            max_rows_number=pandas_dataframe_max_rows_number,
            number_column_probability=pandas_dataframe_number_column_probability,
            include_header=not pandas_dataframe_no_header,
        )

        csv_grammar_fuzzer = GrammarFuzzer(
            csv_grammar,
            pandas_dataframe_min_non_terminal,
            pandas_dataframe_max_non_terminal,
        )

        pandas_dataframe_ret = test_case.add_variable_creating_statement(
            PandasDataframeStatement(test_case, csv_grammar_fuzzer),
            position,
        )
        pandas_dataframe_ret.distance = recursion_depth

        return pandas_dataframe_ret


class PandasDataframeStatement(VariableCreatingStatement):
    """A statement that creates a Pandas dataframe."""

    def __init__(
        self,
        test_case: TestCase,
        fuzzer: GrammarFuzzer,
        derivation_tree: GrammarDerivationTree | None = None,
    ) -> None:
        """Create a new Pandas dataframe statement.

        Args:
            test_case: The test case
            fuzzer: The fuzzer to use
            derivation_tree: The derivation tree to use
        """
        if derivation_tree is None:
            derivation_tree = fuzzer.create_tree()

        self._derivation_tree = derivation_tree
        self._fuzzer = fuzzer
        self._csv_string = str(derivation_tree)

        pandas_dataframe_type_info = test_case.test_cluster.type_system.to_type_info(
            pd.DataFrame
        )

        pandas_dataframe_instance = Instance(pandas_dataframe_type_info)

        super().__init__(
            test_case,
            vr.VariableReference(test_case, pandas_dataframe_instance),
        )

    @property
    def csv_string(self) -> str:
        """The CSV string representation.

        Returns:
            The CSV string representation.
        """
        return self._csv_string

    def clone(  # noqa: D102
        self,
        test_case: TestCase,
        memo: dict[VariableReference, VariableReference],
    ) -> PandasDataframeStatement:
        return PandasDataframeStatement(
            test_case, self._fuzzer, deepcopy(self._derivation_tree)
        )

    def accessible_object(self) -> None:  # noqa: D102
        return None

    def mutate(self) -> bool:  # noqa: D102
        mutated = self._fuzzer.mutate_tree(self._derivation_tree)

        if mutated:
            self._csv_string = str(self._derivation_tree)

        return mutated

    def get_variable_references(self) -> set[vr.VariableReference]:  # noqa: D102
        return {self.ret_val}

    def replace(  # noqa: D102
        self, old: vr.VariableReference, new: vr.VariableReference
    ) -> None:
        if self.ret_val == old:
            self.ret_val = new

    def structural_eq(  # noqa: D102
        self, other: Statement, memo: dict[vr.VariableReference, vr.VariableReference]
    ) -> bool:
        return (
            isinstance(other, PandasDataframeStatement)
            and self.ret_val.structural_eq(other.ret_val, memo)
            and self._csv_string == other._csv_string  # noqa: SLF001
            and self._fuzzer.grammar == other._fuzzer.grammar  # noqa: SLF001
        )

    def structural_hash(  # noqa: D102
        self, memo: dict[vr.VariableReference, int]
    ) -> int:
        return hash(
            (
                self.ret_val.structural_hash(memo),
                self._csv_string,
                self._fuzzer.grammar,
            )
        )
