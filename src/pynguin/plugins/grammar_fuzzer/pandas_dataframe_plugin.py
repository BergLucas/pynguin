#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides a plugin to generate Pandas dataframes as test data."""
from argparse import ArgumentParser
from argparse import Namespace

import pandas as pd

import pynguin.testcase.statement as stmt
import pynguin.utils.generic.genericaccessibleobject as gao

from pynguin.analyses.typesystem import Instance
from pynguin.analyses.typesystem import ProperType
from pynguin.analyses.typesystem import TupleType
from pynguin.ga.postprocess import UnusedPrimitiveOrCollectionStatementRemoverFunction
from pynguin.ga.postprocess import remove_collection_or_primitive
from pynguin.plugins.grammar_fuzzer.csv import create_csv_grammar
from pynguin.plugins.grammar_fuzzer.csv_plugin import (
    GrammarBasedFileLikeObjectStatement,
)
from pynguin.plugins.grammar_fuzzer.csv_plugin import (
    transform_grammar_based_file_like_object_statement,
)
from pynguin.plugins.grammar_fuzzer.fuzzer import GrammarFuzzer
from pynguin.testcase.statement_to_ast import StatementToAstTransformerFunction
from pynguin.testcase.testcase import TestCase
from pynguin.testcase.testfactory import SupportedTypes
from pynguin.testcase.testfactory import TestFactory
from pynguin.testcase.testfactory import VariableGenerator
from pynguin.testcase.variablereference import VariableReference
from pynguin.utils import randomness


NAME = "pandas_dataframe_fuzzer"

pandas_dataframe_weight: float = 0.0
pandas_dataframe_min_columns_number: int = 0
pandas_dataframe_max_columns_number: int = 0
pandas_dataframe_min_field_length: int = 0
pandas_dataframe_max_field_length: int = 0
pandas_dataframe_min_rows_number: int = 0
pandas_dataframe_max_rows_number: int = 0
pandas_dataframe_min_non_terminal: int = 0
pandas_dataframe_max_non_terminal: int = 0


def parser_hook(parser: ArgumentParser) -> None:  # noqa: D103
    parser.add_argument(
        "--pandas_dataframe_weight",
        type=float,
        default=0.0,
        help="""Weight to use a Pandas dataframe object as parameter type."""
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
        default=0,
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
        "--pandas_dataframe_min_non_terminal",
        type=int,
        default=10,
        help="Minimum number of non-terminal symbols in the grammar",
    )
    parser.add_argument(
        "--pandas_dataframe_max_non_terminal",
        type=int,
        default=25,
        help="Maximum number of non-terminal symbols in the grammar",
    )


def configuration_hook(plugin_config: Namespace) -> None:  # noqa: D103
    global pandas_dataframe_weight  # noqa: PLW0603
    global pandas_dataframe_min_columns_number  # noqa: PLW0603
    global pandas_dataframe_max_columns_number  # noqa: PLW0603
    global pandas_dataframe_min_field_length  # noqa: PLW0603
    global pandas_dataframe_max_field_length  # noqa: PLW0603
    global pandas_dataframe_min_rows_number  # noqa: PLW0603
    global pandas_dataframe_max_rows_number  # noqa: PLW0603
    global pandas_dataframe_min_non_terminal  # noqa: PLW0603
    global pandas_dataframe_max_non_terminal  # noqa: PLW0603

    pandas_dataframe_weight = plugin_config.pandas_dataframe_weight
    pandas_dataframe_min_columns_number = (
        plugin_config.pandas_dataframe_min_columns_number
    )
    pandas_dataframe_max_columns_number = (
        plugin_config.pandas_dataframe_max_columns_number
    )
    pandas_dataframe_min_field_length = plugin_config.pandas_dataframe_min_field_length
    pandas_dataframe_max_field_length = plugin_config.pandas_dataframe_min_field_length
    pandas_dataframe_min_rows_number = plugin_config.pandas_dataframe_min_rows_number
    pandas_dataframe_max_rows_number = plugin_config.pandas_dataframe_max_rows_number
    pandas_dataframe_min_non_terminal = plugin_config.pandas_dataframe_min_non_terminal
    pandas_dataframe_max_non_terminal = plugin_config.pandas_dataframe_max_non_terminal


def ast_transformer_hook(  # noqa: D103
    transformer_functions: dict[type, StatementToAstTransformerFunction]
) -> None:
    transformer_functions[GrammarBasedFileLikeObjectStatement] = (
        transform_grammar_based_file_like_object_statement
    )


def types_hook() -> list[type]:  # noqa: D103
    return [pd.DataFrame]


def statement_remover_hook(  # noqa: D103
    remover_functions: dict[type, UnusedPrimitiveOrCollectionStatementRemoverFunction]
) -> None:
    remover_functions[GrammarBasedFileLikeObjectStatement] = (
        remove_collection_or_primitive
    )


def variable_generator_hook(  # noqa: D103
    generators: dict[VariableGenerator, float]
) -> None:
    generators[PandasVariableGenerator()] = pandas_dataframe_weight


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
        )

        csv_grammar_fuzzer = GrammarFuzzer(
            csv_grammar,
            pandas_dataframe_min_non_terminal,
            pandas_dataframe_max_non_terminal,
        )

        string_io_ret = test_case.add_variable_creating_statement(
            GrammarBasedFileLikeObjectStatement(test_case, csv_grammar_fuzzer),
            position,
        )
        string_io_ret.distance = recursion_depth

        dataframe_accessible = gao.GenericFunction(
            pd.read_csv,
            test_case.test_cluster.type_system.infer_type_info(pd.read_csv),
        )

        dataframe_statement = stmt.FunctionStatement(
            test_case, dataframe_accessible, {"filepath_or_buffer": string_io_ret}
        )

        dataframe_ret = test_case.add_variable_creating_statement(
            dataframe_statement, position + 1
        )
        dataframe_ret.distance = recursion_depth

        return dataframe_ret
