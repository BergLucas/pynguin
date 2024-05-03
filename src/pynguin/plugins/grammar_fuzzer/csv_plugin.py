#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides a plugin to generate CSV file-like object as test data."""
from __future__ import annotations

import ast
import io

from copy import deepcopy
from typing import TYPE_CHECKING

import pynguin.utils.ast_util as au
import pynguin.utils.generic.genericaccessibleobject as gao

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

    import pynguin.utils.namingscope as ns

    from pynguin.analyses.typesystem import Instance
    from pynguin.analyses.typesystem import ProperType
    from pynguin.analyses.typesystem import TupleType
    from pynguin.testcase.statement import Statement
    from pynguin.testcase.testcase import TestCase
    from pynguin.testcase.variablereference import VariableReference

NAME = "csv_fuzzer"

csv_weight: float = 0.0
csv_min_columns_number: int = 0
csv_max_columns_number: int = 0
csv_min_field_length: int = 0
csv_max_field_length: int = 0
csv_min_rows_number: int = 0
csv_max_rows_number: int = 0
csv_min_non_terminal: int = 0
csv_max_non_terminal: int = 0


def parser_hook(parser: ArgumentParser) -> None:  # noqa: D103
    parser.add_argument(
        "--csv_weight",
        type=float,
        default=100.0,
        help="""Weight to use a CSV file-like object as parameter type."""
        """Expects values > 0""",
    )
    parser.add_argument(
        "--csv_min_columns_number",
        type=int,
        default=1,
        help="Minimum number of columns in the CSV file-like object",
    )
    parser.add_argument(
        "--csv_max_columns_number",
        type=int,
        default=5,
        help="Maximum number of columns in the CSV file-like object",
    )
    parser.add_argument(
        "--csv_min_field_length",
        type=int,
        default=0,
        help="Minimum length of a field in the CSV file-like object",
    )
    parser.add_argument(
        "--csv_max_field_length",
        type=int,
        default=10,
        help="Maximum length of a field in the CSV file-like object",
    )
    parser.add_argument(
        "--csv_min_rows_number",
        type=int,
        default=1,
        help="Minimum number of rows in the CSV file-like object",
    )
    parser.add_argument(
        "--csv_max_rows_number",
        type=int,
        default=5,
        help="Maximum number of rows in the CSV file-like object",
    )
    parser.add_argument(
        "--csv_min_non_terminal",
        type=int,
        default=10,
        help="Minimum number of non-terminal symbols in the grammar",
    )
    parser.add_argument(
        "--csv_max_non_terminal",
        type=int,
        default=25,
        help="Maximum number of non-terminal symbols in the grammar",
    )


def configuration_hook(plugin_config: Namespace) -> None:  # noqa: D103
    global csv_weight  # noqa: PLW0603
    global csv_min_columns_number  # noqa: PLW0603
    global csv_max_columns_number  # noqa: PLW0603
    global csv_min_field_length  # noqa: PLW0603
    global csv_max_field_length  # noqa: PLW0603
    global csv_min_rows_number  # noqa: PLW0603
    global csv_max_rows_number  # noqa: PLW0603
    global csv_min_non_terminal  # noqa: PLW0603
    global csv_max_non_terminal  # noqa: PLW0603

    csv_weight = plugin_config.csv_weight
    csv_min_columns_number = plugin_config.csv_min_columns_number
    csv_max_columns_number = plugin_config.csv_max_columns_number
    csv_min_field_length = plugin_config.csv_min_field_length
    csv_max_field_length = plugin_config.csv_max_field_length
    csv_min_rows_number = plugin_config.csv_min_rows_number
    csv_max_rows_number = plugin_config.csv_max_rows_number
    csv_min_non_terminal = plugin_config.csv_min_non_terminal
    csv_max_non_terminal = plugin_config.csv_max_non_terminal


def types_hook() -> list[type | tuple[type, str]]:  # noqa: D103
    return [(io.StringIO, "io")]


def ast_transformer_hook(  # noqa: D103
    transformer_functions: dict[type, StatementToAstTransformerFunction]
) -> None:
    transformer_functions[GrammarBasedFileLikeObjectStatement] = (
        transform_grammar_based_file_like_object_statement
    )


def statement_remover_hook(  # noqa: D103
    remover_functions: dict[type, UnusedPrimitiveOrCollectionStatementRemoverFunction]
) -> None:
    remover_functions[GrammarBasedFileLikeObjectStatement] = (
        remove_collection_or_primitive
    )


def variable_generator_hook(  # noqa: D103
    generators: dict[VariableGenerator, float]
) -> None:
    generators[CsvVariableGenerator()] = csv_weight


def transform_grammar_based_file_like_object_statement(
    stmt: GrammarBasedFileLikeObjectStatement,
    module_aliases: ns.AbstractNamingScope,
    variable_names: ns.AbstractNamingScope,
    store_call_return: bool,  # noqa: FBT001
) -> ast.stmt:
    """Transform a grammar based file-like object statement to an AST node.

    Args:
        stmt: The statement to transform.
        module_aliases: A naming scope for module alias names.
        variable_names: A naming scope for variable names.
        store_call_return: Should the result of a call be stored in a variable?

    Returns:
        The AST node.
    """
    accessible_object = stmt.accessible_object()
    owner = accessible_object.owner
    assert owner
    call = ast.Call(
        func=ast.Attribute(
            attr=owner.name,
            ctx=ast.Load(),
            value=create_module_alias(
                accessible_object.exporter_module, module_aliases
            ),
        ),
        args=[ast.Constant(value=stmt.csv_string)],
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


class _CsvSupportedTypes(SupportedTypes):
    """Supported types for CSV files."""

    def visit_instance(self, left: Instance) -> bool:
        try:
            return issubclass(left.type.raw_type, io.TextIOBase)
        except TypeError:
            return False

    def visit_tuple_type(self, left: TupleType) -> bool:
        return False


csv_supported_types = _CsvSupportedTypes()


class CsvVariableGenerator(VariableGenerator):
    """A CSV variable generator."""

    @property
    def supported_types(self) -> SupportedTypes:  # noqa: D102
        return csv_supported_types

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
            csv_min_columns_number, csv_max_columns_number
        )

        csv_grammar = create_csv_grammar(
            columns_number=columns_number,
            min_field_length=csv_min_field_length,
            max_field_length=csv_max_field_length,
            min_rows_number=csv_min_rows_number,
            max_rows_number=csv_max_rows_number,
        )

        csv_grammar_fuzzer = GrammarFuzzer(
            csv_grammar,
            csv_min_non_terminal,
            csv_max_non_terminal,
        )

        string_io_ret = test_case.add_variable_creating_statement(
            GrammarBasedFileLikeObjectStatement(test_case, csv_grammar_fuzzer),
            position,
        )
        string_io_ret.distance = recursion_depth

        return string_io_ret


class GrammarBasedFileLikeObjectStatement(VariableCreatingStatement):
    """A statement that creates a grammar based file-like object."""

    def __init__(
        self,
        test_case: TestCase,
        fuzzer: GrammarFuzzer,
        derivation_tree: GrammarDerivationTree | None = None,
    ) -> None:
        """Create a new grammar based file-like object statement.

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

        string_io_type_info = test_case.test_cluster.type_system.to_type_info(
            io.StringIO
        )

        assert string_io_type_info is not None

        self._string_io_accessible = gao.GenericConstructor(
            string_io_type_info,
            test_case.test_cluster.type_system.infer_type_info(io.StringIO),
            exporter_module="io",
        )

        super().__init__(
            test_case,
            vr.CallBasedVariableReference(test_case, self._string_io_accessible),
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
    ) -> GrammarBasedFileLikeObjectStatement:
        return GrammarBasedFileLikeObjectStatement(
            test_case, self._fuzzer, deepcopy(self._derivation_tree)
        )

    def accessible_object(self) -> gao.GenericAccessibleObject:  # noqa: D102
        return self._string_io_accessible

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
            isinstance(other, GrammarBasedFileLikeObjectStatement)
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
