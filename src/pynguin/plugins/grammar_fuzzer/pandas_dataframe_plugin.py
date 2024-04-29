import io

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
from pynguin.plugins.grammar_fuzzer.fuzzer import GrammarFuzzer
from pynguin.plugins.grammar_fuzzer.csv_plugin import GrammarBasedStringPrimitiveStatement
from pynguin.testcase.statement_to_ast import StatementToAstTransformerFunction
from pynguin.testcase.statement_to_ast import transform_primitive_statement
from pynguin.testcase.testcase import TestCase
from pynguin.testcase.testfactory import AbstractVariableGenerator
from pynguin.testcase.testfactory import SupportedTypes
from pynguin.testcase.testfactory import TestFactory
from pynguin.testcase.variablereference import VariableReference
from pynguin.utils import randomness


NAME = "pandas_fuzzer"

dataframe_weight: float = 0


def parser_hook(parser: ArgumentParser) -> None:  # noqa: D103
    parser.add_argument(
        "--dataframe_weight",
        type=float,
        default=0,
        help="""Weight to use a Pandas dataframe object as parameter type
        during test generation. Expects values > 0""",
    )


def configuration_hook(plugin_config: Namespace) -> None:  # noqa: D103
    global dataframe_weight  # noqa: PLW0603
    dataframe_weight = plugin_config.dataframe_weight


def ast_transformer_hook(  # noqa: D103
    transformer_functions: dict[type, StatementToAstTransformerFunction]
) -> None:
    transformer_functions[GrammarBasedStringPrimitiveStatement] = (
        transform_primitive_statement
    )


def statement_remover_hook(  # noqa: D103
    remover_functions: dict[type, UnusedPrimitiveOrCollectionStatementRemoverFunction]
) -> None:
    remover_functions[GrammarBasedStringPrimitiveStatement] = (
        remove_collection_or_primitive
    )


def variable_generator_hook(  # noqa: D103
    generators: dict[AbstractVariableGenerator, float]
) -> None:
    generators[PandasVariableGenerator()] = dataframe_weight


class DataframeSupportedTypes(SupportedTypes):
    """Supported types for Pandas dataframes."""

    def visit_instance(self, left: Instance) -> bool:  # noqa: D102
        try:
            return issubclass(left.type.raw_type, pd.DataFrame)
        except TypeError:
            return False

    def visit_tuple_type(self, left: TupleType) -> bool:  # noqa: D102
        return False


dataframe_supported_types = DataframeSupportedTypes()


class PandasVariableGenerator(AbstractVariableGenerator):
    """A Pandas dataframes variable generator."""

    @property
    def supported_types(self) -> SupportedTypes:  # noqa: D102
        return dataframe_supported_types

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
        string_io_type_info = test_case.test_cluster.type_system.alias_to_type_info(
            "io.StringIO"
        )

        assert string_io_type_info is not None

        string_io_accessible = gao.GenericConstructor(
            string_io_type_info,
            test_case.test_cluster.type_system.infer_type_info(io.StringIO),
        )

        csv_grammar = create_csv_grammar(
            randomness.next_int(1, 10),
            min_field_length=3,
        )

        ref = test_factory.add_primitive(
            test_case,
            GrammarBasedStringPrimitiveStatement(
                test_case, GrammarFuzzer(csv_grammar, 0, 100)
            ),
            position,
        )

        string_io_statement = stmt.ConstructorStatement(
            test_case, string_io_accessible, {"initial_value": ref}
        )
        string_io_ret = test_case.add_variable_creating_statement(
            string_io_statement, position + 1
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
            dataframe_statement, position + 2
        )
        dataframe_ret.distance = recursion_depth

        return dataframe_ret
