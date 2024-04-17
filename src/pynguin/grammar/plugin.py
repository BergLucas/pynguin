import io

from argparse import ArgumentParser
from argparse import Namespace
from copy import deepcopy

import pynguin.testcase.statement as stmt
import pynguin.utils.generic.genericaccessibleobject as gao

from pynguin.analyses.typesystem import Instance
from pynguin.analyses.typesystem import ProperType
from pynguin.analyses.typesystem import TupleType
from pynguin.analyses.typesystem import UnionType
from pynguin.ga.postprocess import UnusedPrimitiveOrCollectionStatementRemoverFunction
from pynguin.ga.postprocess import remove_collection_or_primitive
from pynguin.grammar.csv import create_csv_grammar
from pynguin.grammar.fuzzer import GrammarDerivationTree
from pynguin.grammar.fuzzer import GrammarFuzzer
from pynguin.testcase.statement import StringPrimitiveStatement
from pynguin.testcase.statement_to_ast import StatementToAstTransformerFunction
from pynguin.testcase.statement_to_ast import transform_primitive_statement
from pynguin.testcase.testcase import TestCase
from pynguin.testcase.testfactory import AbstractVariableGenerator
from pynguin.testcase.testfactory import SupportedTypes
from pynguin.testcase.testfactory import TestFactory
from pynguin.testcase.variablereference import VariableReference
from pynguin.utils import randomness


NAME = "csv_fuzzer"

csv_weight: float = 0


def parser_hook(parser: ArgumentParser) -> None:  # noqa: D103
    parser.add_argument(
        "--csv_weight",
        type=float,
        default=0,
        help="""Weight to use a CSV file-like object as parameter type during test generation.
        Expects values > 0""",  # noqa: E501
    )


def configuration_hook(plugin_config: Namespace) -> None:  # noqa: D103
    global csv_weight  # noqa: PLW0603
    csv_weight = plugin_config.csv_weight


def ast_transformer_hook(  # noqa: D103
    transformer_functions: dict[type, StatementToAstTransformerFunction]
) -> None:
    transformer_functions[GrammarBasedStringPrimitiveStatement] = (
        transform_primitive_statement
    )


def statement_remover_hook(  # noqa: D103
    remover_functions: dict[type, UnusedPrimitiveOrCollectionStatementRemoverFunction]
) -> None:
    remover_functions[GrammarBasedStringPrimitiveStatement] = remove_collection_or_primitive


def variable_generator_hook(  # noqa: D103
    generators: dict[AbstractVariableGenerator, float]
) -> None:
    generators[CsvVariableGenerator()] = csv_weight


class CsvSupportedTypes(SupportedTypes):
    """Supported types for CSV files."""

    def visit_instance(self, left: Instance) -> bool:  # noqa: D102
        try:
            return issubclass(left.type.raw_type, io.TextIOBase)
        except TypeError:
            return False

    def visit_tuple_type(self, left: TupleType) -> bool:  # noqa: D102
        return False

    def visit_union_type(self, left: UnionType) -> bool:  # noqa: D102
        return any(item.accept(self) for item in left.items)


csv_supported_types = CsvSupportedTypes()


class CsvVariableGenerator(AbstractVariableGenerator):
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
        print("generate_variable")
        type_info = test_case.test_cluster.type_system.alias_to_type_info("io.StringIO")

        assert type_info is not None

        accessible = gao.GenericConstructor(
            type_info, test_case.test_cluster.type_system.infer_type_info(io.StringIO)
        )

        csv_grammar = create_csv_grammar(randomness.next_int(1, 10), min_field_length=3)

        ref = test_factory.add_primitive(
            test_case,
            GrammarBasedStringPrimitiveStatement(
                test_case, GrammarFuzzer(csv_grammar, 0, 100)
            ),
            position,
        )

        statement = stmt.ConstructorStatement(
            test_case, accessible, {"initial_value": ref}
        )
        ret = test_case.add_variable_creating_statement(statement, position + 1)
        ret.distance = recursion_depth

        return ret


class GrammarBasedStringPrimitiveStatement(StringPrimitiveStatement):
    """Primitive Statement that creates a grammar based String."""

    def __init__(  # noqa: D107
        self,
        test_case: TestCase,
        fuzzer: GrammarFuzzer,
        derivation_tree: GrammarDerivationTree | None = None,
    ) -> None:
        if derivation_tree is None:
            derivation_tree = fuzzer.create_tree()

        self._derivation_tree = derivation_tree
        self._fuzzer = fuzzer

        super().__init__(test_case, str(derivation_tree), None)

    def randomize_value(self) -> None:  # noqa: D102
        self._fuzzer.mutate_tree(self._derivation_tree)
        self._value = str(self._derivation_tree)

    def clone(  # noqa: D102
        self,
        test_case: TestCase,
        memo: dict[VariableReference, VariableReference],
    ) -> StringPrimitiveStatement:
        return GrammarBasedStringPrimitiveStatement(
            test_case, self._fuzzer, deepcopy(self._derivation_tree)
        )