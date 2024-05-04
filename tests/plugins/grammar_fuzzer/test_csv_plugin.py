#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
import importlib
import threading

from pynguin import configuration as config
from pynguin.instrumentation.machinery import install_import_hook
from pynguin.plugins.grammar_fuzzer.csv import create_csv_grammar
from pynguin.plugins.grammar_fuzzer.csv_plugin import (
    GrammarBasedFileLikeObjectStatement,
)
from pynguin.plugins.grammar_fuzzer.csv_plugin import (
    transform_grammar_based_file_like_object_statement,
)
from pynguin.plugins.grammar_fuzzer.fuzzer import GrammarFuzzer
from pynguin.testcase.execution import ExecutionTracer
from pynguin.testcase.execution import TestCaseExecutor
from pynguin.testcase.statement_to_ast import BUILTIN_TRANSFORMER_FUNCTIONS
from pynguin.testcase.statement_to_ast import StatementToAstTransformer
from pynguin.utils import randomness
from tests.testutils import create_source_from_ast


def test_grammar_based_file_like_object_statement_to_ast(
    module_aliases, var_names, default_test_case
):
    randomness.RNG.seed(42)

    csv_grammar = create_csv_grammar(
        columns_number=4,
        min_field_length=2,
        max_field_length=10,
        min_rows_number=1,
        max_rows_number=5,
    )
    csv_grammar_fuzzer = GrammarFuzzer(
        csv_grammar, min_non_terminal=50, max_non_terminal=100
    )
    file_like_object_stmt = GrammarBasedFileLikeObjectStatement(
        default_test_case, csv_grammar_fuzzer
    )
    ast_node = transform_grammar_based_file_like_object_statement(
        file_like_object_stmt, module_aliases, var_names, True
    )
    assert (
        create_source_from_ast(ast_node)
        == 'var_0 = module_0.StringIO(\'"i1VhdjzIGo","DVA73ae4z4","rT6fTipS6M","qhIBvql6L9"\\n"XhMZOjDpdv","aHsLf6SpUn","2VyESlyjPH","KVjX2AYH6b"\\n"f8M1D5kCJO","vl3b5CT7WG","TAUJyf2hlI","rXiEqGhOeS"\\n"X2XQhBR5vM","polsk4lSdP","US60PyaUDq","fZVku98c4P"\\n"l2d5L7Rkv7","sFDv0HVDBv","1M8h5kmxst","sVH4J9qRgw"\\n\')'
    )


def test_simple_execution(default_test_case):
    statement_transformer = StatementToAstTransformer(
        {
            **BUILTIN_TRANSFORMER_FUNCTIONS,
            GrammarBasedFileLikeObjectStatement: transform_grammar_based_file_like_object_statement,
        }
    )
    config.configuration.module_name = "tests.fixtures.accessibles.accessible"
    tracer = ExecutionTracer()
    tracer.current_thread_identifier = threading.current_thread().ident
    with install_import_hook(config.configuration.module_name, tracer):
        module = importlib.import_module(config.configuration.module_name)
        importlib.reload(module)
        csv_grammar = create_csv_grammar(3, 2)
        csv_grammar_fuzzer = GrammarFuzzer(csv_grammar)
        default_test_case.add_statement(
            GrammarBasedFileLikeObjectStatement(default_test_case, csv_grammar_fuzzer)
        )
        executor = TestCaseExecutor(tracer, statement_transformer)
        print(default_test_case.statements)
        assert not executor.execute(default_test_case).has_test_exceptions()
