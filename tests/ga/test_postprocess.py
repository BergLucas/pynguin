#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
from unittest import mock
from unittest.mock import MagicMock
from unittest.mock import call

import pytest

import pynguin.ga.postprocess as pp
import pynguin.ga.testcasechromosome as tcc
import pynguin.testcase.defaulttestcase as dtc
import pynguin.testcase.statement as stmt

from pynguin.analyses.module import ModuleTestCluster
from pynguin.assertion.assertion import ExceptionAssertion
from pynguin.utils.orderedset import OrderedSet


def test_not_failing():
    trunc = pp.ExceptionTruncation()
    test_case = MagicMock()
    chromosome = MagicMock(test_case=test_case)
    chromosome.is_failing.return_value = False
    trunc.visit_test_case_chromosome(chromosome)
    test_case.chop.assert_not_called()


def test_simple_chop():
    trunc = pp.ExceptionTruncation()
    test_case = MagicMock()
    chromosome = MagicMock(test_case=test_case)
    chromosome.is_failing.return_value = True
    chromosome.get_last_mutatable_statement.return_value = 42
    trunc.visit_test_case_chromosome(chromosome)
    test_case.chop.assert_called_once_with(42)


def test_suite_chop():
    trunc = pp.ExceptionTruncation()
    chromosome = MagicMock()
    suite = MagicMock(test_case_chromosomes=[chromosome, chromosome])
    trunc.visit_test_suite_chromosome(suite)
    chromosome.accept.assert_has_calls([call(trunc), call(trunc)])


def test_suite_assertion_minimization():
    ass_min = pp.AssertionMinimization()
    chromosome = MagicMock()
    suite = MagicMock(test_case_chromosomes=[chromosome, chromosome])
    ass_min.visit_test_suite_chromosome(suite)
    chromosome.accept.assert_has_calls([call(ass_min), call(ass_min)])


def test_test_case_assertion_minimization(default_test_case):
    ass_min = pp.AssertionMinimization()
    statement = stmt.IntPrimitiveStatement(default_test_case)

    assertion_1 = MagicMock(
        checked_instructions=[MagicMock(lineno=1), MagicMock(lineno=2)]
    )
    assertion_2 = MagicMock(checked_instructions=[MagicMock(lineno=1)])

    statement.add_assertion(assertion_1)
    statement.add_assertion(assertion_2)
    default_test_case.add_statement(statement)

    chromosome = tcc.TestCaseChromosome(test_case=default_test_case)
    ass_min.visit_test_case_chromosome(chromosome)

    assert ass_min.remaining_assertions == OrderedSet([assertion_1])
    assert ass_min.deleted_assertions == OrderedSet([assertion_2])
    assert default_test_case.get_assertions() == [assertion_1]


def test_test_case_assertion_minimization_does_not_remove_exception_assertion(
    default_test_case,
):
    ass_min = pp.AssertionMinimization()
    statement = stmt.IntPrimitiveStatement(default_test_case)

    assertion_1 = MagicMock(
        checked_instructions=[MagicMock(lineno=1), MagicMock(lineno=2)]
    )
    assertion_2 = MagicMock(
        spec=ExceptionAssertion, checked_instructions=[MagicMock(lineno=1)]
    )

    statement.add_assertion(assertion_1)
    statement.add_assertion(assertion_2)
    default_test_case.add_statement(statement)

    chromosome = tcc.TestCaseChromosome(test_case=default_test_case)
    ass_min.visit_test_case_chromosome(chromosome)

    assert ass_min.remaining_assertions == OrderedSet([assertion_1, assertion_2])
    assert ass_min.deleted_assertions == OrderedSet()
    assert default_test_case.get_assertions() == [assertion_1, assertion_2]


def test_test_case_assertion_minimization_does_not_remove_empty_assertion(
    default_test_case,
):
    ass_min = pp.AssertionMinimization()
    statement = stmt.IntPrimitiveStatement(default_test_case)

    assertion_1 = MagicMock(checked_instructions=[])

    statement.add_assertion(assertion_1)
    default_test_case.add_statement(statement)

    chromosome = tcc.TestCaseChromosome(test_case=default_test_case)
    ass_min.visit_test_case_chromosome(chromosome)

    assert ass_min.remaining_assertions == OrderedSet([assertion_1])
    assert ass_min.deleted_assertions == OrderedSet()
    assert default_test_case.get_assertions() == [assertion_1]


def test_test_case_postprocessor_suite():
    dummy_visitor = MagicMock()
    tcpp = pp.TestCasePostProcessor([dummy_visitor])
    chromosome = MagicMock()
    suite = MagicMock(test_case_chromosomes=[chromosome, chromosome])
    tcpp.visit_test_suite_chromosome(suite)
    chromosome.accept.assert_has_calls([call(tcpp), call(tcpp)])


def test_test_case_postprocessor_test():
    dummy_visitor = MagicMock()
    tcpp = pp.TestCasePostProcessor([dummy_visitor])
    test_case = MagicMock()
    test_chromosome = MagicMock(test_case=test_case)
    tcpp.visit_test_case_chromosome(test_chromosome)
    test_case.accept.assert_has_calls([call(dummy_visitor)])


def test_unused_primitives_visitor():
    remover_function = MagicMock()
    primitive_remover = pp.UnusedPrimitiveOrCollectionStatementRemover(
        {stmt.IntPrimitiveStatement: remover_function}
    )
    visitor = pp.UnusedStatementsTestCaseVisitor(primitive_remover)
    test_case = MagicMock()
    statement = stmt.IntPrimitiveStatement(test_case)
    test_case.statements = [statement]
    visitor.visit_default_test_case(test_case)
    assert remover_function.call_count == 1


# TODO(fk) replace with ast_to_stmt
def test_remove_integration(constructor_mock):
    cluster = ModuleTestCluster(0)
    test_case = dtc.DefaultTestCase(cluster)
    test_case.add_statement(stmt.IntPrimitiveStatement(test_case))
    test_case.add_statement(stmt.FloatPrimitiveStatement(test_case))
    int0 = stmt.IntPrimitiveStatement(test_case)
    test_case.add_statement(int0)
    list0 = stmt.ListStatement(
        test_case, cluster.type_system.convert_type_hint(list[int]), [int0.ret_val]
    )
    test_case.add_statement(list0)
    float0 = stmt.FloatPrimitiveStatement(test_case)
    test_case.add_statement(float0)
    ctor0 = stmt.ConstructorStatement(
        test_case, constructor_mock, {"foo": float0.ret_val, "bar": list0.ret_val}
    )
    test_case.add_statement(ctor0)
    assert test_case.size() == 6
    visitor = pp.UnusedStatementsTestCaseVisitor(
        pp.UnusedPrimitiveOrCollectionStatementRemover(pp.BUILTIN_REMOVER_FUNCTIONS)
    )
    test_case.accept(visitor)
    assert test_case.statements == [int0, list0, float0, ctor0]


@pytest.mark.parametrize(
    "statement_type, args",
    [
        (stmt.IntPrimitiveStatement, (MagicMock(),)),
        (stmt.FloatPrimitiveStatement, (MagicMock(),)),
        (stmt.StringPrimitiveStatement, (MagicMock(),)),
        (stmt.BytesPrimitiveStatement, (MagicMock(),)),
        (stmt.BooleanPrimitiveStatement, (MagicMock(),)),
        (stmt.EnumPrimitiveStatement, (MagicMock(), MagicMock(), 0)),
        (stmt.NoneStatement, (MagicMock(),)),
        (stmt.ConstructorStatement, (MagicMock(), MagicMock())),
        (stmt.MethodStatement, (MagicMock(), MagicMock(), MagicMock())),
        (stmt.FunctionStatement, (MagicMock(), MagicMock())),
        (stmt.ListStatement, (MagicMock(), MagicMock(), MagicMock())),
        (stmt.SetStatement, (MagicMock(), MagicMock(), MagicMock())),
        (stmt.TupleStatement, (MagicMock(), MagicMock(), MagicMock())),
        (stmt.DictStatement, (MagicMock(), MagicMock(), MagicMock())),
    ],
)
def test_all_primitive_statements(statement_type, args):
    remover_function = MagicMock()
    primitive_remover = pp.UnusedPrimitiveOrCollectionStatementRemover(
        {statement_type: remover_function}
    )
    test_case = args[0]
    statement = statement_type(*args)
    test_case.statements = [statement]
    primitive_remover.delete_statements_indexes([statement])
    remover_function.assert_called_once()


@pytest.mark.parametrize(
    "statement_type, args",
    [
        (stmt.FieldStatement, (MagicMock(), MagicMock(), MagicMock())),
        (stmt.AssignmentStatement, (MagicMock(), MagicMock(), MagicMock())),
    ],
)
def test_not_implemented_statements(statement_type, args):
    primitive_remover = pp.UnusedPrimitiveOrCollectionStatementRemover(
        pp.BUILTIN_REMOVER_FUNCTIONS
    )
    test_case = args[0]
    statement = statement_type(*args)
    test_case.statements = [statement]
    with pytest.raises(NotImplementedError):
        primitive_remover.delete_statements_indexes([statement])
