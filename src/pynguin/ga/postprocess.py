#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides chromosome visitors to perform post-processing."""
from __future__ import annotations

import logging

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Protocol
from typing import TypeVar

import pynguin.ga.chromosomevisitor as cv
import pynguin.testcase.testcasevisitor as tcv
import pynguin.testcase.variablereference as vr

from pynguin.assertion.assertion import Assertion
from pynguin.assertion.assertion import ExceptionAssertion
from pynguin.testcase.statement import BooleanPrimitiveStatement
from pynguin.testcase.statement import BytesPrimitiveStatement
from pynguin.testcase.statement import ClassPrimitiveStatement
from pynguin.testcase.statement import ComplexPrimitiveStatement
from pynguin.testcase.statement import ConstructorStatement
from pynguin.testcase.statement import DictStatement
from pynguin.testcase.statement import EnumPrimitiveStatement
from pynguin.testcase.statement import FloatPrimitiveStatement
from pynguin.testcase.statement import FunctionStatement
from pynguin.testcase.statement import IntPrimitiveStatement
from pynguin.testcase.statement import ListStatement
from pynguin.testcase.statement import MethodStatement
from pynguin.testcase.statement import NoneStatement
from pynguin.testcase.statement import SetStatement
from pynguin.testcase.statement import Statement
from pynguin.testcase.statement import StringPrimitiveStatement
from pynguin.testcase.statement import TupleStatement
from pynguin.utils.orderedset import OrderedSet


if TYPE_CHECKING:
    import pynguin.ga.testcasechromosome as tcc
    import pynguin.ga.testsuitechromosome as tsc


class ExceptionTruncation(cv.ChromosomeVisitor):
    """Truncates test cases after an exception-raising statement."""

    def visit_test_suite_chromosome(  # noqa: D102
        self, chromosome: tsc.TestSuiteChromosome
    ) -> None:
        for test_case_chromosome in chromosome.test_case_chromosomes:
            test_case_chromosome.accept(self)

    def visit_test_case_chromosome(  # noqa: D102
        self, chromosome: tcc.TestCaseChromosome
    ) -> None:
        if chromosome.is_failing():
            chop_position = chromosome.get_last_mutatable_statement()
            if chop_position is not None:
                chromosome.test_case.chop(chop_position)


class AssertionMinimization(cv.ChromosomeVisitor):
    """Calculates the checked lines of each assertion.

    If an assertion does not cover new lines, it is removed from the resulting test
    case.
    """

    _logger = logging.getLogger(__name__)

    def __init__(self):  # noqa: D107
        self._remaining_assertions: OrderedSet[Assertion] = OrderedSet()
        self._deleted_assertions: OrderedSet[Assertion] = OrderedSet()
        self._checked_line_numbers: OrderedSet[int] = OrderedSet()

    @property
    def remaining_assertions(self) -> OrderedSet[Assertion]:
        """Provides a set of remaining assertions.

        Returns:
            The remaining assertions
        """
        return self._remaining_assertions

    @property
    def deleted_assertions(self) -> OrderedSet[Assertion]:
        """Provides a set of deleted assertions.

        Returns:
            The deleted assertions
        """
        return self._deleted_assertions

    def visit_test_suite_chromosome(  # noqa: D102
        self, chromosome: tsc.TestSuiteChromosome
    ) -> None:
        for test_case_chromosome in chromosome.test_case_chromosomes:
            test_case_chromosome.accept(self)

        self._logger.debug(
            f"Removed {len(self._deleted_assertions)} assertion(s) from "  # noqa: G004
            f"test suite that do not increase checked coverage",
        )

    def visit_test_case_chromosome(  # noqa: D102
        self, chromosome: tcc.TestCaseChromosome
    ) -> None:
        for stmt in chromosome.test_case.statements:
            to_remove: OrderedSet[Assertion] = OrderedSet()
            for assertion in stmt.assertions:
                new_checked_lines: OrderedSet[int] = OrderedSet()
                for instr in assertion.checked_instructions:
                    new_checked_lines.add(instr.lineno)  # type: ignore[arg-type]
                if (
                    # keep exception assertions to catch the exceptions
                    isinstance(assertion, ExceptionAssertion)
                    # keep assertions when they check "nothing", since this is
                    # more likely due to pyChecco's limitation, rather than an actual
                    # assertion that checks nothing at all
                    or not new_checked_lines
                    # keep assertions that increase checked coverage
                    or not new_checked_lines.issubset(self._checked_line_numbers)
                ):
                    self._checked_line_numbers.update(new_checked_lines)
                    self._remaining_assertions.add(assertion)
                else:
                    to_remove.add(assertion)
            for assertion in to_remove:
                stmt.assertions.remove(assertion)
                self._deleted_assertions.add(assertion)


class TestCasePostProcessor(cv.ChromosomeVisitor):
    """Applies all given visitors to the visited test cases."""

    def __init__(  # noqa: D107
        self, test_case_visitors: list[ModificationAwareTestCaseVisitor]
    ):
        self._test_case_visitors = test_case_visitors

    def visit_test_suite_chromosome(  # noqa: D102
        self, chromosome: tsc.TestSuiteChromosome
    ) -> None:
        for test_case_chromosome in chromosome.test_case_chromosomes:
            test_case_chromosome.accept(self)

    def visit_test_case_chromosome(  # noqa: D102
        self, chromosome: tcc.TestCaseChromosome
    ) -> None:
        for visitor in self._test_case_visitors:
            chromosome.test_case.accept(visitor)
            if (last_exec := chromosome.get_last_execution_result()) is not None:
                # We don't want to re-execute the test cases here, so we also remove
                # information about the deleted statements from the execution result.
                # TODO(fk) we could also re-execute, but with flakiness this could
                #  cause inconsistent results
                last_exec.delete_statement_data(visitor.deleted_statement_indexes)


class ModificationAwareTestCaseVisitor(tcv.TestCaseVisitor, ABC):
    """Visitor that keep information on modifications."""

    def __init__(self):  # noqa: D107
        self._deleted_statement_indexes: set[int] = set()

    @property
    def deleted_statement_indexes(self) -> set[int]:
        """Provides a set of deleted statement indexes.

        Returns:
            The deleted statement indexes
        """
        return self._deleted_statement_indexes


class UnusedStatementsTestCaseVisitor(ModificationAwareTestCaseVisitor):
    """Removes unused primitive and collection statements."""

    _logger = logging.getLogger(__name__)

    def __init__(self, primitive_remover: UnusedPrimitiveOrCollectionStatementRemover):
        """Create a new unused statement visitor.

        Args:
            primitive_remover: The primitive remover to use
        """
        super().__init__()
        self._primitive_remover = primitive_remover

    def visit_default_test_case(self, test_case) -> None:  # noqa: D102
        self._deleted_statement_indexes.clear()
        size_before = test_case.size()
        # Iterate over copy, to be able to modify original.
        deleted_statement_indexes = self._primitive_remover.delete_statements_indexes(
            list(test_case.statements)
        )
        self._logger.debug(
            "Removed %s unused primitives/collections from test case",
            size_before - test_case.size(),
        )
        self._deleted_statement_indexes.update(deleted_statement_indexes)


class UnusedPrimitiveOrCollectionStatementRemover:
    """Remove the unused primitive and collection statements."""

    def __init__(
        self,
        remover_functions: dict[
            type, UnusedPrimitiveOrCollectionStatementRemoverFunction
        ],
    ):
        """Create a new statement remover.

        Args:
            remover_functions: A dictionary that maps statement types to functions that
                remove unused statements.
        """
        self._remover_functions = remover_functions

    def delete_statements_indexes(self, statements: list[Statement]) -> set[int]:
        """Delete unused primitive or collection statements and returns their indexes.

        Args:
            statements: The statements to check

        Returns:
            The deleted statement indexes
        """
        used_references: set[vr.VariableReference] = set()
        deleted_statement_indexes: set[int] = set()

        for stmt in reversed(statements):
            try:
                statement_remover_function = self._remover_functions[type(stmt)]
            except KeyError as e:
                raise NotImplementedError(
                    f"Unknown statement type: {type(stmt)}"
                ) from e

            statement_remover_function(stmt, used_references, deleted_statement_indexes)

        return deleted_statement_indexes


S_contra = TypeVar("S_contra", bound=Statement, contravariant=True)


class UnusedPrimitiveOrCollectionStatementRemoverFunction(Protocol[S_contra]):
    """Protocol for removing unused primitive or collection statement."""

    @abstractmethod
    def __call__(
        self,
        stmt: S_contra,
        used_references: set[vr.VariableReference],
        deleted_statement_indexes: set[int],
    ) -> None:
        """Remove unused statements from the test case.

        Args:
            stmt: The statement to convert.
            used_references: The used references.
            deleted_statement_indexes: The deleted statement indexes.
        """


def remove_remaining(
    stmt: Statement,
    used_references: set[vr.VariableReference],
    deleted_statement_indexes: set[int],  # noqa: ARG001
) -> None:
    """Remove all remaining statements.

    Args:
        stmt: The statement to remove.
        used_references: The used references.
        deleted_statement_indexes: The deleted statement indexes.
    """
    used = stmt.get_variable_references()
    if stmt.ret_val is not None:
        used.discard(stmt.ret_val)
    used_references.update(used)


def remove_collection_or_primitive(
    stmt: Statement,
    used_references: set[vr.VariableReference],
    deleted_statement_indexes: set[int],
) -> None:
    """Remove collection or primitive statements.

    Args:
        stmt: The statement to remove.
        used_references: The used references.
        deleted_statement_indexes: The deleted statement indexes.
    """
    if stmt.ret_val in used_references:
        remove_remaining(stmt, used_references, deleted_statement_indexes)
    else:
        deleted_statement_indexes.add(stmt.get_position())
        stmt.test_case.remove_statement(stmt)


BUILTIN_REMOVER_FUNCTIONS: dict[
    type, UnusedPrimitiveOrCollectionStatementRemoverFunction
] = {
    IntPrimitiveStatement: remove_collection_or_primitive,
    FloatPrimitiveStatement: remove_collection_or_primitive,
    ComplexPrimitiveStatement: remove_collection_or_primitive,
    StringPrimitiveStatement: remove_collection_or_primitive,
    BytesPrimitiveStatement: remove_collection_or_primitive,
    BooleanPrimitiveStatement: remove_collection_or_primitive,
    EnumPrimitiveStatement: remove_collection_or_primitive,
    NoneStatement: remove_remaining,
    ClassPrimitiveStatement: remove_collection_or_primitive,
    ConstructorStatement: remove_remaining,
    MethodStatement: remove_remaining,
    FunctionStatement: remove_remaining,
    ListStatement: remove_collection_or_primitive,
    SetStatement: remove_collection_or_primitive,
    TupleStatement: remove_collection_or_primitive,
    DictStatement: remove_collection_or_primitive,
}
