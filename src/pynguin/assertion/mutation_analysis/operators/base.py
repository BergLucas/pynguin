#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019-2023 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides base classes for mutation operators.

Based on https://github.com/se2p/mutpy-pynguin/blob/main/mutpy/operators/base.py
and integrated in Pynguin.
"""
from __future__ import annotations

import abc
import ast
import copy

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import TypeVar


if TYPE_CHECKING:
    import types

    from collections.abc import Callable
    from collections.abc import Generator
    from collections.abc import Iterable


def fix_lineno(node: ast.AST) -> None:
    """Fix the line number of a node if it is not set.

    Args:
        node: The node to fix.
    """
    parent = node.parent  # type: ignore[attr-defined]
    if not hasattr(node, "lineno") and parent is not None and hasattr(parent, "lineno"):
        parent_lineno = parent.lineno
        node.lineno = parent_lineno


def fix_node_internals(old_node: ast.AST, new_node: ast.AST) -> None:
    """Fix the internals of a node.

    Args:
        old_node: The old node.
        new_node: The new node.
    """
    if not hasattr(new_node, "parent"):
        old_node_children = old_node.children  # type: ignore[attr-defined]
        old_node_parent = old_node.parent  # type: ignore[attr-defined]
        new_node.children = old_node_children  # type: ignore[attr-defined]
        new_node.parent = old_node_parent  # type: ignore[attr-defined]

    if not hasattr(new_node, "lineno") and hasattr(old_node, "lineno"):
        old_node_lineno = old_node.lineno
        new_node.lineno = old_node_lineno

    if hasattr(old_node, "marker"):
        old_node_marker = old_node.marker
        new_node.marker = old_node_marker  # type: ignore[attr-defined]


def set_lineno(node: ast.AST, lineno: int) -> None:
    """Set the line number of a node.

    Args:
        node: The node to set the line number for.
        lineno: The line number to set.
    """
    for child_node in ast.walk(node):
        if hasattr(child_node, "lineno"):
            child_node.lineno = lineno


T = TypeVar("T", bound=ast.AST)


def shift_lines(nodes: list[T], shift_by: int = 1) -> None:
    """Shift the line numbers of a list of nodes.

    Args:
        nodes: The nodes to shift.
        shift_by: The amount to shift by.
    """
    for node in nodes:
        ast.increment_lineno(node, shift_by)


@dataclass
class Mutation:
    """Represents a mutation."""

    node: ast.AST
    replacement_node: ast.AST
    operator: type[MutationOperator]
    visitor_name: str


def copy_node(node: T) -> T:
    """Copy a node.

    Args:
        node: The node to copy.

    Returns:
        The copied node.
    """
    parent = node.parent  # type: ignore[attr-defined]
    return copy.deepcopy(
        node,
        memo={
            id(parent): parent,
        },
    )


class MutationOperator:
    """A class that represents a mutation operator."""

    @classmethod
    def mutate(
        cls,
        node: T,
        module: types.ModuleType,
        only_mutation: Mutation | None = None,
    ) -> Generator[tuple[Mutation, ast.AST], None, None]:
        """Mutate a node.

        Args:
            node: The node to mutate.
            module: The module to use.
            only_mutation: The mutation to apply.

        Yields:
            A tuple containing the mutation and the mutated node.
        """
        operator = cls(module, only_mutation)

        for (
            current_node,
            replacement_node,
            mutated_node,
            visitor_name,
        ) in operator.visit(node):
            yield Mutation(
                current_node, replacement_node, cls, visitor_name
            ), mutated_node

    def __init__(
        self,
        module: types.ModuleType,
        only_mutation: Mutation | None,
    ) -> None:
        """Initializes the operator.

        Args:
            module: The module to use.
            only_mutation: The mutation to apply.
        """
        self.module = module
        self.only_mutation = only_mutation

    def visit(
        self, node: T
    ) -> Generator[tuple[ast.AST, ast.AST, ast.AST, str], None, None]:
        """Visit a node.

        Args:
            node: The node to visit.

        Yields:
            A tuple containing the current node, the mutated node, and the visitor name.
        """
        node_children = node.children  # type: ignore[attr-defined]

        if (
            self.only_mutation
            and self.only_mutation.node != node
            and self.only_mutation.node not in node_children
        ):
            return

        fix_lineno(node)

        for visitor in self._find_visitors(node):
            if (
                self.only_mutation is None
                or (
                    self.only_mutation.node == node
                    and self.only_mutation.visitor_name == visitor.__name__
                )
            ) and (mutated_node := visitor(node)) is not None:
                fix_node_internals(node, mutated_node)
                ast.fix_missing_locations(mutated_node)

                yield node, mutated_node, mutated_node, visitor.__name__

        yield from self._generic_visit(node)

    def _generic_visit(
        self, node: ast.AST
    ) -> Generator[tuple[ast.AST, ast.AST, ast.AST, str], None, None]:
        for field, old_value in ast.iter_fields(node):
            generator: Iterable[tuple[ast.AST, ast.AST, str]]
            if isinstance(old_value, list):
                generator = self._generic_visit_list(old_value)
            elif isinstance(old_value, ast.AST):
                generator = self._generic_visit_real_node(node, field, old_value)
            else:
                generator = ()

            for current_node, replacement_node, visitor_name in generator:
                yield current_node, replacement_node, node, visitor_name

    def _generic_visit_list(
        self, old_value: list
    ) -> Generator[tuple[ast.AST, ast.AST, str], None, None]:
        for position, value in enumerate(old_value.copy()):
            if isinstance(value, ast.AST):
                for (
                    current_node,
                    replacement_node,
                    mutated_node,
                    visitor_name,
                ) in self.visit(value):
                    old_value[position] = mutated_node
                    yield current_node, replacement_node, visitor_name

                old_value[position] = value

    def _generic_visit_real_node(
        self, node: ast.AST, field: str, old_value: ast.AST
    ) -> Generator[tuple[ast.AST, ast.AST, str], None, None]:
        for current_node, replacement_node, mutated_node, visitor_name in self.visit(
            old_value
        ):
            setattr(node, field, mutated_node)
            yield current_node, replacement_node, visitor_name

        setattr(node, field, old_value)

    def _find_visitors(self, node: T) -> list[Callable[[T], ast.AST | None]]:
        node_name = node.__class__.__name__
        method_prefix = f"mutate_{node_name}"
        return [
            visitor
            for attr in dir(self)
            if attr.startswith(method_prefix)
            and callable(visitor := getattr(self, attr))
        ]


class AbstractUnaryOperatorDeletion(abc.ABC, MutationOperator):
    """An abstract class that mutates unary operators by deleting them."""

    @abc.abstractmethod
    def get_operator_type(self) -> type:
        """Get the operator type.

        Returns:
            The operator type.
        """

    def mutate_UnaryOp(self, node: ast.UnaryOp) -> ast.expr | None:  # noqa: N802
        """Mutate a unary operator.

        Args:
            node: The node to mutate.

        Returns:
            The mutated node, or None if the node should not be mutated.
        """
        if not isinstance(node.op, self.get_operator_type()):
            return None

        return node.operand
