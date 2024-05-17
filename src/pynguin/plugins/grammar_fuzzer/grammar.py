#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides classes to define a grammar and its rules."""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Protocol
from typing import TypeVar


if TYPE_CHECKING:
    from frozendict import frozendict


@dataclass(frozen=True)
class Grammar:
    """A grammar."""

    initial_rule: str
    rules: frozendict[str, GrammarRule]


T_co = TypeVar("T_co", covariant=True)


class GrammarRuleVisitor(Protocol[T_co]):
    """A visitor for grammar rules."""

    @abstractmethod
    def visit_constant(self, constant: Constant) -> T_co:
        """Visit a constant node.

        Args:
            constant (Constant): The constant node to visit.

        Returns:
            T: The result of visiting the constant node.
        """

    @abstractmethod
    def visit_sequence(self, sequence: Sequence) -> T_co:
        """Visit a sequence node.

        Args:
            sequence (Sequence): The sequence node to visit.

        Returns:
            T: The result of visiting the sequence node.
        """

    @abstractmethod
    def visit_rule_reference(self, rule_reference: RuleReference) -> T_co:
        """Visit a rule reference node.

        Args:
            rule_reference (RuleReference): The rule reference node to visit.

        Returns:
            T: The result of visiting the rule reference node.
        """

    @abstractmethod
    def visit_any_char(self, any_char: AnyChar) -> T_co:
        """Visit an any char node.

        Args:
            any_char (AnyChar): The any char node to visit.

        Returns:
            T: The result of visiting the any char node.
        """

    @abstractmethod
    def visit_choice(self, choice: Choice) -> T_co:
        """Visit a choice node.

        Args:
            choice (Choice): The choice node to visit.

        Returns:
            T: The result of visiting the choice node.
        """

    @abstractmethod
    def visit_repeat(self, repeat: Repeat) -> T_co:
        """Visit a repeat node.

        Args:
            repeat (Repeat): The repeat node to visit.

        Returns:
            T: The result of visiting the repeat node.
        """


class GrammarRule(Protocol):
    """A grammar rule."""

    @abstractmethod
    def accept(self, visitor: GrammarRuleVisitor[T_co]) -> T_co:
        """Accept a visitor.

        Args:
            visitor: The visitor to accept.

        Returns:
            The result of accepting the visitor.
        """


@dataclass(frozen=True)
class Constant(GrammarRule):
    """A constant value."""

    value: str

    def accept(self, visitor: GrammarRuleVisitor[T_co]) -> T_co:  # noqa: D102
        return visitor.visit_constant(self)


@dataclass(frozen=True)
class Sequence(GrammarRule):
    """A sequence of grammar rules."""

    rules: tuple[GrammarRule, ...]

    def accept(self, visitor: GrammarRuleVisitor[T_co]) -> T_co:  # noqa: D102
        return visitor.visit_sequence(self)


@dataclass(frozen=True)
class RuleReference(GrammarRule):
    """A reference to another rule."""

    name: str

    def accept(self, visitor: GrammarRuleVisitor[T_co]) -> T_co:  # noqa: D102
        return visitor.visit_rule_reference(self)


@dataclass(frozen=True)
class AnyChar(GrammarRule):
    """A rule that represents any character."""

    codes: tuple[int, ...]

    @classmethod
    def from_range(cls, min_code: int, max_code: int) -> AnyChar:
        """Create a rule that represents characters within a range.

        Args:
            min_code: The minimum code.
            max_code: The maximum code.

        Returns:
            A rule that represents characters within a range.
        """
        return cls(tuple(range(min_code, max_code)))

    @classmethod
    def printable(cls) -> AnyChar:
        """Return a rule that represents printable characters.

        Returns:
            A rule that represents printable characters.
        """
        return cls.from_range(32, 128)

    @classmethod
    def letters(cls) -> AnyChar:
        """Return a rule that represents letters.

        Returns:
            A rule that represents letters.
        """
        return cls((*range(65, 91), *range(97, 123)))

    @classmethod
    def digits(cls) -> AnyChar:
        """Return a rule that represents digits.

        Returns:
            A rule that represents digits.
        """
        return cls.from_range(48, 58)

    @classmethod
    def letters_and_digits(cls) -> AnyChar:
        """Return a rule that represents letters and digits.

        Returns:
            A rule that represents letters and digits.
        """
        return cls((*range(65, 91), *range(97, 123), *range(48, 58)))

    def accept(self, visitor: GrammarRuleVisitor[T_co]) -> T_co:  # noqa: D102
        return visitor.visit_any_char(self)


@dataclass(frozen=True)
class Choice(GrammarRule):
    """A grammar rule that chooses between multiple rules."""

    rules: tuple[GrammarRule, ...]

    def accept(self, visitor: GrammarRuleVisitor[T_co]) -> T_co:  # noqa: D102
        return visitor.visit_choice(self)


@dataclass(frozen=True)
class Repeat(GrammarRule):
    """A grammar rule that repeats another rule."""

    rule: GrammarRule
    min: int = 0
    max: int | None = None

    @classmethod
    def optional(cls, rule: GrammarRule) -> Repeat:
        """Create an optional rule.

        Args:
            rule: The rule that should be optional.

        Returns:
            The optional rule.
        """
        return cls(rule, min=0, max=1)

    def accept(self, visitor: GrammarRuleVisitor[T_co]) -> T_co:  # noqa: D102
        return visitor.visit_repeat(self)
