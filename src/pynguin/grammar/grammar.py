from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, TypeVar
from abc import abstractmethod


@dataclass(frozen=True)
class Grammar:
    initial_rule: str
    expansions: dict[str, tuple[GrammarRule, ...]]

T = TypeVar("T", covariant=True)

class GrammarRuleVisitor(Protocol[T]):
    @abstractmethod
    def visit_constant(self, constant: Constant) -> T:
        """Visit a constant node.

        Args:
            constant (Constant): The constant node to visit.

        Returns:
            T: The result of visiting the constant node.
        """

    @abstractmethod
    def visit_sequence(self, sequence: Sequence) -> T:
        """Visit a sequence node.

        Args:
            sequence (Sequence): The sequence node to visit.

        Returns:
            T: The result of visiting the sequence node.
        """

    @abstractmethod
    def visit_rule_reference(self, rule_reference: RuleReference) -> T:
        """Visit a rule reference node.
        
        Args:
            rule_reference (RuleReference): The rule reference node to visit.
        
        Returns:
            T: The result of visiting the rule reference node.
        """

    @abstractmethod
    def visit_any_char(self, any_char: AnyChar) -> T:
        """Visit an any char node.
        
        Args:
            any_char (AnyChar): The any char node to visit.
        
        Returns:
            T: The result of visiting the any char node.
        """

    @abstractmethod
    def visit_choice(self, choice: Choice) -> T:
        """Visit a choice node.
        
        Args:
            choice (Choice): The choice node to visit.
        
        Returns:
            T: The result of visiting the choice node.
        """

    @abstractmethod
    def visit_repeat(self, repeat: Repeat) -> T:
        """Visit a repeat node.

        Args:
            repeat (Repeat): The repeat node to visit.

        Returns:
            T: The result of visiting the repeat node.
        """

class GrammarRule(Protocol):
    @abstractmethod
    def accept(self, visitor: GrammarRuleVisitor[T]) -> T:
        """Accept a visitor.

        Args:
            visitor (GrammarVisitor[T]): The visitor to accept.

        Returns:
            T: The result of accepting the visitor.
        """

@dataclass(frozen=True)
class Constant(GrammarRule):
    value: str

    def accept(self, visitor: GrammarRuleVisitor[T]) -> T:
        return visitor.visit_constant(self)

@dataclass(frozen=True)
class Sequence(GrammarRule):
    rules: tuple[GrammarRule, ...]

    def accept(self, visitor: GrammarRuleVisitor[T]) -> T:
        return visitor.visit_sequence(self)

@dataclass(frozen=True)
class RuleReference(GrammarRule):
    name: str

    def accept(self, visitor: GrammarRuleVisitor[T]) -> T:
        return visitor.visit_rule_reference(self)

@dataclass(frozen=True)
class AnyChar(GrammarRule):
    min_code: int = 32
    max_code: int = 128

    def accept(self, visitor: GrammarRuleVisitor[T]) -> T:
        return visitor.visit_any_char(self)

@dataclass(frozen=True)
class Choice(GrammarRule):
    rules: tuple[GrammarRule, ...]

    def accept(self, visitor: GrammarRuleVisitor[T]) -> T:
        return visitor.visit_choice(self)

@dataclass(frozen=True)
class Repeat(GrammarRule):
    rule: GrammarRule
    min: int = 0
    max: int | None = None

    def accept(self, visitor: GrammarRuleVisitor[T]) -> T:
        return visitor.visit_repeat(self)
