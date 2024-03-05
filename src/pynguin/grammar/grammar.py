from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, TypeVar


@dataclass
class Grammar:
    initial_rule: str
    rules: dict[str, GrammarRule]

T = TypeVar("T")

class GrammarVisitor(Protocol[T]):
    def visit_terminal(self, terminal: Terminal) -> T:
        """Visit a terminal node.
        
        Args:
            terminal (Terminal): The terminal node to visit.
        
        Returns:
            T: The result of visiting the terminal node.
        """

    def visit_non_terminal(self, non_terminal: NonTerminal) -> T:
        """Visit a non-terminal node.

        Args:
            non_terminal (NonTerminal): The non-terminal node to visit.

        Returns:
            T: The result of visiting the non-terminal node.
        """

    def visit_rule_reference(self, rule_reference: RuleReference) -> T:
        """Visit a rule reference node.
        
        Args:
            rule_reference (RuleReference): The rule reference node to visit.
        
        Returns:
            T: The result of visiting the rule reference node.
        """

    def visit_any_char(self, any_char: AnyChar) -> T:
        """Visit an any char node.
        
        Args:
            any_char (AnyChar): The any char node to visit.
        
        Returns:
            T: The result of visiting the any char node.
        """

    def visit_choice(self, choice: Choice) -> T:
        """Visit a choice node.
        
        Args:
            choice (Choice): The choice node to visit.
        
        Returns:
            T: The result of visiting the choice node.
        """

    def visit_repeat(self, repeat: Repeat) -> T:
        """Visit a repeat node.

        Args:
            repeat (Repeat): The repeat node to visit.

        Returns:
            T: The result of visiting the repeat node.
        """

class GrammarRule(Protocol):
    def accept(self, visitor: GrammarVisitor[T]) -> T:
        """Accept a visitor.

        Args:
            visitor (GrammarVisitor[T]): The visitor to accept.

        Returns:
            T: The result of accepting the visitor.
        """

@dataclass
class Terminal(GrammarRule):
    value: str

    def accept(self, visitor: GrammarVisitor[T]) -> T:
        return visitor.visit_terminal(self)

@dataclass
class NonTerminal(GrammarRule):
    children: list[GrammarRule]

    def accept(self, visitor: GrammarVisitor[T]) -> T:
        return visitor.visit_non_terminal(self)

@dataclass 
class RuleReference(GrammarRule):
    name: str

    def accept(self, visitor: GrammarVisitor[T]) -> T:
        return visitor.visit_rule_reference(self)

@dataclass
class AnyChar(GrammarRule):
    min_code: int = 32
    max_code: int = 128

    def accept(self, visitor: GrammarVisitor[T]) -> T:
        return visitor.visit_any_char(self)

@dataclass
class Choice(GrammarRule):
    rules: list[GrammarRule]

    def accept(self, visitor: GrammarVisitor[T]) -> T:
        return visitor.visit_choice(self)

@dataclass
class Repeat(GrammarRule):
    rule: GrammarRule
    min: int = 0
    max: int | None = None

    def accept(self, visitor: GrammarVisitor[T]) -> T:
        return visitor.visit_repeat(self)
