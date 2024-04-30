#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides a grammar-based fuzzer for generating test data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pynguin.plugins.grammar_fuzzer.grammar import AnyChar
from pynguin.plugins.grammar_fuzzer.grammar import Choice
from pynguin.plugins.grammar_fuzzer.grammar import Constant
from pynguin.plugins.grammar_fuzzer.grammar import Grammar
from pynguin.plugins.grammar_fuzzer.grammar import GrammarRule
from pynguin.plugins.grammar_fuzzer.grammar import GrammarRuleVisitor
from pynguin.plugins.grammar_fuzzer.grammar import Repeat
from pynguin.plugins.grammar_fuzzer.grammar import RuleReference
from pynguin.plugins.grammar_fuzzer.grammar import Sequence
from pynguin.utils import randomness


if TYPE_CHECKING:
    from collections.abc import Callable


class GrammarNonTerminalVisitor(GrammarRuleVisitor[bool]):
    """A visitor for grammar rules that returns whether a rule is a non-terminal."""

    def visit_constant(self, constant: Constant) -> bool:  # noqa: D102
        return False

    def visit_sequence(self, sequence: Sequence) -> bool:  # noqa: D102
        return any(rule.accept(self) for rule in sequence.rules)

    def visit_rule_reference(self, rule_reference: RuleReference) -> bool:  # noqa: D102
        return True

    def visit_any_char(self, any_char: AnyChar) -> bool:  # noqa: D102
        return False

    def visit_choice(self, choice: Choice) -> bool:  # noqa: D102
        return any(rule.accept(self) for rule in choice.rules)

    def visit_repeat(self, repeat: Repeat) -> bool:  # noqa: D102
        if repeat.max is None:
            return True
        return repeat.rule.accept(self)


non_terminal_visitor = GrammarNonTerminalVisitor()


class GrammarValueVisitor(GrammarRuleVisitor[str | None]):
    """A visitor for grammar rules that returns rules values."""

    def visit_constant(self, constant: Constant) -> str:  # noqa: D102
        return constant.value

    def visit_sequence(self, sequence: Sequence) -> None:  # noqa: D102
        return None

    def visit_rule_reference(self, rule_reference: RuleReference) -> None:  # noqa: D102
        return None

    def visit_any_char(self, any_char: AnyChar) -> None:  # noqa: D102
        return None

    def visit_choice(self, choice: Choice) -> None:  # noqa: D102
        return None

    def visit_repeat(self, repeat: Repeat) -> None:  # noqa: D102
        return None


value_visitor = GrammarValueVisitor()


class GrammarSymbolVisitor(GrammarRuleVisitor[str]):
    """A visitor for grammar rules that returns a string representation of the rule."""

    def visit_constant(self, constant: Constant) -> str:  # noqa: D102
        return repr(constant.value)

    def visit_sequence(self, sequence: Sequence) -> str:  # noqa: D102
        return "sequence"

    def visit_rule_reference(self, rule_reference: RuleReference) -> str:  # noqa: D102
        return f"<{rule_reference.name}>"

    def visit_any_char(self, any_char: AnyChar) -> str:  # noqa: D102
        return "any_char"

    def visit_choice(self, choice: Choice) -> str:  # noqa: D102
        return "choice"

    def visit_repeat(self, repeat: Repeat) -> str:  # noqa: D102
        return "repeat"


symbol_visitor = GrammarSymbolVisitor()


class GrammarExpansionsVisitor(GrammarRuleVisitor[list[tuple[GrammarRule, ...]]]):
    """A visitor for grammar rules that returns possible expansions."""

    def __init__(self, grammar: Grammar) -> None:
        """Create a new grammar expansions visitor.

        Args:
            grammar: The grammar to use.
        """
        self._grammar = grammar

    def visit_constant(  # noqa: D102
        self, constant: Constant
    ) -> list[tuple[GrammarRule, ...]]:
        return [()]

    def visit_sequence(  # noqa: D102
        self, sequence: Sequence
    ) -> list[tuple[GrammarRule, ...]]:
        return [sequence.rules]

    def visit_rule_reference(  # noqa: D102
        self, rule_reference: RuleReference
    ) -> list[tuple[GrammarRule, ...]]:
        return [(rule,) for rule in self._grammar.expansions[rule_reference.name]]

    def visit_any_char(  # noqa: D102
        self, any_char: AnyChar
    ) -> list[tuple[GrammarRule, ...]]:
        return [(Constant(chr(code)),) for code in any_char.codes]

    def visit_choice(  # noqa: D102
        self, choice: Choice
    ) -> list[tuple[GrammarRule, ...]]:
        return [(rule,) for rule in choice.rules]

    def visit_repeat(  # noqa: D102
        self, repeat: Repeat
    ) -> list[tuple[GrammarRule, ...]]:
        if repeat.min > 0:
            new_max_repeat = repeat.max - repeat.min if repeat.max is not None else None
            return [
                (
                    Sequence(
                        (
                            *(repeat.rule for _ in range(repeat.min)),
                            Repeat(repeat.rule, min=0, max=new_max_repeat),
                        )
                    ),
                )
            ]

        expansions: list[tuple[GrammarRule, ...]] = [()]
        if repeat.max is None:
            expansions.append((repeat.rule, repeat))
        elif repeat.max > 1:
            expansions.append(
                (repeat.rule, Repeat(repeat.rule, repeat.min, repeat.max - 1))
            )
        return expansions


@dataclass
class GrammarDerivationTree:
    """A derivation tree for a grammar."""

    symbol: str
    rule: GrammarRule
    children: list[GrammarDerivationTree] | None

    def possible_expansions(self) -> int:
        """Get the number of possible expansions."""
        if self.children is None:
            return 1

        return sum(child.possible_expansions() for child in self.children)

    def __deepcopy__(self, memo: dict[int, object]) -> GrammarDerivationTree:
        if self.children is None:
            return GrammarDerivationTree(self.symbol, self.rule, None)

        return GrammarDerivationTree(
            self.symbol,
            self.rule,
            [child.__deepcopy__(memo) for child in self.children],
        )

    def __str__(self) -> str:
        value = self.rule.accept(value_visitor)

        if self.children is None:
            return ""

        if value is not None:
            return value

        return "".join(str(child) for child in self.children)

    def __repr__(self) -> str:
        return f"GrammarDerivationTree({self.symbol}, {self.children})"


class GrammarFuzzer:
    """A grammar-based fuzzer for generating test data."""

    def __init__(
        self, grammar: Grammar, min_non_terminal: int = 0, max_non_terminal: int = 10
    ) -> None:
        """Create a new grammar fuzzer.

        Args:
            grammar: The grammar to use.
            min_non_terminal: The minimum number of non-terminal expansions.
            max_non_terminal: The maximum number of non-terminal expansions.
        """
        assert min_non_terminal <= max_non_terminal

        self._grammar = grammar
        self._expansions_visitor = GrammarExpansionsVisitor(grammar)
        self._min_non_terminal = min_non_terminal
        self._max_non_terminal = max_non_terminal

    @property
    def grammar(self) -> Grammar:
        """Get the grammar.

        Returns:
            The grammar.
        """
        return self._grammar

    def create_tree(self) -> GrammarDerivationTree:
        """Create a derivation tree.

        Returns:
            A derivation tree.
        """
        rule = RuleReference(self._grammar.initial_rule)

        derivation_tree = GrammarDerivationTree(rule.accept(symbol_visitor), rule, None)

        self._expand_tree_stategy(derivation_tree)

        return derivation_tree

    def mutate_tree(self, derivation_tree: GrammarDerivationTree) -> None:
        """Mutate a derivation tree.

        Args:
            derivation_tree (GrammarDerivationTree): The derivation tree to mutate.
        """
        if randomness.next_float() < 0.1:
            derivation_tree.children = None
            self._expand_tree_stategy(derivation_tree)
            return

        if derivation_tree.children is None or not derivation_tree.children:
            return

        child = randomness.choice(derivation_tree.children)

        self.mutate_tree(child)

    def _expand_node_randomly(self, node: GrammarDerivationTree) -> None:
        assert node.children is None

        expansions = node.rule.accept(self._expansions_visitor)

        expansion = randomness.choice(expansions)

        node.children = [
            GrammarDerivationTree(rule.accept(symbol_visitor), rule, None)
            for rule in expansion
        ]

    @staticmethod
    def _non_terminal(expansion: tuple[GrammarRule, ...]) -> tuple[GrammarRule, ...]:
        return tuple(rule for rule in expansion if rule.accept(non_terminal_visitor))

    def _rule_cost(self, rule: GrammarRule, seen: set[GrammarRule]) -> float:
        return min(
            self._expansion_cost(expansion, seen)
            for expansion in rule.accept(self._expansions_visitor)
        )

    def _expansion_cost(
        self, expansions: tuple[GrammarRule, ...], seen: set[GrammarRule] | None = None
    ) -> float:
        if seen is None:
            seen = set()

        rules = self._non_terminal(expansions)

        if not rules:
            return 1.0

        if any(rule in seen for rule in rules):
            return float("inf")

        seen.update(rule for rule in rules)

        return 1.0 + sum(self._rule_cost(rule, seen) for rule in rules)

    def _expand_node_by_cost(
        self, node: GrammarDerivationTree, cost_function: Callable
    ) -> None:
        assert node.children is None

        expansions = node.rule.accept(self._expansions_visitor)

        expansions_costs: dict[float, list[tuple[GrammarRule, ...]]] = {}

        for expansion in expansions:
            cost = self._expansion_cost(expansion)
            if cost not in expansions_costs:
                expansions_costs[cost] = []
            expansions_costs[cost].append(expansion)

        expansion = randomness.choice(expansions_costs[cost_function(expansions_costs)])

        node.children = [
            GrammarDerivationTree(rule.accept(symbol_visitor), rule, None)
            for rule in expansion
        ]

    def _expand_node_min_cost(self, node: GrammarDerivationTree) -> None:
        self._expand_node_by_cost(node, min)

    def _expand_node_max_cost(self, node: GrammarDerivationTree) -> None:
        self._expand_node_by_cost(node, max)

    def _expand_tree_once(
        self, tree: GrammarDerivationTree, expand_node_function: Callable
    ) -> None:
        if tree.children is None:
            expand_node_function(tree)
            return

        children_to_expand = [
            child for child in tree.children if child.possible_expansions() > 0
        ]

        child = randomness.choice(children_to_expand)

        self._expand_tree_once(child, expand_node_function)

    def _expand_tree(
        self,
        tree: GrammarDerivationTree,
        expand_node_function: Callable,
        limit: float | None,
    ) -> None:
        while (possible_expansions := tree.possible_expansions()) > 0 and (
            limit is None or possible_expansions < limit
        ):
            self._expand_tree_once(tree, expand_node_function)

    def _expand_tree_stategy(self, tree: GrammarDerivationTree) -> None:
        self._expand_tree(tree, self._expand_node_max_cost, self._min_non_terminal)
        self._expand_tree(tree, self._expand_node_randomly, self._max_non_terminal)
        self._expand_tree(tree, self._expand_node_min_cost, None)

        assert tree.possible_expansions() == 0
