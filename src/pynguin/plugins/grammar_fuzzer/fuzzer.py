#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides a grammar-based fuzzer for generating test data."""
from __future__ import annotations

import sys

from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Protocol
from typing import TypeVar

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
    from collections.abc import Iterable


T_co = TypeVar("T_co", covariant=True)


class GrammarDerivationTreeVisitor(Protocol[T_co]):
    """A visitor for grammar derivation trees."""

    @abstractmethod
    def visit_leaf(self, leaf: GrammarDerivationLeaf) -> T_co:
        """Visit a leaf node.

        Args:
            leaf: The leaf node to visit.

        Returns:
            The result of visiting the leaf node.
        """

    @abstractmethod
    def visit_node(self, node: GrammarDerivationNode) -> T_co:
        """Visit a node.

        Args:
            node: The node to visit.

        Returns:
            The result of visiting the node.
        """


class GrammarDerivationTree(Protocol):
    """A derivation tree of a grammar."""

    @abstractmethod
    def accept(self, visitor: GrammarDerivationTreeVisitor[T_co]) -> T_co:
        """Accept a visitor.

        Args:
            visitor: The visitor to accept.

        Returns:
            The result of the visit.
        """

    @abstractmethod
    def __deepcopy__(self, memo: dict) -> GrammarDerivationTree:
        """Deep copy the derivation tree.

        Args:
            memo: The memo dictionary.

        Returns:
            A deep copy of the derivation tree.
        """

    @abstractmethod
    def __str__(self) -> str:
        """Get a string representation of the derivation tree.

        Returns:
            A string representation of the derivation tree.
        """


@dataclass
class GrammarDerivationLeaf(GrammarDerivationTree):
    """A leaf in a grammar derivation tree."""

    value: str

    def accept(self, visitor: GrammarDerivationTreeVisitor[T_co]) -> T_co:  # noqa: D102
        return visitor.visit_leaf(self)

    def __deepcopy__(self, memo: dict[int, object]) -> GrammarDerivationLeaf:
        return GrammarDerivationLeaf(self.value)

    def __str__(self) -> str:
        return self.value


@dataclass
class GrammarDerivationNode(GrammarDerivationTree):
    """A node in a grammar derivation tree."""

    rule: GrammarRule
    children: list[GrammarDerivationTree] | None = None

    def accept(self, visitor: GrammarDerivationTreeVisitor[T_co]) -> T_co:  # noqa: D102
        return visitor.visit_node(self)

    def __str__(self) -> str:
        if self.children is None:
            return ""

        return "".join(str(child) for child in self.children)

    def __deepcopy__(self, memo: dict[int, object]) -> GrammarDerivationNode:
        if self.children is None:
            return GrammarDerivationNode(self.rule)

        return GrammarDerivationNode(
            self.rule,
            deepcopy(self.children, memo),
        )


class GrammarDerivationTreePossibleExpansionsCalculator(
    GrammarDerivationTreeVisitor[int]
):
    """A visitor that calculates the number of possible expansions."""

    def visit_leaf(self, leaf: GrammarDerivationLeaf) -> int:  # noqa: D102
        return 0

    def visit_node(self, node: GrammarDerivationNode) -> int:  # noqa: D102
        if node.children is None:
            return 1

        return sum(child.accept(self) for child in node.children)


class GrammarDerivationTreeExpander(GrammarDerivationTreeVisitor[bool]):
    """A visitor that expands a node."""

    def __init__(
        self, grammar_expander: GrammarRuleVisitor[list[GrammarDerivationTree]]
    ):
        """Create a new derivation tree expander.

        Args:
            grammar_expander: The grammar expander to use.
        """
        self._grammar_expander = grammar_expander

    def visit_leaf(self, leaf: GrammarDerivationLeaf) -> bool:  # noqa: D102
        return False

    def visit_node(self, node: GrammarDerivationNode) -> bool:  # noqa: D102
        if node.children is None:
            node.children = node.rule.accept(self._grammar_expander)
            return True

        shuffled_children = randomness.RNG.sample(node.children, len(node.children))

        return any(child.accept(self) for child in shuffled_children)


class GrammarDerivationTreeMutator(GrammarDerivationTreeVisitor[bool]):
    """A visitor that mutates a node."""

    def __init__(self, mutation_rate: float) -> None:
        """Create a new derivation tree mutator.

        Args:
            mutation_rate: The mutation rate.
        """
        self._mutation_rate = mutation_rate

    def visit_leaf(self, leaf: GrammarDerivationLeaf) -> bool:  # noqa: D102
        return False

    def visit_node(self, node: GrammarDerivationNode) -> bool:  # noqa: D102
        if randomness.next_float() < self._mutation_rate:
            node.children = None
            return True

        if node.children is None:
            return False

        shuffled_children = randomness.RNG.sample(node.children, len(node.children))

        return any(child.accept(self) for child in shuffled_children)


class GrammarRuleCost(GrammarRuleVisitor[int | None]):
    """A visitor that calculates the cost of a rule."""

    def __init__(self, grammar: Grammar) -> None:
        """Create a new rule cost visitor.

        Args:
            grammar: The grammar to use.
        """
        self._grammar = grammar
        self._seen_references: set[RuleReference] = set()

    def visit_constant(self, constant: Constant) -> int | None:  # noqa: D102
        return 1

    def visit_sequence(self, sequence: Sequence) -> int | None:  # noqa: D102
        total = 1

        for rule in sequence.rules:
            rule_cost = rule.accept(self)

            if rule_cost is None:
                return None

            total += rule_cost

        return total

    def visit_rule_reference(  # noqa: D102
        self, rule_reference: RuleReference
    ) -> int | None:
        if rule_reference in self._seen_references:
            return None

        self._seen_references.add(rule_reference)

        rule = self._grammar.rules[rule_reference.name]

        rule_cost = rule.accept(self)

        if rule_cost is None:
            return None

        return 1 + rule_cost

    def visit_any_char(self, any_char: AnyChar) -> int | None:  # noqa: D102
        return 1

    def visit_choice(self, choice: Choice) -> int | None:  # noqa: D102
        min_rule_cost = sys.maxsize

        for rule in choice.rules:
            rule_cost = rule.accept(self)

            if rule_cost is None:
                return None

            min_rule_cost = min(min_rule_cost, rule_cost)

        return 1 + min_rule_cost

    def visit_repeat(self, repeat: Repeat) -> int | None:  # noqa: D102
        if repeat.max is None:
            return None

        rule_cost = repeat.rule.accept(self)

        if rule_cost is None:
            return None

        return 1 + rule_cost * repeat.min + 1 + 1


class GrammarRuleRandomExpander(GrammarRuleVisitor[list[GrammarDerivationTree]]):
    """A visitor that generates rules expansions randomly."""

    def __init__(self, grammar: Grammar) -> None:
        """Create a new grammar expander.

        Args:
            grammar: The grammar to use.
        """
        self._grammar = grammar

    def visit_constant(  # noqa: D102
        self, constant: Constant
    ) -> list[GrammarDerivationTree]:
        return [GrammarDerivationLeaf(constant.value)]

    def visit_sequence(  # noqa: D102
        self, sequence: Sequence
    ) -> list[GrammarDerivationTree]:
        return [GrammarDerivationNode(rule) for rule in sequence.rules]

    def visit_rule_reference(  # noqa: D102
        self, rule_reference: RuleReference
    ) -> list[GrammarDerivationTree]:
        rule = self._grammar.rules[rule_reference.name]
        return [GrammarDerivationNode(rule)]

    def visit_any_char(  # noqa: D102
        self, any_char: AnyChar
    ) -> list[GrammarDerivationTree]:
        code = randomness.choice(any_char.codes)
        return [GrammarDerivationLeaf(chr(code))]

    def visit_choice(self, choice: Choice) -> list[GrammarDerivationTree]:  # noqa: D102
        rule = randomness.choice(choice.rules)
        return [GrammarDerivationNode(rule)]

    def visit_repeat(self, repeat: Repeat) -> list[GrammarDerivationTree]:  # noqa: D102
        nodes: list[GrammarDerivationTree] = [
            GrammarDerivationNode(repeat.rule) for _ in range(repeat.min)
        ]

        if repeat.min == repeat.max:
            return nodes

        new_repeat_max = repeat.max - repeat.min - 1 if repeat.max is not None else None

        new_repeat = Repeat(repeat.rule, min=0, max=new_repeat_max)

        rule = Sequence(rules=(repeat.rule, new_repeat))

        empty = Sequence(rules=())

        choice_rule = Choice(rules=(empty, rule))

        nodes.append(GrammarDerivationNode(choice_rule))

        return nodes


class GrammarRuleCostExpander(GrammarRuleRandomExpander):
    """A visitor that generates rules expansions based on a cost function."""

    def __init__(
        self, grammar: Grammar, cost_function: Callable[[Iterable[float]], float]
    ) -> None:
        """Create a new grammar expander.

        Args:
            grammar: The grammar to use.
            cost_function: The cost function to use.
        """
        super().__init__(grammar)
        self._cost_function = cost_function

    def _rule_cost(self, rule: GrammarRule) -> float:
        rule_cost = rule.accept(GrammarRuleCost(self._grammar))

        if rule_cost is None:
            return sys.maxsize

        return rule_cost

    def visit_choice(self, choice: Choice) -> list[GrammarDerivationTree]:  # noqa: D102
        rule_costs = {self._rule_cost(rule): rule for rule in choice.rules}
        best_rule_cost = self._cost_function(rule_costs)
        rule = rule_costs[best_rule_cost]
        return [GrammarDerivationNode(rule)]


class GrammarFuzzer:
    """A grammar-based fuzzer for generating test data."""

    def __init__(
        self,
        grammar: Grammar,
        min_non_terminal: int = 0,
        max_non_terminal: int = 25,
        mutation_rate: float = 0.1,
    ) -> None:
        """Create a new grammar fuzzer.

        Args:
            grammar: The grammar to use.
            min_non_terminal: The minimum number of non-terminal expansions.
            max_non_terminal: The maximum number of non-terminal expansions.
            mutation_rate: The mutation rate.
        """
        assert min_non_terminal <= max_non_terminal

        self._grammar = grammar
        self._random_expander = GrammarDerivationTreeExpander(
            GrammarRuleRandomExpander(grammar)
        )
        self._min_cost_expander = GrammarDerivationTreeExpander(
            GrammarRuleCostExpander(grammar, min)
        )
        self._max_cost_expander = GrammarDerivationTreeExpander(
            GrammarRuleCostExpander(grammar, max)
        )
        self._possible_expansions_calculator = (
            GrammarDerivationTreePossibleExpansionsCalculator()
        )
        self._derivation_tree_mutator = GrammarDerivationTreeMutator(mutation_rate)
        self._min_non_terminal = min_non_terminal
        self._max_non_terminal = max_non_terminal
        self._mutation_rate = mutation_rate

    @property
    def grammar(self) -> Grammar:
        """Get the grammar.

        Returns:
            The grammar.
        """
        return self._grammar

    @property
    def min_non_terminal(self) -> int:
        """Get the minimum number of non-terminal expansions.

        Returns:
            The minimum number of non-terminal expansions.
        """
        return self._min_non_terminal

    @property
    def max_non_terminal(self) -> int:
        """Get the maximum number of non-terminal expansions.

        Returns:
            The maximum number of non-terminal expansions.
        """
        return self._max_non_terminal

    @property
    def mutation_rate(self) -> float:
        """Get the mutation rate.

        Returns:
            The mutation rate.
        """
        return self._mutation_rate

    def create_tree(self) -> GrammarDerivationTree:
        """Create a derivation tree.

        Returns:
            A derivation tree.
        """
        rule = RuleReference(self._grammar.initial_rule)

        derivation_node = GrammarDerivationNode(rule)

        self._expand_tree_strategy(derivation_node)

        return derivation_node

    def mutate_tree(self, derivation_tree: GrammarDerivationTree) -> bool:
        """Mutate a derivation tree.

        Args:
            derivation_tree: The derivation tree to mutate.

        Returns:
            True if the tree was mutated, False otherwise.
        """
        return derivation_tree.accept(self._derivation_tree_mutator)

    def _expand_tree(
        self,
        tree: GrammarDerivationTree,
        tree_expander: GrammarDerivationTreeExpander,
        limit: float | None,
    ) -> None:
        while (
            possible_expansions := tree.accept(self._possible_expansions_calculator)
        ) > 0 and (limit is None or possible_expansions < limit):
            tree.accept(tree_expander)

    def _expand_tree_strategy(self, tree: GrammarDerivationTree) -> None:
        self._expand_tree(tree, self._max_cost_expander, self._min_non_terminal)
        self._expand_tree(tree, self._random_expander, self._max_non_terminal)
        self._expand_tree(tree, self._min_cost_expander, None)

        assert tree.accept(self._possible_expansions_calculator) == 0

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GrammarFuzzer)
            and self._grammar == other._grammar
            and self._min_non_terminal == other._min_non_terminal
            and self._max_non_terminal == other._max_non_terminal
            and self._mutation_rate == other._mutation_rate
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._grammar,
                self._min_non_terminal,
                self._max_non_terminal,
                self._mutation_rate,
            )
        )
