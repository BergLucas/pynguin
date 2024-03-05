from __future__ import annotations
from dataclasses import dataclass
from pynguin.grammar.grammar import Grammar, GrammarRule, GrammarVisitor, AnyChar, Choice, NonTerminal, Repeat, RuleReference, Terminal
from pynguin.utils import randomness
from typing import Callable


class GrammarValueVisitor(GrammarVisitor[str | None]):
    def visit_terminal(self, terminal: Terminal) -> str:
        return terminal.value

    def visit_non_terminal(self, non_terminal: NonTerminal) -> None:
        return None

    def visit_rule_reference(self, rule_reference: RuleReference) -> None:
        return None

    def visit_any_char(self, any_char: AnyChar) -> None:
        return None

    def visit_choice(self, choice: Choice) -> None:
        return None

    def visit_repeat(self, repeat: Repeat) -> None:
        return None

value_visitor = GrammarValueVisitor()

class GrammarSymbolVisitor(GrammarVisitor[str]):
    def visit_terminal(self, terminal: Terminal) -> str:
        return f"{repr(terminal.value)}"

    def visit_non_terminal(self, non_terminal: NonTerminal) -> str:
        return "non_terminal"

    def visit_rule_reference(self, rule_reference: RuleReference) -> str:
        return f"<{rule_reference.name}>"

    def visit_any_char(self, any_char: AnyChar) -> str:
        return "any_char"

    def visit_choice(self, choice: Choice) -> str:
        return "choice"

    def visit_repeat(self, repeat: Repeat) -> str:
        return "repeat"

symbol_visitor = GrammarSymbolVisitor()

class GrammarCostVisitor(GrammarVisitor[float]):
    def __init__(self, grammar: Grammar) -> None:
        self._grammar = grammar
        self._seen: set[str] = set()

    def visit_terminal(self, terminal: Terminal) -> float:
        return 1.0

    def visit_non_terminal(self, non_terminal: NonTerminal) -> float:
        return sum(
            child.accept(self)
            for child
            in non_terminal.children
        )

    def visit_rule_reference(self, rule_reference: RuleReference) -> int:
        if rule_reference.name in self._seen:
            return float("inf")

        self._seen.add(rule_reference.name)

        return self._grammar.rules[rule_reference.name].accept(self)

    def visit_any_char(self, any_char: AnyChar) -> int:
        return 1.0

    def visit_choice(self, choice: Choice) -> int:
        return min(
            child.accept(self)
            for child
            in choice.rules
        )

    def visit_repeat(self, repeat: Repeat) -> int:
        if repeat.max is None:
            return float("inf")

        nb_repeats = repeat.max - repeat.min

        return nb_repeats + repeat.rule.accept(self)

class GrammarExpansionsVisitor(GrammarVisitor[list[list[GrammarRule]]]):

    def __init__(self, grammar: Grammar) -> None:
        self._grammar = grammar

    def visit_terminal(self, terminal: Terminal) -> list[list[GrammarRule]]:
        return [[]]

    def visit_non_terminal(self, non_terminal: NonTerminal) -> list[list[GrammarRule]]:
        return [non_terminal.children.copy()]

    def visit_rule_reference(self, rule_reference: RuleReference) -> list[list[GrammarRule]]:
        return [[self._grammar.rules[rule_reference.name]]]

    def visit_any_char(self, any_char: AnyChar) -> list[list[GrammarRule]]:
        return [[Terminal(chr(i))] for i in range(any_char.min_code, any_char.max_code)]

    def visit_choice(self, choice: Choice) -> list[list[GrammarRule]]:
        return [[rule] for rule in choice.rules]

    def visit_repeat(self, repeat: Repeat) -> list[list[GrammarRule]]: 
        expansions = []
        if repeat.min == 0:
            expansions.append([])
        expansions.append([repeat.rule])
        if repeat.max is None or repeat.max > 1:
            max_repeat = None if repeat.max is None else repeat.max - 1
            expansions.append([repeat.rule, Repeat(repeat.rule, repeat.min, max_repeat)])
        return expansions

@dataclass
class GrammarDerivationTree:
    symbol: str
    rule: GrammarRule
    children: list[GrammarDerivationTree]

    def possible_expansions(self) -> int:
        if self.rule.accept(value_visitor) is not None:
            return 0

        elif not self.children:
            return 1

        return sum(
            child.possible_expansions()
            for child
            in self.children
        )

    def any_possible_expansions(self) -> bool:
        if self.rule.accept(value_visitor) is not None:
            return False

        elif not self.children:
            return True

        return any(
            child.any_possible_expansions()
            for child
            in self.children
        )

    def __str__(self) -> str:
        value = self.rule.accept(value_visitor)

        if value is not None:
            return value

        return "".join(
            str(child)
            for child
            in self.children
        )

    def __repr__(self) -> str:
        return f"GrammarDerivationTree({self.symbol}, {self.children})"

class GrammarFuzzer:
    def __init__(self, grammar: Grammar) -> None:
        self._grammar = grammar
        self._expansions_visitor = GrammarExpansionsVisitor(grammar)
        self._min_non_terminal = 0
        self._max_non_terminal = 100

    @property
    def grammar(self) -> Grammar:
        return self._grammar

    def create_tree(self) -> GrammarDerivationTree:
        rule = RuleReference(self._grammar.initial_rule)

        derivation_tree = GrammarDerivationTree(rule.accept(symbol_visitor), rule, [])

        self._expand_tree_stategy(derivation_tree)

        return derivation_tree

    def mutate_tree(self, derivation_tree: GrammarDerivationTree) -> None:
        if randomness.next_float() < 0.1:
            derivation_tree.children.clear()
            self._expand_tree_stategy(derivation_tree)
            return

        if not derivation_tree.children:
            return

        child = randomness.choice(derivation_tree.children)

        self.mutate_tree(child)

    def _expand_node_randomly(self, node: GrammarDerivationTree) -> None:
        assert not node.children

        expansions = node.rule.accept(self._expansions_visitor)

        expansion = randomness.choice(expansions)

        node.children.extend(
            GrammarDerivationTree(rule.accept(symbol_visitor), rule, [])
            for rule in expansion
        )

    def _expansion_cost(self, expansion: list[GrammarRule]) -> float:
        return sum(
            rule.accept(GrammarCostVisitor(self._grammar))
            for rule in expansion
        )

    def _expand_node_by_cost(self, node: GrammarDerivationTree, cost_function: Callable) -> None:
        assert not node.children

        expansions = node.rule.accept(self._expansions_visitor)

        expansions_costs: dict[float, list[list[GrammarRule]]] = {}

        for expansion in expansions:
            cost = self._expansion_cost(expansion)
            if cost not in expansions_costs:
                expansions_costs[cost] = []
            expansions_costs[cost].append(expansion)

        expansion = randomness.choice(expansions_costs[cost_function(expansions_costs)])

        node.children.extend(
            GrammarDerivationTree(rule.accept(symbol_visitor), rule, [])
            for rule in expansion
        )
    
    def _expand_node_min_cost(self, node: GrammarDerivationTree) -> None:
        self._expand_node_by_cost(node, min)

    def _expand_node_max_cost(self, node: GrammarDerivationTree) -> None:
        self._expand_node_by_cost(node, max)

    def _expand_tree_once(self, tree: GrammarDerivationTree, expand_node_function: Callable) -> None:
        if not tree.children:
            expand_node_function(tree)
            return

        children_to_expand = [
            child
            for child in tree.children
            if child.any_possible_expansions()
        ]

        child = randomness.choice(children_to_expand)

        self._expand_tree_once(child, expand_node_function)

    def _expand_tree(self, tree: GrammarDerivationTree, expand_node_function: Callable, limit: float | None) -> None:
        while (
            (limit is None or tree.possible_expansions() < limit)
            and tree.any_possible_expansions()
        ):
            self._expand_tree_once(tree, expand_node_function)

    def _expand_tree_stategy(self, tree: GrammarDerivationTree) -> None:
        self._expand_tree(tree, self._expand_node_max_cost, self._min_non_terminal)
        self._expand_tree(tree, self._expand_node_randomly, self._max_non_terminal)
        self._expand_tree(tree, self._expand_node_min_cost, None)

        assert not tree.any_possible_expansions()
