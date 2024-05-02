#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides a function to create a grammar for CSV files."""
from frozendict import frozendict

from pynguin.plugins.grammar_fuzzer.grammar import AnyChar
from pynguin.plugins.grammar_fuzzer.grammar import Choice
from pynguin.plugins.grammar_fuzzer.grammar import Constant
from pynguin.plugins.grammar_fuzzer.grammar import Grammar
from pynguin.plugins.grammar_fuzzer.grammar import GrammarRule
from pynguin.plugins.grammar_fuzzer.grammar import Repeat
from pynguin.plugins.grammar_fuzzer.grammar import RuleReference
from pynguin.plugins.grammar_fuzzer.grammar import Sequence


def create_csv_grammar(  # noqa: PLR0917
    nb_columns: int,
    string_constants: list[str] | None = None,
    min_field_length: int = 0,
    max_field_length: int = 10,
    min_nb_rows: int = 1,
    max_nb_rows: int = 10,
) -> Grammar:
    """Create a grammar for CSV files.

    Args:
        nb_columns: The number of columns in the CSV file.
        string_constants: A list of string constants to use in the CSV file.
        min_field_length: The minimum length of a field.
        max_field_length: The maximum length of a field.
        min_nb_rows: The minimum number of rows in the CSV file.
        max_nb_rows: The maximum number of rows in the CSV file.

    Returns:
        Grammar: A grammar for CSV files.
    """
    assert nb_columns > 0

    rules: list[GrammarRule] = [RuleReference("field")]
    for _ in range(nb_columns - 1):
        rules.extend((Constant(","), RuleReference("field")))
    rules.append(Constant("\n"))

    field_rule: GrammarRule
    if string_constants is None:
        field_rule = Repeat(
            AnyChar.letters_and_digits(), min=min_field_length, max=max_field_length
        )
    else:
        field_rule = Choice(tuple(Constant(constant) for constant in string_constants))

    return Grammar(
        "csv",
        frozendict(
            csv=Repeat(RuleReference("row"), min=min_nb_rows, max=max_nb_rows),
            row=Sequence(tuple(rules)),
            field=Choice(
                rules=(
                    field_rule,
                    Sequence((Constant('"'), field_rule, Constant('"'))),
                )
            ),
        ),
    )
