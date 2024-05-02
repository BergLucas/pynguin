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
    column_number: int,
    min_field_length: int = 0,
    max_field_length: int | None = 10,
    min_row_number: int = 1,
    max_row_number: int | None = 10,
    string_constants: list[str] | None = None,
) -> Grammar:
    """Create a grammar for CSV files.

    Args:
        column_number: The number of columns in the CSV file.
        min_field_length: The minimum length of a field.
        max_field_length: The maximum length of a field.
        min_row_number: The minimum number of rows in the CSV file.
        max_row_number: The maximum number of rows in the CSV file.
        string_constants: A list of string constants to use in the CSV file.

    Returns:
        Grammar: A grammar for CSV files.
    """
    assert column_number > 0

    rules: list[GrammarRule] = [RuleReference("field")]
    for _ in range(column_number - 1):
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
            csv=Repeat(RuleReference("row"), min=min_row_number, max=max_row_number),
            row=Sequence(tuple(rules)),
            field=Choice(
                rules=(
                    field_rule,
                    Sequence((Constant('"'), field_rule, Constant('"'))),
                )
            ),
        ),
    )
