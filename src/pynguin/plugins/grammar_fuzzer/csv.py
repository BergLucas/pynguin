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
from pynguin.utils import randomness


def create_number_rule(
    min_field_length: int, max_field_length: int | None
) -> GrammarRule:
    """Create a grammar rule for numbers.

    Args:
        min_field_length: The minimum length of a field.
        max_field_length: The maximum length of a field.

    Returns:
        GrammarRule: A grammar rule for numbers.
    """
    min_new_length = min_field_length - 1 if min_field_length > 0 else 0
    max_new_length = max_field_length - 1 if max_field_length is not None else None

    rule = Sequence(
        rules=(
            Repeat.optional(Constant("-")),
            AnyChar.from_range(49, 58),
            Repeat(
                AnyChar.digits(),
                min=min_new_length,
                max=max_new_length,
            ),
        )
    )

    if min_field_length == 0:
        return Repeat.optional(rule)

    return rule


def create_csv_grammar(  # noqa: PLR0917
    columns_number: int,
    min_field_length: int = 0,
    max_field_length: int | None = 10,
    min_rows_number: int = 1,
    max_rows_number: int | None = 10,
    number_column_probability: float = 0.75,
    include_header: bool = True,  # noqa: FBT001, FBT002
    string_constants: list[str] | None = None,
) -> Grammar:
    """Create a grammar for CSV files.

    Args:
        columns_number: The number of columns in the CSV file.
        min_field_length: The minimum length of a field.
        max_field_length: The maximum length of a field.
        min_rows_number: The minimum number of rows in the CSV file.
        max_rows_number: The maximum number of rows in the CSV file.
        number_column_probability: The probability that a column has the number type.
        include_header: Whether headers should be included to the CSV file.
        string_constants: A list of string constants to use in the CSV file.

    Returns:
        Grammar: A grammar for CSV files.
    """
    assert columns_number > 0

    header: list[GrammarRule] = [RuleReference("field")]
    rules: list[GrammarRule] = [
        RuleReference(
            "number_field"
            if randomness.next_float() < number_column_probability
            else "field"
        )
    ]
    for _ in range(columns_number - 1):
        header.extend(
            (
                Constant(","),
                RuleReference("field"),
            )
        )
        rules.extend(
            (
                Constant(","),
                RuleReference(
                    "number_field"
                    if randomness.next_float() < number_column_probability
                    else "field"
                ),
            )
        )
    rules.append(Constant("\n"))

    field_rule: GrammarRule
    if string_constants is None:
        field_rule = Repeat(
            AnyChar.letters_and_digits(), min=min_field_length, max=max_field_length
        )
    else:
        field_rule = Choice(tuple(Constant(constant) for constant in string_constants))

    csv: GrammarRule = Repeat(
        RuleReference("row"), min=min_rows_number, max=max_rows_number
    )

    if include_header:
        csv = Sequence(
            rules=(
                RuleReference("header"),
                Constant("\n"),
                csv,
            )
        )

    return Grammar(
        "csv",
        frozendict(
            csv=csv,
            header=Sequence(tuple(header)),
            row=Sequence(tuple(rules)),
            field=Choice(
                rules=(
                    field_rule,
                    Sequence((Constant('"'), field_rule, Constant('"'))),
                )
            ),
            number_field=create_number_rule(min_field_length, max_field_length),
        ),
    )
