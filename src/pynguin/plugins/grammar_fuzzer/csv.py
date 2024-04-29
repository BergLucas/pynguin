from pynguin.plugins.grammar_fuzzer.grammar import AnyChar
from pynguin.plugins.grammar_fuzzer.grammar import Choice
from pynguin.plugins.grammar_fuzzer.grammar import Constant
from pynguin.plugins.grammar_fuzzer.grammar import Grammar
from pynguin.plugins.grammar_fuzzer.grammar import GrammarRule
from pynguin.plugins.grammar_fuzzer.grammar import Repeat
from pynguin.plugins.grammar_fuzzer.grammar import RuleReference
from pynguin.plugins.grammar_fuzzer.grammar import Sequence


def create_csv_grammar(
    nb_columns: int,
    string_constants: list[str] | None = None,
    min_field_length: int = 0,
) -> Grammar:
    assert nb_columns > 0

    rules: list[GrammarRule] = [RuleReference("field")]
    for _ in range(nb_columns - 1):
        rules.append(Constant(","))
        rules.append(RuleReference("field"))
    rules.append(Constant("\n"))

    field_rule: GrammarRule
    if string_constants is None:
        field_rule = Repeat(AnyChar.letters_and_digits(), min=min_field_length)
    else:
        field_rule = Choice(tuple(Constant(constant) for constant in string_constants))

    return Grammar(
        "csv",
        dict(
            csv=(Repeat(RuleReference("row")),),
            row=(Sequence(tuple(rules)),),
            field=(
                field_rule,
                Sequence((Constant('"'), field_rule, Constant('"'))),
            ),
        ),
    )