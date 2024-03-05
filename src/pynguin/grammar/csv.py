from pynguin.grammar.grammar import Grammar, RuleReference, Sequence, Repeat, Constant, AnyChar

CSV_GRAMMAR = Grammar(
    "csv",
    dict(
        csv=[Repeat(RuleReference("row"))],
        row=[Sequence([
            RuleReference("field"),
            Repeat(Sequence([Constant(","), RuleReference("field")])),
            Constant("\n"),
        ])],
        field=[
            Repeat(AnyChar()),
            Sequence([Constant('"'), Repeat(AnyChar()), Constant('"')]),
        ],
    )
)
