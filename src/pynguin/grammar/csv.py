from pynguin.grammar.grammar import GrammarExpansion, Grammar, RuleReference, Sequence, Repeat, Constant, AnyChar

CSV_GRAMMAR = Grammar(
    "csv",
    dict(
        csv=GrammarExpansion([Repeat(RuleReference("row"))]),
        row=GrammarExpansion([Sequence([
            RuleReference("field"),
            Repeat(Sequence([Constant(","), RuleReference("field")])),
            Constant("\n"),
        ])]),
        field=GrammarExpansion([
            Repeat(AnyChar()),
            Sequence([Constant('"'), Repeat(AnyChar()), Constant('"')]),
        ])
    )
)
