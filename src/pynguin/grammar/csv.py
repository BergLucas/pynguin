from pynguin.grammar.grammar import Grammar, RuleReference, NonTerminal, Repeat, Terminal, Choice, AnyChar

CSV_GRAMMAR = Grammar(
    "csv",
    dict(
        csv=Repeat(RuleReference("row")),
        row=NonTerminal([
            RuleReference("field"),
            Repeat(NonTerminal([Terminal(","), RuleReference("field")])),
            Terminal("\n"),
        ]),
        field=Choice([
            Repeat(AnyChar()),
            NonTerminal([Terminal('"'), Repeat(AnyChar()), Terminal('"')]),
        ])
    )
)
