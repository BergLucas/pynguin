#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides class prompt for generating tests for a module."""

import ast
import textwrap
from collections.abc import Sequence

from pynguin.large_language_model.prompts.prompt import Prompt
from pynguin.utils.generic.genericaccessibleobject import (
    GenericCallableAccessibleObject,
    GenericConstructor,
    GenericFunction,
    GenericMethod,
)


class UncoveredTargetsPrompt(Prompt):
    """Implementation prompt for generating tests for a module."""

    def __init__(
        self,
        callables: Sequence[GenericCallableAccessibleObject],
        module_code: str,
        module_path: str,
        max_uncovered_callables: int = 5,  # TODO (Oetgin): Find good threshold
    ):
        """Initializes the prompt.

        Args:
            callables (list[GenericCallableAccessibleObject]): List of
                uncovered callables.
            module_path (str): Path to the module.
            module_code (str): Source code of the module.
            max_uncovered_callables (int):
                Maximum number of uncovered callables to include in the prompt.
        """
        super().__init__(module_code, module_path)
        self.callables: Sequence[GenericCallableAccessibleObject] = callables
        self.module_path = module_path
        self.module_code = module_code
        self.max_uncovered_callables = max_uncovered_callables

    def build_callables_prompt_section(self) -> list[str]:
        """Generates a list of function headers and their signatures.

        Returns:
            list[str]: A list of formatted function headers with their signatures.
        """
        callables_list = []

        for gao in self.callables:
            signature = str(gao.inferred_signature)
            if gao.is_method() and isinstance(gao, GenericMethod):
                method_gao: GenericMethod = gao
                callable_list_item = (
                    f"- The method {method_gao.method_name} of class "
                    f"{method_gao.owner.name}{signature}"
                )
            elif gao.is_function() and isinstance(gao, GenericFunction):
                fn_gao: GenericFunction = gao
                callable_list_item = f"- The function {fn_gao.function_name}{signature}"
            elif gao.is_constructor() and isinstance(gao, GenericConstructor):
                constructor_gao: GenericConstructor = gao
                class_name = constructor_gao.owner.name  # type: ignore[union-attr]
                callable_list_item = f"- The constructor of the class {class_name}{signature}"
            else:
                continue  # Skip unknown callable types

            callables_list.append(callable_list_item)

        return callables_list

    def _get_gao_name(self, gao: GenericCallableAccessibleObject) -> str | None:
        """Returns the name of the GenericCallableAccessibleObject.

        Args:
            gao (GenericCallableAccessibleObject): The callable object.

        Returns:
            str | None: The name of the callable, or None if the type is unknown.
        """
        if gao.is_method() and isinstance(gao, GenericMethod):
            method_gao: GenericMethod = gao
            return method_gao.method_name
        if gao.is_function() and isinstance(gao, GenericFunction):
            fn_gao: GenericFunction = gao
            return fn_gao.function_name
        if gao.is_constructor() and isinstance(gao, GenericConstructor):
            constructor_gao: GenericConstructor = gao
            return constructor_gao.owner.name  # type: ignore[union-attr]
        return None

    def _compress_module_code(
        self, module_code: str, uncovered_callables: Sequence[GenericCallableAccessibleObject]
    ) -> str:
        """Compresses the module code by removing code unrelated to the uncovered callables.

        Args:
            module_code (str): The source code of the module.
            uncovered_callables (Sequence[GenericCallableAccessibleObject]):
                Sequence of uncovered callables.

        Returns:
            str: The compressed module code.
        """
        uncovered_names = {
            self._get_gao_name(gao)
            for gao in uncovered_callables
            if self._get_gao_name(gao) is not None
        }
        module_ast = ast.parse(module_code)

        return self._compress_node(module_ast, uncovered_names)

    ELLIPSIS = "\n# [...]\n\n"

    def _last_line_is_ellipsis(self, s: str) -> bool:
        lines = s.strip().splitlines()
        return lines[-1].strip() == UncoveredTargetsPrompt.ELLIPSIS.strip()

    def _compress_class_node(
        self, class_node: ast.ClassDef, uncovered_names: set[str | None]
    ) -> str:
        compressed = ""

        for child in class_node.body:
            if not (
                isinstance(child, ast.FunctionDef) and child.name == "__init__"
            ):  # Skip constructor, as it is handled separately
                compressed_child = self._compress_node(child, uncovered_names)
                if compressed_child and compressed_child != UncoveredTargetsPrompt.ELLIPSIS:
                    compressed += compressed_child
                elif not compressed or not self._last_line_is_ellipsis(compressed):
                    compressed += UncoveredTargetsPrompt.ELLIPSIS

        class_signature = f"class {class_node.name}:\n"

        if class_node.name in uncovered_names:
            class_constructor = next(
                (
                    child
                    for child in class_node.body
                    if isinstance(child, ast.FunctionDef) and child.name == "__init__"
                ),
                None,
            )
            if class_constructor:
                class_constructor_code = ast.unparse(class_constructor) + "\n"
                return (
                    class_signature
                    + textwrap.indent(class_constructor_code, "    ")
                    + textwrap.indent(compressed, "    ")
                )
            return class_signature + compressed
        if compressed and compressed != UncoveredTargetsPrompt.ELLIPSIS:
            return class_signature + compressed

        return compressed

    def _compress_node(self, node: ast.AST, uncovered_names: set[str | None]) -> str:
        """Recursively compresses the AST nodes based on uncovered names.

        Args:
            node (ast.AST): Node to compress.
            uncovered_names (set[str | None]): Name of the uncovered callables.

        Returns:
            str: The compressed code representing the node.
        """
        compressed = ""
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in uncovered_names
        ):
            # TODO (Oetgin): Consider including related functions and classes, not just the
            # uncovered ones
            return ast.unparse(node) + "\n"  # Nested functions and classes are included.

        if isinstance(node, ast.ClassDef):
            compressed = self._compress_class_node(node, uncovered_names)

        if isinstance(node, ast.Module):
            for child in node.body:
                compressed_child = self._compress_node(child, uncovered_names)
                if compressed_child and compressed_child != UncoveredTargetsPrompt.ELLIPSIS:
                    compressed += compressed_child
                elif not compressed or not self._last_line_is_ellipsis(compressed):
                    compressed += UncoveredTargetsPrompt.ELLIPSIS

        return compressed

    def build_prompt(self) -> str:
        """Builds the prompt message."""
        if len(self.callables) > self.max_uncovered_callables:
            self.callables = self.callables[: self.max_uncovered_callables]

        callables_list = self.build_callables_prompt_section()
        callables_section = "\n".join(callables_list)

        # TODO (Oetgin): Finalize and test prompt compression

        compressed_module_code = self._compress_module_code(self.module_code, self.callables)

        return f"""
You are writing tests to be used as seed for a SBST algorithm. Your goal is to improve as much as possible the coverage of the tests.
Write unit tests for the following callables that Pynguin failed to cover:
{callables_section}
Module path: `{self.module_path}`
Module source code:
```python
{compressed_module_code}
```
You answer will be parsed for mutations, so here are the guidelines you need to follow:
- Answer in a code block only using; one function for each test case, with NO ARGUMENTS, NO HELPER FUNCTIONS AND NO CLASSES.
- If needed, instantiate vars in the body of the test func or use pytest.parametrize, but DO NOT USE ANY OTHER PYTEST FEATURE (e.g. DO NOT USE FIXTURES), as that will make the parsing fail.
- Do not rewrite the SUT's code in the tests. If you want for example to call a function or instanciante a class, import it.
- Think first then answer in simple, concise assertion tests, split in small functions. Follow the Arrange, Act, Assert pattern.

Here are some examples; *NEVER DO*:
def func_all_tests(param):
    var0 = param
    ...

*INSTEAD DO*:
I need to test the function `func` that takes a parameter and returns a value. I will test the following scenarios: ...
```python
from module_to_test import func
def test_feat1():
    var0 = ...
    var1 = func(var0)
    assert ...
```

REMEMBER: NO ARGUMENTS IN YOUR TEST FUNCTIONS (if you don't use pytest.parametrize)"
"""  # noqa: E501
