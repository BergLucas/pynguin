#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2026 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides a basic API to communicate with LLMs."""

from __future__ import annotations

import abc
import logging
import re
import typing

from pynguin.configuration import LLMProvider
from pynguin.utils.api_key_resolver import (
    get_llm_url,
    get_model_name,
    require_api_key,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable


try:
    import openai
    from openai.types.chat import (
        ChatCompletionAssistantMessageParam,
        ChatCompletionDeveloperMessageParam,
        ChatCompletionFunctionMessageParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionToolMessageParam,
        ChatCompletionUserMessageParam,
    )

    if typing.TYPE_CHECKING:
        from pydantic import SecretStr

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


LOGGER = logging.getLogger(__name__)


class LLM(abc.ABC):
    """An abstract interface for LLM communications."""

    def __init__(
        self,
        api_key: SecretStr | None,
        temperature: float,
        system_prompt: str,
        model: str | None = None,
    ) -> None:
        """Initialises the LLM communication interface.

        Args:
            api_key: the API key to authenticate with the LLM
            temperature: the temperature setting for the LLM
            system_prompt: the system prompt for the LLM
            model: the LLM model
        """
        self._api_key = api_key
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._model = model

    @abc.abstractmethod
    def chat(
        self, prompt: str, system_prompt: str | None = None
    ) -> tuple[str | None, dict[str, int]]:
        """Sends a message to the LLM and returns its answer.

        Args:
            prompt: the (user) prompt send to the LLM
            system_prompt: the system prompt send to the LLM, if empty use the one from the
                           constructor of this class.

        Returns:
            Either a tuple containing :

                - The raw answer from the LLM
                - The usage, a dict containing
                    prompt tokens (`"prompt_tokens"`)
                    and completion tokens (`"completion_tokens"`)

            or `None`
        """

    @classmethod
    def create(cls, provider: LLMProvider, **kwargs) -> LLM:
        """Creates the LLM communication interface based on the given provider.

        Args:
            provider: the provider of the LLM
            **kwargs: optionals arguments to be passed for the creation of the client.
                See :class:`LLM`.

        Returns:
            The concrete LLM communication interface
        """
        match provider:
            case LLMProvider.OPENAI:
                if not OPENAI_AVAILABLE:
                    raise ValueError(
                        "OpenAI API library is not available. You can install it with poetry "
                        "install --with openai."
                    )
                return OpenAI(**kwargs)
            case LLMProvider.OLLAMA:
                if not OLLAMA_AVAILABLE:
                    raise ValueError(
                        "Ollama API library is not available. You can install it with poetry "
                        "install --with ollama."
                    )
                return Ollama(**kwargs)
            case _:
                raise NotImplementedError(f"Unknown provider {provider}")

    @property
    @abc.abstractmethod
    def response_error(self) -> type[BaseException]:
        """The class of the exceptions raised when an error is encountered during LLM response."""


def extract_code(llm_response: str) -> str:
    """Takes the response from the LLM and attempts to extract the answer.

    Args:
        llm_response: the response from the LLM

    Returns:
        the extracted answer, i.e., the extracted pytest code
    """
    md_source_block_pattern = r"^```(?:\w+)?\s*\n(.*?)(?=^```)```"
    result = re.findall(md_source_block_pattern, llm_response, re.DOTALL | re.MULTILINE)
    return "\n".join(result)


if OPENAI_AVAILABLE:
    OPENAI_SYSTEM_PROMPT = """You are a senior level Python developer with a focus on testing
    with the pytest framework. Provide the generated tests in the style of the pytest framework.
    Provide the generated tests inside a Markdown-style code block."""

    MessageTypes: typing.TypeAlias = (
        ChatCompletionDeveloperMessageParam  # pyright: ignore[reportPossiblyUnboundVariable]
        | ChatCompletionSystemMessageParam  # pyright: ignore[reportPossiblyUnboundVariable]
        | ChatCompletionUserMessageParam  # pyright: ignore[reportPossiblyUnboundVariable]
        | ChatCompletionAssistantMessageParam  # pyright: ignore[reportPossiblyUnboundVariable]
        | ChatCompletionToolMessageParam  # pyright: ignore[reportPossiblyUnboundVariable]
        | ChatCompletionFunctionMessageParam  # pyright: ignore[reportPossiblyUnboundVariable]
    )

    class OpenAI(LLM):
        """An interface for communication with OpenAI."""

        def __init__(  # noqa: D107
            self,
            api_key: SecretStr | None = None,
            temperature: float = 0.2,
            system_prompt: str = OPENAI_SYSTEM_PROMPT,
            model: str | None = None,
        ) -> None:
            if api_key is None or not api_key.get_secret_value():
                api_key = require_api_key()
            super().__init__(api_key, temperature, system_prompt)
            llm_url = get_llm_url()
            kwargs: dict = {"api_key": api_key.get_secret_value()}
            if llm_url:
                kwargs["base_url"] = llm_url
            self.__client = openai.OpenAI(  # pyright: ignore[reportPossiblyUnboundVariable]
                **kwargs
            )
            self.__model = model or get_model_name()

        def chat(  # noqa: D102
            self, prompt: str, system_prompt: str | None = None
        ) -> tuple[str | None, dict[str, int]]:
            if not system_prompt:
                system_prompt = self._system_prompt

            messages: Iterable[MessageTypes] = [
                ChatCompletionSystemMessageParam(  # pyright: ignore[reportPossiblyUnboundVariable]
                    content=system_prompt, role="system"
                ),
                ChatCompletionUserMessageParam(  # pyright: ignore[reportPossiblyUnboundVariable]
                    content=prompt, role="user"
                ),
            ]
            try:
                response = self.__client.chat.completions.create(
                    messages=messages,
                    model=self.__model,
                )
                usage = {}
                if response.usage and response.usage.prompt_tokens:
                    usage["prompt_tokens"] = response.usage.prompt_tokens
                if response.usage and response.usage.completion_tokens:
                    usage["completion_tokens"] = response.usage.completion_tokens

                return (
                    response.choices[0].message.content,
                    usage,
                )

            except openai.OpenAIError as e:  # pyright: ignore[reportPossiblyUnboundVariable]
                LOGGER.exception(e)
            return (None, {})

        @property
        def response_error(self) -> type[BaseException]:  # noqa: D102
            return openai.APIError  # pyright: ignore[reportPossiblyUnboundVariable]


if OLLAMA_AVAILABLE:
    OLLAMA_SYSTEM_PROMPT = """You are a senior level Python developer with a focus on testing
    with the pytest framework. Provide the generated tests in the style of the pytest framework.
    Provide the generated tests inside a Markdown-style code block."""

    class Ollama(LLM):
        """An interface for communication with Ollama."""

        def __init__(  # noqa: D107
            self,
            api_key: SecretStr | None = None,
            temperature: float = 0.2,
            system_prompt: str = OLLAMA_SYSTEM_PROMPT,
            model: str = "qwen2.5-coder:3b",  # TODO (Oetgin): Benchmark models
        ) -> None:
            super().__init__(api_key, temperature, system_prompt)
            if api_key is None:
                llm_url = get_llm_url()
                if not llm_url:
                    llm_url = "http://localhost:11434"
                self.__client = ollama.Client(host=llm_url)  # pyright: ignore[reportPossiblyUnboundVariable]
            else:
                llm_url = get_llm_url()
                if not llm_url:
                    llm_url = "https://ollama.com"
                self.__client = ollama.Client(  # pyright: ignore[reportPossiblyUnboundVariable]
                    host=llm_url,
                    headers={"Authorization": "Bearer " + api_key.get_secret_value()},
                )
            self.__model = model

        def chat(  # noqa: D102
            self, prompt: str, system_prompt: str | None = None
        ) -> tuple[str | None, dict[str, int]]:
            if not system_prompt:
                system_prompt = self._system_prompt

            messages = [
                {"content": system_prompt, "role": "developer"},
                {"content": prompt, "role": "user"},
            ]
            try:
                response: ollama.ChatResponse = self.__client.chat(
                    messages=messages,
                    model=self.__model,
                )
                usage = {}
                if response.prompt_eval_count is not None:
                    usage["prompt_tokens"] = response.prompt_eval_count
                if response.eval_count is not None:
                    usage["completion_tokens"] = response.eval_count
                return (response.message.content, usage)
            except ollama.ResponseError as e:  # pyright: ignore[reportPossiblyUnboundVariable]
                LOGGER.exception(e)
            return (None, {})

        @property
        def response_error(self) -> type[BaseException]:  # noqa: D102
            return ollama.ResponseError  # pyright: ignore[reportPossiblyUnboundVariable]
