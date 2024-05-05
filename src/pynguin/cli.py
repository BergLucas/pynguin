#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Pynguin is an automated unit test generation framework for Python.

This module provides the main entry location for the program execution from the command
line.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys

from argparse import SUPPRESS
from argparse import Action
from argparse import ArgumentError
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

import simple_parsing

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

import pynguin.configuration as config

from pynguin.__version__ import __version__
from pynguin.generator import run_pynguin
from pynguin.generator import set_configuration
from pynguin.generator import set_plugins


if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType


class _PynguinHelpAction(Action):
    """Custom help action for the argument parser."""

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str = SUPPRESS,
        default: str = SUPPRESS,
        help: str | None = None,  # noqa: A002
    ):
        """Initialize the help action.

        Args:
            option_strings: The option strings
            dest: The destination
            default: The default value
            help: The help message
        """
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help,
        )

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[str] | None,
        option_string: str | None = None,
    ):
        """Print help message and exit.

        Args:
            parser: The argument parser
            namespace: The namespace
            values: The values
            option_string: The option string
        """
        _setup_logging(namespace.verbosity, namespace.no_rich)
        plugins = _setup_plugins(namespace.plugins)
        _run_plugins_parser_hook(plugins, parser)
        parser.print_help()
        parser.exit()


def _create_argument_parser() -> ArgumentParser:
    parser = simple_parsing.ArgumentParser(
        add_option_string_dash_variants=simple_parsing.DashVariant.UNDERSCORE_AND_DASH,
        description="Pynguin is an automatic unit test generation framework for Python",
        fromfile_prefix_chars="@",
        add_help=False,
    )
    parser.register("action", "help", _PynguinHelpAction)
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=SUPPRESS,
        help="show this help message and exit",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s " + __version__
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        dest="verbosity",
        default=0,
        help="verbose output (repeat for increased verbosity)",
    )
    parser.add_argument(
        "--no-rich",
        "--no_rich",
        "--poor",  # hehe
        dest="no_rich",
        action="store_true",
        default=False,
        help="Don't use rich for nicer consoler output.",
    )
    parser.add_argument(
        "--plugins",
        metavar="list",
        type=str,
        nargs="+",
        dest="plugins",
        default=[],
        help="List of plugins to load",
    )
    parser.add_arguments(config.Configuration, dest="config")

    return parser


def _expand_arguments_if_necessary(arguments: list[str]) -> list[str]:
    """Expand command-line arguments, if necessary.

    This is a hacky way to pass comma separated output variables.  The reason to have
    this is an issue with the automatically-generated bash scripts for Pynguin cluster
    execution, for which I am not able to solve the (I assume) globbing issues.  This
    function allows to provide the output variables either separated by spaces or by
    commas, which works as a work-around for the aforementioned problem.

    This function replaces the commas for the ``--output-variables`` parameter and
    the ``--coverage-metrics`` by spaces that can then be handled by the argument-
    parsing code.

    Args:
        arguments: The list of command-line arguments
    Returns:
        The (potentially) processed list of command-line arguments
    """
    if (
        "--output_variables" not in arguments
        and "--output-variables" not in arguments
        and "--coverage_metrics" not in arguments
        and "--coverage-metrics" not in arguments
    ):
        return arguments
    if "--output_variables" in arguments:
        arguments = _parse_comma_separated_option(arguments, "--output_variables")
    elif "--output-variables" in arguments:
        arguments = _parse_comma_separated_option(arguments, "--output-variables")

    if "--coverage_metrics" in arguments:
        arguments = _parse_comma_separated_option(arguments, "--coverage_metrics")
    elif "--coverage-metrics" in arguments:
        arguments = _parse_comma_separated_option(arguments, "--coverage-metrics")
    return arguments


def _parse_comma_separated_option(arguments: list[str], option: str) -> list[str]:
    index = arguments.index(option)
    if "," not in arguments[index + 1]:
        return arguments
    variables = arguments[index + 1].split(",")
    return arguments[: index + 1] + variables + arguments[index + 2 :]


def _setup_output_path(output_path: str) -> None:
    path = Path(output_path).resolve()
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def _setup_logging(verbosity: int, no_rich: bool) -> Console | None:  # noqa: FBT001
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    if verbosity >= 2:
        level = logging.DEBUG

    console = None
    if no_rich:
        handler: logging.Handler = logging.StreamHandler()
    else:
        install()
        console = Console(tab_size=4)
        handler = RichHandler(
            rich_tracebacks=True, log_time_format="[%X]", console=console
        )
        handler.setFormatter(logging.Formatter("%(message)s"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s]"
        "(%(name)s:%(funcName)s:%(lineno)d): %(message)s",
        datefmt="[%X]",
        handlers=[handler],
    )
    return console


def _setup_plugins(plugin_paths: list[str]) -> list[ModuleType]:
    plugins: list[ModuleType] = []

    for plugin_path in plugin_paths:
        plugin_path_object = Path(plugin_path).resolve()

        if plugin_path_object.is_file():
            plugin_module_name = plugin_path_object.stem
            sys.path.insert(0, plugin_path_object.parent.as_posix())
        elif plugin_path_object.is_dir():
            plugin_module_name = plugin_path_object.name
            sys.path.insert(0, plugin_path_object.parent.as_posix())
        else:
            plugin_module_name = plugin_path

        try:
            plugin_module = importlib.import_module(plugin_module_name)
        except ImportError:
            logging.exception(
                "Could not load plugin %s",
                plugin_path,
            )
            continue

        try:
            plugin_name = plugin_module.NAME
        except AttributeError:
            logging.exception(
                "Plugin %s does not have a NAME attribute",
                plugin_path,
            )
            continue

        plugins.append(plugin_module)
        logging.info(
            'Loaded plugin "%s" (%s)',
            plugin_name,
            plugin_path,
        )

    return plugins


def _run_plugins_parser_hook(plugins: list[ModuleType], parser: ArgumentParser) -> None:
    for plugin in plugins:
        try:
            parser_hook = plugin.parser_hook
        except AttributeError:
            logging.debug(
                'Plugin "%s" does not have a parser_hook attribute',
                plugin.NAME,
            )
            continue

        try:
            parser_hook(parser)
        except BaseException:
            logging.exception(
                'Failed to run parser_hook for plugin "%s"',
                plugin.NAME,
            )
            continue

        logging.info('Added plugin "%s" arguments', plugin.NAME)


def _run_plugins_configuration_hook(
    plugins: list[ModuleType], plugin_config: Namespace
) -> None:
    for plugin in plugins:
        try:
            configuration_hook = plugin.configuration_hook
        except AttributeError:
            logging.debug(
                'Plugin "%s" does not have a configuration_hook attribute',
                plugin.NAME,
            )
            continue

        try:
            configuration_hook(plugin_config)
        except BaseException:
            logging.exception(
                'Failed to run configuration_hook for plugin "%s"',
                plugin.NAME,
            )
            continue

        logging.info('Configured plugin "%s"', plugin.NAME)


# People may wipe their disk, so we give them a heads-up.
_DANGER_ENV = "PYNGUIN_DANGER_AWARE"


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI of the Pynguin automatic unit test generation framework.

    This method behaves like a standard UNIX command-line application, i.e.,
    the return value `0` signals a successful execution.  Any other return value
    signals some errors.  This is, e.g., the case if the framework was not able
    to generate one successfully running test case for the class under test.

    Args:
        argv: List of command-line arguments

    Returns:
        An integer representing the success of the program run.  0 means
        success, all non-zero exit codes indicate errors.
    """
    if _DANGER_ENV not in os.environ:
        print(  # noqa: T201
            f"""Environment variable '{_DANGER_ENV}' not set.
Aborting to avoid harming your system.
Please refer to the documentation
(https://pynguin.readthedocs.io/en/latest/user/quickstart.html)
to see why this happens and what you must do to prevent it."""
        )
        return -1

    if argv is None:
        argv = sys.argv
    if len(argv) <= 1:
        argv.append("--help")
    argv = _expand_arguments_if_necessary(argv[1:])

    argument_parser = _create_argument_parser()
    parsed, plugin_argv = argument_parser.parse_known_args(argv)

    console = _setup_logging(parsed.verbosity, parsed.no_rich)

    plugins = _setup_plugins(parsed.plugins)

    # We need another parser because using the previous one with only the remaining
    # arguments would not work because of the required arguments
    plugin_parser = ArgumentParser(add_help=False, exit_on_error=False)

    _run_plugins_parser_hook(plugins, plugin_parser)

    # Run the parser hooks on the argument parser for a better help message
    logging.root.disabled = True
    _run_plugins_parser_hook(plugins, argument_parser)
    logging.root.disabled = False

    try:
        plugin_parsed, remaining_argv = plugin_parser.parse_known_args(plugin_argv)
    except ArgumentError as error:
        argument_parser.error(str(error))
        return -1

    if remaining_argv:
        msg = "unrecognized arguments: %s"
        argument_parser.error(msg % " ".join(remaining_argv))
        return -1

    _run_plugins_configuration_hook(plugins, plugin_parsed)

    _setup_output_path(parsed.config.test_case_output.output_path)

    set_configuration(parsed.config)
    set_plugins(plugins)
    if console is not None:
        with console.status("Running Pynguin..."):
            return run_pynguin().value
    else:
        return run_pynguin().value


if __name__ == "__main__":
    sys.exit(main(sys.argv))
