#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2025 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides instrumentation mechanisms using sys.monitoring."""

from __future__ import annotations

import inspect
import sys


if sys.version_info < (3, 12):
    # sys.monitoring is only available in Python 3.12 and later
    raise ImportError("sys.monitoring is not available in Python versions < 3.12")

from collections import defaultdict
from enum import Enum
from functools import wraps
from types import CodeType
from typing import TYPE_CHECKING
from typing import Concatenate
from typing import ParamSpec
from typing import Protocol
from typing import TypeAlias
from typing import cast

from pynguin.instrumentation import InstrumentationTransformer
from pynguin.instrumentation import get_basic_block


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType

    from pynguin.analyses.controlflow import CFG
    from pynguin.analyses.controlflow import ProgramGraphNode
    from pynguin.instrumentation.tracer import InstrumentationExecutionTracer


class MonitoringEvent(Enum):
    """Enumeration of monitoring events."""

    PY_START = sys.monitoring.events.PY_START
    PY_RESUME = sys.monitoring.events.PY_RESUME
    INSTRUCTION = sys.monitoring.events.INSTRUCTION
    PY_RETURN = sys.monitoring.events.PY_RETURN
    PY_YIELD = sys.monitoring.events.PY_YIELD
    CALL = sys.monitoring.events.CALL
    C_RAISE = sys.monitoring.events.C_RAISE
    C_RETURN = sys.monitoring.events.C_RETURN
    RAISE = sys.monitoring.events.RAISE
    RERAISE = sys.monitoring.events.RERAISE
    EXCEPTION_HANDLED = sys.monitoring.events.EXCEPTION_HANDLED
    PY_UNWIND = sys.monitoring.events.PY_UNWIND
    PY_THROW = sys.monitoring.events.PY_THROW
    STOP_ITERATION = sys.monitoring.events.STOP_ITERATION
    BRANCH = sys.monitoring.events.BRANCH
    JUMP = sys.monitoring.events.JUMP
    LINE = sys.monitoring.events.LINE


class InstructionCallback(Protocol):
    """A function that is called in case of PY_INSTRUCTION, PY_START and PY_RESUME events."""

    def __call__(self, frame: FrameType) -> None:
        """Handle the event for the given frame.

        Args:
            frame: The frame object associated with the event.
        """


class ReturnInstructionCallback(Protocol):
    """A function that is called in case of PY_RETURN and PY_YIELD events."""

    def __call__(self, frame: FrameType, retval: object) -> None:
        """Handle the event for the given frame and return value.

        Args:
            frame: The frame object associated with the event.
            retval: The return value of the function.
        """


class CallInstructionCallback(Protocol):
    """A function that is called in case of CALL, C_RAISE and C_RETURN events."""

    def __call__(self, frame: FrameType, callable: object) -> None:  # noqa: A002
        """Handle the event for the given frame and callable.

        Args:
            frame: The frame object associated with the event.
            callable: The callable object that was called (e.g., a function or method).
        """


class ExceptionInstructionCallback(Protocol):
    """A function that is called in case of RAISE, RERAISE, EXCEPTION_HANDLED, PY_UNWIND, PY_THROW and STOP_ITERATION events."""  # noqa: E501

    def __call__(self, frame: FrameType, exception: BaseException) -> None:
        """Handle the event for the given frame and exception.

        Args:
            frame: The frame object associated with the event.
            exception: The exception that was raised or handled.
        """


class BranchInstructionCallback(Protocol):
    """A function that is called in case of BRANCH and JUMP events."""

    def __call__(self, frame: FrameType, destination_offset: int) -> None:
        """Handle the event for the given frame and destination offset.

        Args:
            frame: The frame object associated with the event.
            destination_offset: The offset in the bytecode to which the branch or jump is made.
        """


class LineCallback(Protocol):
    """A function that is called in case of LINE events."""

    def __call__(self, frame: FrameType) -> None:
        """Handle the event for the given frame and line number.

        Args:
            frame: The frame object associated with the event.
        """


MONITORING_CALLBACKS: TypeAlias = (
    InstructionCallback
    | ReturnInstructionCallback
    | CallInstructionCallback
    | ExceptionInstructionCallback
    | BranchInstructionCallback
    | LineCallback
)


P = ParamSpec("P")


def monitoring_callback(
    func: Callable[Concatenate[FrameType, P], None],
) -> Callable[P, None]:
    """Decorator to wrap a function as a monitoring callback.

    Args:
        func: The function to be wrapped.

    Returns:
        A wrapper function that add a frame argument to the original function.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        current_frame = inspect.currentframe()
        assert current_frame is not None
        instrumented_frame = current_frame.f_back
        assert instrumented_frame is not None

        func(instrumented_frame, *args, **kwargs)

    return wrapper


class EventMonitor:
    """A class to monitor events using sys.monitoring."""

    def __init__(
        self, tool_id: int = sys.monitoring.COVERAGE_ID, tool_name: str = "pynguin"
    ) -> None:
        """Initialize the event monitor.

        Args:
            tool_id: The tool ID for sys.monitoring events.
            tool_name: The name of the tool for sys.monitoring.
        """
        self.tool_id = tool_id
        self.tool_name = tool_name
        self._callbacks: dict[
            CodeType,
            dict[MonitoringEvent, dict[int, list[MONITORING_CALLBACKS]]],
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def register_callback(
        self,
        code: CodeType,
        event_type: MonitoringEvent,
        instruction_offset: int,
        callback: MONITORING_CALLBACKS,
    ) -> None:
        """Register a callback for a specific code object and event type.

        Args:
            code: The code object to register the callback for.
            event_type: The type of event to monitor.
            instruction_offset: The instruction offset within the code object.
            callback: The callback function to be called when the event occurs.
        """
        self._callbacks[code][event_type][instruction_offset].append(callback)

    def enable(self) -> None:
        """Enable monitoring for all registered callbacks."""
        sys.monitoring.use_tool_id(self.tool_id, self.tool_name)

        for event in (
            MonitoringEvent.PY_START,
            MonitoringEvent.PY_RESUME,
            MonitoringEvent.INSTRUCTION,
        ):
            self._enable_instruction_callback(event)

        for event in (
            MonitoringEvent.PY_RETURN,
            MonitoringEvent.PY_YIELD,
        ):
            self._enable_return_callback(event)

        for event in (
            MonitoringEvent.CALL,
            MonitoringEvent.C_RAISE,
            MonitoringEvent.C_RETURN,
        ):
            self._enable_call_callback(event)

        for event in (
            MonitoringEvent.RAISE,
            MonitoringEvent.RERAISE,
            MonitoringEvent.EXCEPTION_HANDLED,
            MonitoringEvent.PY_UNWIND,
            MonitoringEvent.PY_THROW,
            MonitoringEvent.STOP_ITERATION,
        ):
            self._enable_exception_callback(event)

        self._enable_line_callback(MonitoringEvent.LINE)

        for event in (MonitoringEvent.BRANCH, MonitoringEvent.JUMP):
            self._enable_branch_callback(event)

        for code, events in self._callbacks.items():
            sys.monitoring.set_local_events(
                self.tool_id,
                code,
                sum(event.value for event in events),
            )

    def _enable_callback(
        self,
        event: MonitoringEvent,
        callback: Callable[..., None],
    ) -> None:
        sys.monitoring.register_callback(
            self.tool_id,
            event.value,
            callback,
        )

    def _enable_instruction_callback(
        self,
        event: MonitoringEvent,
    ) -> None:
        @monitoring_callback
        def func(
            frame: FrameType,
            code: CodeType,
            instruction_offset: int,
        ) -> None:
            for callback in cast(
                "list[InstructionCallback]",
                self._callbacks[code][event][instruction_offset],
            ):
                callback(frame)

        self._enable_callback(event, func)

    def _enable_return_callback(
        self,
        event: MonitoringEvent,
    ) -> None:
        @monitoring_callback
        def func(
            frame: FrameType,
            code: CodeType,
            instruction_offset: int,
            retval: object,
        ) -> None:
            for callback in cast(
                "list[ReturnInstructionCallback]",
                self._callbacks[code][event][instruction_offset],
            ):
                callback(frame, retval)

        self._enable_callback(event, func)

    def _enable_call_callback(
        self,
        event: MonitoringEvent,
    ) -> None:
        @monitoring_callback
        def func(
            frame: FrameType,
            code: CodeType,
            instruction_offset: int,
            callable: object,  # noqa: A002
            _: object,
        ) -> None:
            for callback in cast(
                "list[CallInstructionCallback]",
                self._callbacks[code][event][instruction_offset],
            ):
                callback(frame, callable)

        self._enable_callback(event, func)

    def _enable_exception_callback(
        self,
        event: MonitoringEvent,
    ) -> None:
        @monitoring_callback
        def func(
            frame: FrameType,
            code: CodeType,
            instruction_offset: int,
            exception: BaseException,
        ) -> None:
            for callback in cast(
                "list[ExceptionInstructionCallback]",
                self._callbacks[code][event][instruction_offset],
            ):
                callback(frame, exception)

        self._enable_callback(event, func)

    def _enable_line_callback(
        self,
        event: MonitoringEvent,
    ) -> None:
        @monitoring_callback
        def func(
            frame: FrameType,
            code: CodeType,
            line_number: int,
        ) -> None:
            for callback in cast(
                "list[LineCallback]",
                self._callbacks[code][event][line_number],
            ):
                callback(frame)

        self._enable_callback(event, func)

    def _enable_branch_callback(
        self,
        event: MonitoringEvent,
    ) -> None:
        @monitoring_callback
        def func(
            frame: FrameType,
            code: CodeType,
            instruction_offset: int,
            destination_offset: int,
        ) -> None:
            for callback in cast(
                "list[BranchInstructionCallback]",
                self._callbacks[code][event][instruction_offset],
            ):
                callback(frame, destination_offset)

        self._enable_callback(event, func)


class MonitoringInstrumentationAdapter:
    """Abstract base class for sys.monitoring injection instrumentation adapters."""

    def visit_entry_node(
        self,
        event_monitor: EventMonitor,
        code: CodeType,
        node: ProgramGraphNode,
        code_object_id: int,
    ) -> None:
        """Called when we visit the entry node of a code object.

        Args:
            event_monitor: The event monitor to register callbacks with.
            code: The code object of the entry node.
            node: The entry node of the control flow graph.
            code_object_id: The code object id of the containing code object.
        """

    def visit_node(
        self,
        event_monitor: EventMonitor,
        code: CodeType,
        cfg: CFG,
        code_object_id: int,
        node: ProgramGraphNode,
    ) -> None:
        """Called for each node that have a basic block.

        Args:
            event_monitor: The event monitor to register callbacks with.
            code: The code object of the node.
            cfg: The control flow graph.
            code_object_id: The code object id of the containing code object.
            node: The node in the control flow graph.
        """


class MonitoringInstrumentationTransformer(InstrumentationTransformer):
    """Applies a given list of monitoring instrumentation adapters to code objects."""

    def __init__(  # noqa: D107
        self,
        instrumentation_tracer: InstrumentationExecutionTracer,
        instrumentation_adapters: list[MonitoringInstrumentationAdapter],
    ):
        super().__init__(instrumentation_tracer)
        self._instrumentation_adapters = instrumentation_adapters

    def instrument_module(self, module_code: CodeType) -> CodeType:  # noqa: D102
        self._check_module_not_instrumented(module_code)

        event_monitor = EventMonitor()

        def create_instrumented_code(
            code: CodeType,
            cfg: CFG,
            code_object_id: int,
            entry_node: ProgramGraphNode,
        ) -> CodeType:
            for adapter in self._instrumentation_adapters:
                adapter.visit_entry_node(event_monitor, code, entry_node, code_object_id)

            for node in cfg.nodes:
                if node.is_artificial:
                    # Artificial nodes don't have a basic block, so we don't need to
                    # instrument anything.
                    continue

                for adapter in self._instrumentation_adapters:
                    adapter.visit_node(event_monitor, code, cfg, code_object_id, node)

            for const in code.co_consts:
                if isinstance(const, CodeType):
                    self._instrument_code_recursive(
                        const,
                        create_instrumented_code,
                        code_object_id,
                    )

            return code

        instrumented_code = self._instrument_code_recursive(
            module_code,
            create_instrumented_code,
        )

        event_monitor.enable()

        return instrumented_code


class LineCoverageMonitoringInstrumentation(MonitoringInstrumentationAdapter):
    """Instruments code objects to enable tracking of executed lines.

    This results in line coverage.
    """

    def __init__(  # noqa: D107
        self, instrumentation_tracer: InstrumentationExecutionTracer
    ) -> None:
        self._instrumentation_tracer = instrumentation_tracer

    def visit_node(  # noqa: D102
        self,
        event_monitor: EventMonitor,
        code: CodeType,
        cfg: CFG,
        code_object_id: int,
        node: ProgramGraphNode,
    ) -> None:
        basic_block = get_basic_block(node)
        file_name = cfg.bytecode_cfg().filename

        lineno: int = basic_block[0].lineno  # type: ignore[union-attr,arg-type,unused-ignore]
        for instr in basic_block:
            if instr.lineno != lineno:  # type: ignore[union-attr,arg-type,unused-ignore]
                lineno = instr.lineno  # type: ignore[union-attr,arg-type,unused-ignore]

                line_id = self._instrumentation_tracer.register_line(
                    code_object_id,
                    file_name,
                    lineno,  # type: ignore[union-attr,arg-type,unused-ignore]
                )

                self._instrument_line(
                    event_monitor,
                    code,
                    line_id,
                    lineno,  # type: ignore[union-attr,arg-type,unused-ignore]
                )

    def _instrument_line(
        self,
        event_monitor: EventMonitor,
        code: CodeType,
        line_id: int,
        lineno: int,
    ) -> None:
        def callback(frame: FrameType) -> None:  # noqa: ARG001
            self._instrumentation_tracer.track_line_visit(line_id)

        event_monitor.register_callback(
            code,
            MonitoringEvent.LINE,
            lineno,
            callback,
        )
