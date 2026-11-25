import asyncio
from collections.abc import Callable, Coroutine, Iterator
from concurrent.futures import TimeoutError
from contextlib import contextmanager
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from typing import TypeVar

from minisweagent.exceptions import InterruptAgentFlow

_T = TypeVar("_T")
_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = Lock()


class TurnCancelled(InterruptAgentFlow):
    def __init__(self) -> None:
        super().__init__({
            "role": "exit", "content": "Turn cancelled.",
            "extra": {"exit_status": "Cancelled", "submission": ""},
        })


def check_cancelled(cancel: Event) -> None:
    if cancel.is_set():
        raise TurnCancelled()


def wait_for_model(call: Callable[[], Coroutine[object, object, _T]], cancel: Event) -> _T:
    check_cancelled(cancel)
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            Thread(target=_loop.run_forever, daemon=True, name="workbench-models").start()
    future = asyncio.run_coroutine_threadsafe(call(), _loop)
    try:
        while not future.done():
            check_cancelled(cancel)
            try:
                future.result(timeout=0.1)
            except TimeoutError:
                if future.done():
                    break
        check_cancelled(cancel)
        return future.result()
    finally:
        future.cancel()


class WorkspaceCommands:
    def __init__(self) -> None:
        self.condition = Condition()
        self.active: set[Path] = set()

    @contextmanager
    def acquire(self, workspace: Path, cancel: Event) -> Iterator[None]:
        with self.condition:
            while any(workspace.is_relative_to(root) or root.is_relative_to(workspace) for root in self.active):
                check_cancelled(cancel)
                self.condition.wait(timeout=0.1)
            check_cancelled(cancel)
            self.active.add(workspace)
        try:
            yield
        finally:
            with self.condition:
                self.active.remove(workspace)
                self.condition.notify_all()
