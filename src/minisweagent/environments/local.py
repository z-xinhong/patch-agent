import os
import platform
import signal
import subprocess
import time
from threading import Event
from typing import Any

from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge


class LocalEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30


class LocalEnvironment:
    def __init__(self, *, config_class: type = LocalEnvironmentConfig, **kwargs):
        """This class executes bash commands directly on the local machine."""
        self.config = config_class(**kwargs)

    def execute(
        self, action: dict, cwd: str = "", *,
        timeout: int | None = None, check_submission: bool = True, cancel_event: Event | None = None,
    ) -> dict[str, Any]:
        """Execute a command in the local environment and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd or os.getcwd()
        try:
            result = _run(command, cwd, os.environ | self.config.env, timeout or self.config.timeout, cancel_event)
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        if check_submission:
            self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), os.environ, kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


class CommandCancelled(subprocess.SubprocessError):
    def __init__(self, output: str):
        self.output = output
        super().__init__("Command cancelled.")


def _run(
    command: str | list[str],
    cwd: str | None,
    env: dict[str, str] | None,
    timeout: int,
    cancel_event: Event | None = None,
) -> subprocess.CompletedProcess[str]:
    """Collect combined output and kill the command group on timeout or cancellation."""
    if cancel_event is not None and cancel_event.is_set():
        raise CommandCancelled("")
    process = subprocess.Popen(
        command,
        shell=isinstance(command, str),
        text=True,
        cwd=cwd,
        env=env,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=os.name == "posix",
    )
    try:
        if cancel_event is None:
            stdout, _ = process.communicate(timeout=timeout)
        else:
            deadline = time.monotonic() + timeout
            while not cancel_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    stdout, _ = process.communicate(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            else:
                raise CommandCancelled("")
    except (CommandCancelled, subprocess.TimeoutExpired) as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
        except ProcessLookupError:
            pass
        stdout, _ = process.communicate()
        exc.output = stdout
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout=stdout)
