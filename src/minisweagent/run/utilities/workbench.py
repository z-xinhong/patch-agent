"""Run the local browser workbench."""

import json
import logging
import mimetypes
import shutil
import shlex
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import typer
from jinja2 import StrictUndefined, Template
from markdown_it import MarkdownIt

from minisweagent import global_config_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.exceptions import Submitted
from minisweagent.models import get_model, get_model_class, get_model_name
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.content_string import get_content_string
from minisweagent.run.utilities.workbench_changes import WorkspaceSnapshot
from minisweagent.run.utilities.workbench_context import (
    ContextBudget,
    ContextBudgetExceeded,
    ContextConfig,
    WorkbenchModel,
    write_json,
)
from minisweagent.run.utilities.workbench_lifecycle import TurnCancelled, WorkspaceCommands, check_cancelled
from minisweagent.run.utilities.workbench_workspace import EvidenceStore, WorkbenchConfig, Workspace
from minisweagent.utils.serialize import recursive_merge

app = typer.Typer()
_ASSETS = Path(__file__).with_name("workbench_assets")
_SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
_COMPLETION_RULE = (
    "When the user asks for explanation or inspection, only do the requested work. "
    "When you have enough information or have finished the requested change, return your final answer as plain "
    "text without tools. You do not need to run a submission command. This completion rule takes precedence "
    "over the workflow's tool call and submission instructions."
)
_LOGGER = logging.getLogger(__name__)
_MARKDOWN = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"]).disable("image")
_INTERRUPTED = (
    "The last run was interrupted. Its pending or running commands may not have completed. "
    "Do not assume their effects or replay them automatically. Inspect the current workspace before continuing."
)


@dataclass
class ToolAction:
    label: str
    purpose: str = "command"
    id: str = field(default_factory=lambda: uuid4().hex)


def render_event(event: dict) -> dict:
    if event["kind"] in {"assistant", "finish"} and (text := event.get("text")):
        return event | {"html": _MARKDOWN.render(text)}
    return event


def _read_json(path: Path, default: dict) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def _storage_dir(config: dict) -> Path:
    return Path(config.get("workbench", {}).get("storage_dir", global_config_dir / "workbench")).expanduser().resolve()


def _read_events(directory: Path) -> list[dict]:
    path = directory / "events.jsonl"
    if not path.is_file():
        return []
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    events = []
    for index, line in enumerate(lines):
        try:
            if line.strip():
                events.append(json.loads(line))
        except ValueError:
            if index != len(lines) - 1:
                raise
    if raw and not raw.endswith("\n"):
        # A process can die during the last append. Preserve the completed records.
        repaired = path.with_suffix(".repair")
        repaired.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        repaired.replace(path)
    return events


class WorkbenchAgent(DefaultAgent):
    def __init__(self, session: "Session", config: dict):
        self.session = session
        agent_config = {
            k: v
            for k, v in config["agent"].items()
            if k not in {"mode", "confirm_exit", "whitelist_actions", "agent_class"}
        }
        model_config = config.get("model", {})
        name = get_model_name(config=model_config)
        if get_model_class(name, model_config.get("model_class", "")) is LitellmModel:
            model_config = model_config | {"model_class": f"{WorkbenchModel.__module__}.{WorkbenchModel.__name__}"}
        model = get_model(config=model_config)
        if isinstance(model, WorkbenchModel):
            model.context = ContextBudget(session.config.context, session.directory, session.emit)
            model.context.check_source = session.source_status
            model.summarize_tasks = self.summarize_tasks
            model.cancel_event = session.cancel_event
        super().__init__(model, session.workspace.env, **(agent_config | {"output_path": None}))

    def run_turn(self, task: str) -> dict:
        if not self.messages:
            result = self.run(task)
            index = next(i for i, m in enumerate(self.messages) if m.get("extra", {}).get("context_category") == "task")
        else:
            if self.messages[-1].get("role") == "exit":
                self.messages.pop()
            if isinstance(self.model, WorkbenchModel) and not any(
                _COMPLETION_RULE in (m.get("content") or "") for m in self.messages if m.get("role") == "system"
            ):
                self.add_messages(self.model.format_message(role="system", content=_COMPLETION_RULE))
            index = len(self.messages)
            self.extra_template_vars["task"] = task
            self.add_messages(self.model.format_message(
                role="user", content=task, extra={"context_category": "task", "task_text": task},
            ))
            result = self._run_loop()
        if result["exit_status"] == "Submitted":
            verification = self.session.verification
            self.messages[index]["extra"]["completion"] = {
                "end": len(self.messages) - 1, "submission": result.get("submission", ""),
                "verification": verification.get("evidence") if verification["status"] == "passed" else None,
            }
            self.save(None)
        return result

    def summarize_tasks(self, messages: list[dict], reserve: int) -> str:
        check_cancelled(self.session.cancel_event)
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise ValueError("The turn has no remaining budget for summarization.")
        if 0 < self.config.wall_time_limit_seconds <= time.time() - self._start_time:
            raise ValueError("The turn has reached its time limit.")
        self.n_calls += 1
        text, cost = self.model.summarize(messages, reserve)
        self.cost += cost
        return text

    def query(self) -> dict:
        check_cancelled(self.session.cancel_event)
        try:
            return super().query()
        except ContextBudgetExceeded:
            self.n_calls -= 1
            raise

    def add_messages(self, *messages: dict) -> list[dict]:
        if not self.messages and len(messages) == 2 and messages[0].get("role") == "system":
            system, task = messages
            root = "/workspace" if self.session.config.backend == "docker" else str(self.session.root)
            scope = f"Workspace root: {root}. Keep actions in this workspace."
            if isinstance(self.model, WorkbenchModel):
                scope += " Use read_file for source evidence and apply_patch for exact tracked edits when practical."
                scope += " When recall_output is available, use it to retrieve archived observations by evidence ID."
                workflow = Template(self.config.instance_template, undefined=StrictUndefined).render(
                    **self.get_template_vars(task="Follow the user task below.")
                )
                scope += f"\n\n{_COMPLETION_RULE}\n\nFor code changes, use this workflow where relevant:\n{workflow}"
                task = self.model.format_message(role="user", content=self.extra_template_vars["task"])
            system = self.model.format_message(
                role="system", content=f"{get_content_string(system)}\n\n{scope}"
            )
            task_metadata = {"context_category": "task", "task_text": self.extra_template_vars["task"]}
            task = task | {"extra": task.get("extra", {}) | task_metadata}
            if self.session.workspace.rules:
                rules = self.model.format_message(
                    role="system", content="Project instructions (AGENTS.md):\n" + self.session.workspace.rules,
                    extra={"context_category": "project_rules"},
                )
                messages = (system, rules, task)
            else:
                messages = (system, task)
        for message in messages:
            if message.get("extra", {}).get("interrupt_type") == "FormatError":
                self.session.emit("notice", text=get_content_string(message))
            elif message.get("role") == "assistant" or message.get("object") == "response":
                text = get_content_string({k: v for k, v in message.items() if k != "tool_calls" and k != "output"})
                if not text and isinstance(message.get("output"), list):
                    text = get_content_string({"output": [m for m in message["output"] if m.get("type") == "message"]})
                if text:
                    self.session.emit("assistant", text=text)
        if any(message.get("extra", {}).get("context_category") == "task" for message in messages):
            if self.session.resume_notice:
                messages += (self.model.format_message(role="user", content=self.session.resume_notice),)
                self.session.resume_notice = ""
            result = super().add_messages(*messages)
            self.save(None)
            return result
        return super().add_messages(*messages)

    @staticmethod
    def action_label(action: dict) -> str:
        if "command" in action:
            return action["command"]
        return f'{action["tool"]} {json.dumps(action["arguments"], ensure_ascii=False)}'

    def execute_action(self, action: dict, recorded: ToolAction) -> dict:
        if "command" in action:
            with self.session.commands.acquire(self.session.root, self.session.cancel_event):
                self.session.record_action(recorded, "started")
                return self.env.execute(action, cancel_event=self.session.cancel_event)
        tool, arguments = action["tool"], action["arguments"]
        handlers = {
            "read_file": self.session.workspace.read_file,
            "apply_patch": self.session.workspace.apply_patch,
            "recall_output": self.session.evidence.recall,
        }
        started = time.monotonic()
        try:
            if tool != "recall_output":
                with self.session.commands.acquire(self.session.root, self.session.cancel_event):
                    self.session.record_action(recorded, "started")
                    result = handlers[tool](**arguments)
            else:
                self.session.record_action(recorded, "started")
                result = handlers[tool](**arguments)
        except (ValueError, TypeError, OSError) as exc:
            return {"output": "", "returncode": 1, "exception_info": str(exc)}
        result = {"returncode": 0, "exception_info": "", **result}
        if tool == "recall_output":
            evidence = result["extra"]["recalled"]
            if (source := self.session.source_status(evidence)) and source["status"] != "current":
                result["output"] = (
                    f"Historical snapshot of {source['path']} ({evidence}); "
                    f"the current file is {source['status']}. Use read_file for the current content.\n"
                    + result["output"]
                )
            self.session.emit(
                "recall", evidence=evidence, characters=len(result["output"]),
                estimated_tokens=ContextBudget.rough(result["output"]),
                elapsed_ms=round((time.monotonic() - started) * 1000, 2),
            )
        return result

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        if isinstance(self.model, WorkbenchModel) and not actions:
            check_cancelled(self.session.cancel_event)
            answer = message["content"]
            completed = Submitted({
                "role": "exit", "content": answer,
                "extra": {"exit_status": "Submitted", "submission": answer, "answered": True},
            })
            if self.session.turn_requires_verification():
                result, terminal = self.session.verify_submission(completed)
                if terminal is None:
                    return self.add_messages(self.model.format_message(role="user", content=result["output"]))
                raise terminal
            raise completed
        records = [ToolAction(self.action_label(action)) for action in actions]
        read_only = {"read_file", "recall_output"}
        gated = [record for action, record in zip(actions, records) if action.get("tool") not in read_only]
        outputs = []
        try:
            check_cancelled(self.session.cancel_event)
            approved = not gated or self.session.confirm(gated)
            for action, record in zip(actions, records):
                check_cancelled(self.session.cancel_event)
                if not approved and action.get("tool") not in read_only:
                    outputs.append(self.session.record_output(
                        record, {"output": "", "returncode": -1, "exception_info": "The user declined this command."},
                        state="declined",
                    ))
                    continue
                try:
                    result = self.execute_action(action, record)
                except Submitted as exc:
                    submission = exc.messages[-1]["extra"]["submission"]
                    archived = self.session.record_output(record, {
                        "output": _SUBMIT_MARKER + "\n" + submission, "returncode": 0, "exception_info": "",
                    })
                    check_cancelled(self.session.cancel_event)
                    result, terminal = self.session.verify_submission(exc)
                    outputs.append(archived if result["returncode"] == 0 else result)
                    if terminal:
                        raise terminal
                    break
                outputs.append(self.session.record_output(record, result))
                check_cancelled(self.session.cancel_event)
        finally:
            observations = self.model.format_observation_messages(message, outputs, self.get_template_vars())
            # Text adapters and function-calling adapters both keep the same evidence references.
            for observation, output in zip(observations, outputs):
                observation["extra"] = observation.get("extra", {}) | output.get("extra", {})
            self.add_messages(*observations)
        return observations

    def save(self, path: Path | None, *extra_dicts) -> dict:
        last = self.messages[-1] if self.messages else {}
        result = last.get("extra", {}) if last.get("role") == "exit" else {}
        trajectory = {
            "trajectory_format": "mini-swe-agent-1.1", "messages": self.messages,
            "info": {
                "workspace": str(self.session.root), "model": self.session.model_name,
                "model_stats": {
                    "api_calls": self.session.calls + self.n_calls,
                    "instance_cost": self.session.cost + self.cost,
                },
                **result,
            },
        }
        write_json(self.session.directory / "trajectory.json", trajectory)
        return trajectory


class Session:
    def __init__(
        self, root: Path, config: dict, *, archive: Path | None = None, metadata: dict | None = None,
    ):
        metadata = metadata or {}
        self.id = archive.name if archive is not None else uuid4().hex
        self.root = root
        self.condition = Condition()
        self.cancel_event = Event()
        self.commands = WorkspaceCommands()
        self.turn_id: str | None = None
        self.worker: Thread | None = None
        self.pending: dict | None = None
        self.decision: bool | None = None
        self.resume_notice = ""
        self.directory = archive if archive is not None else _storage_dir(config) / self.id
        if self.directory.is_relative_to(root):
            raise ValueError("Workbench storage must be outside the workspace.")
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=archive is not None)
        self.created_at = metadata.get("created_at", self.directory.stat().st_mtime)
        self.updated_at = metadata.get("updated_at", self.created_at)
        self.events = _read_events(self.directory) if archive is not None else []
        self.last_goal = ""
        self.action_progress: dict[str, dict] = {}
        for event in self.events:
            self._track_action(event)
        self.archived_trajectory = _read_json(self.directory / "trajectory.json", {})
        self.status = metadata.get("status", "ready")
        if self.status in {"running", "stopping", "error"}:
            self.status = "interrupted"
        stats = self.archived_trajectory.get("info", {}).get("model_stats", {})
        self.calls = stats.get("api_calls", 0)
        self.cost = stats.get("instance_cost", 0.0)
        self.evidence = EvidenceStore(self.directory / "evidence")

        settings = config.get("workbench", {}) | metadata.get("settings", {})
        self.config = WorkbenchConfig.model_validate({"storage_dir": _storage_dir(config), **settings})
        self.startup_context = ContextConfig.model_validate(config.get("workbench", {}).get("context", {}))
        self.context_since = metadata.get("context_since", 1)
        model_config = config.get("model", {}).copy()
        model_config.update(metadata.get("model", {}))
        self.model_name = get_model_name(config=model_config)
        self.model_class = metadata.get("model", {}).get("model_class")
        if not self.model_class:
            model_class = get_model_class(self.model_name, model_config.get("model_class", ""))
            self.model_class = f"{model_class.__module__}.{model_class.__name__}"
        self.startup_config = config | {"workbench": settings, "model": model_config}
        configured_checks = (root / ".mini-workbench.yaml").is_file() or settings.get("checks", {}).get("verify")
        self.verification = _read_json(self.directory / "verification.json", {
            "status": "not_run" if configured_checks else "unconfigured",
        })
        self.verification_attempts = 0
        self.turn_version: str | None = None
        baseline = _read_json(self.directory / "baseline.json", {})
        self.baseline = WorkspaceSnapshot(
            {name: tuple(value) for name, value in baseline["files"].items()}, baseline["limited"],
        ) if baseline else None
        self.workspace: Workspace | None = None
        self.agent: WorkbenchAgent | None = None
        if archive is None:
            self._initialize()
            self.save_metadata()

    @classmethod
    def restore(cls, directory: Path, config: dict) -> "Session | None":
        metadata = _read_json(directory / "session.json", {})
        if not metadata:
            info = _read_json(directory / "trajectory.json", {}).get("info", {})
            if not info.get("workspace"):
                return None
            events = _read_events(directory)
            status = "ready" if events and events[-1]["kind"] == "finish" else "interrupted"
            metadata = {"workspace": info["workspace"], "status": status}
            if model := info.get("model"):
                metadata["model"] = {"model_name": model}
        root = Path(metadata["workspace"]).expanduser()
        if not root.is_absolute():
            raise ValueError("Persisted workspace must use an absolute path.")
        return cls(root, config, archive=directory, metadata=metadata)

    @property
    def workspace_available(self) -> bool:
        return self.root.is_dir() and self.root.resolve() == self.root

    @property
    def title(self) -> str:
        task = next((event["text"] for event in self.events if event["kind"] == "user"), "")
        return " ".join(task.split())[:72] or self.root.name or str(self.root)

    @property
    def strategy(self) -> str | None:
        if self.model_class in {f"{LitellmModel.__module__}.{LitellmModel.__name__}",
                                f"{WorkbenchModel.__module__}.{WorkbenchModel.__name__}"}:
            return self.config.context.strategy
        return None

    def listing(self) -> dict:
        return {
            "id": self.id, "title": self.title, "workspace": str(self.root),
            "status": self.status, "updated_at": self.updated_at,
            "needs_approval": self.status == "running" and self.pending is not None and self.decision is None,
        }

    def save_metadata(self, *, context: ContextConfig | None = None, context_since: int | None = None) -> None:
        write_json(self.directory / "session.json", self.listing() | {
            "created_at": self.created_at,
            "context_since": self.context_since if context_since is None else context_since,
            "model": {"model_class": self.model_class, "model_name": self.model_name},
            "settings": {
                "backend": self.config.backend, "docker_image": self.config.docker_image,
                "context": (context or self.config.context).model_dump(),
                "checks": self.config.checks.model_dump(exclude_unset=True),
            },
        })

    def context_settings(self) -> dict | None:
        if self.strategy is None:
            return None
        return {
            "effective": self.config.context.model_dump(),
            "startup": self.startup_context.model_dump(),
            "differs_from_startup": self.config.context != self.startup_context,
        }

    def apply_startup_context(self) -> None:
        with self.condition:
            if self.running:
                raise ValueError("本轮结束后才能应用启动配置。")
            if self.strategy is None:
                raise ValueError("当前模型未启用上下文预算。")
            if self.config.context == self.startup_context:
                return
            budget = self.agent.model.context if self.agent is not None else None
            last_request = next((event["request"] for event in reversed(self.events) if event["kind"] == "context"), 0)
            context_since = max(last_request, budget.requests if budget is not None else 0) + 1
            self.save_metadata(context=self.startup_context, context_since=context_since)
            self.context_since = context_since
            self.config.context = self.startup_context
            if budget is not None:
                budget.reconfigure(self.config.context)

    def _initialize(self) -> None:
        if self.agent is not None:
            return
        if not self.workspace_available:
            raise ValueError(f"工作区目录不存在：{self.root}")
        self.workspace = Workspace(
            self.root, self.config, self.startup_config.get("environment", {}),
            self.startup_config["workbench"].get("checks", {}),
        )
        if not (self.directory / "verification.json").is_file():
            self.verification = {"status": "not_run" if self.workspace.checks.verify else "unconfigured"}
        if self.baseline is None:
            self.baseline = WorkspaceSnapshot.capture(self.root)
            write_json(self.directory / "baseline.json", {
                "files": self.baseline.files, "limited": self.baseline.limited,
            })
        agent = WorkbenchAgent(self, self.startup_config)
        agent.messages = self.archived_trajectory.get("messages", [])
        if isinstance(agent.model, WorkbenchModel):
            agent.model.context.restore(agent.messages, self.events, context_since=self.context_since)
        self.agent = agent
        self.archived_trajectory = {}

    def emit(self, kind: str, **fields) -> None:
        with self.condition:
            event = {"kind": kind, "turn_id": self.turn_id, **fields}
            with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=True) + "\n")
            self.events.append(event)
            self._track_action(event)

    def _track_action(self, event: dict) -> None:
        if event["kind"] == "user":
            self.last_goal = event["text"]
            self.action_progress.clear()
        elif action_id := event.get("action_id"):
            self.action_progress.pop(action_id, None)
            self.action_progress[action_id] = event

    def record_action(self, action: ToolAction, state: str) -> None:
        self.emit("action_state", action_id=action.id, command=action.label, purpose=action.purpose, state=state)

    def source_status(self, id: str) -> dict | None:
        if source := self.evidence.source(id):
            return source | {"status": self.workspace.check_source(source)}
        return None

    def settle_actions(self, *, cancelled: bool = False) -> None:
        for event in list(self.action_progress.values()):
            state = event["state"]
            if state in {"awaiting_approval", "approved", "started"}:
                action = ToolAction(event["command"], event["purpose"], event["action_id"])
                outcome = "unknown" if state == "started" else "cancelled" if cancelled else "skipped"
                self.record_action(action, outcome)

    def resume_details(self) -> dict | None:
        if self.status not in {"interrupted", "cancelled", "error"} or not self.last_goal:
            return None
        actions = list(reversed(self.action_progress.values()))
        attention = [
            action | {"state": "unknown", "previous_state": "started"} for action in actions
            if action["state"] == "started"
        ]
        outstanding = [action for action in actions if action["state"] in {"unknown", "awaiting_approval", "approved"}]
        action = next(iter(attention or outstanding or actions), None)
        if action is not None:
            action = {key: action[key] for key in ("action_id", "command", "purpose", "state", "evidence") if key in action}
        return {"goal": self.last_goal, "action": action}

    def record_output(self, action: ToolAction, output: dict, *, state: str | None = None) -> dict:
        if state is None:
            if self.cancel_event.is_set() and output["returncode"] != 0:
                state = "cancelled"
            else:
                state = "completed" if output["returncode"] == 0 else "failed"
        evidence = self.evidence.save(action.label, output)
        text = output["output"]
        extra = output.get("extra", {}) | {"evidence_id": evidence, "label": action.label, "action_id": action.id}
        self.emit("tool", action_id=action.id, purpose=action.purpose, state=state, command=action.label,
                  text=text[:24_000], truncated=len(text) > 24_000,
                  returncode=output["returncode"], error=output.get("exception_info", ""), evidence=evidence)
        source = f"\nSHA-256: {extra['sha256']}" if extra.get("sha256") else ""
        return output | {"output": f"Evidence {evidence}{source}\n{text}", "extra": extra}

    def confirm(self, actions: list[ToolAction]) -> bool:
        with self.condition:
            check_cancelled(self.cancel_event)
            commands = [action.label for action in actions]
            purpose = actions[0].purpose
            self.pending = {"id": uuid4().hex, "commands": commands, "purpose": purpose}
            for action in actions:
                self.record_action(action, "awaiting_approval")
            self.emit("command_pending", commands=commands, purpose=purpose)
            self.decision = None
            try:
                self.condition.wait_for(lambda: self.decision is not None or self.cancel_event.is_set())
                check_cancelled(self.cancel_event)
                for action in actions:
                    self.record_action(action, "approved" if self.decision else "declined")
                return bool(self.decision)
            finally:
                self.pending = None

    def decide(self, approval_id: str, approved: bool) -> None:
        with self.condition:
            if (self.cancel_event.is_set() or self.pending is None
                    or self.pending["id"] != approval_id or self.decision is not None):
                raise ValueError("这批命令已处理，请刷新会话。")
            self.decision = approved
            self.condition.notify_all()

    @property
    def running(self) -> bool:
        return self.status in {"running", "stopping"}

    def stop(self, turn_id: str) -> None:
        with self.condition:
            if not self.running or self.turn_id != turn_id:
                raise ValueError("这轮请求已结束。")
            if not self.cancel_event.is_set():
                self.cancel_event.set()
                self.status = "stopping"
                self.condition.notify_all()
                self.save_metadata()

    def start(self, task: str) -> None:
        with self.condition:
            if self.running:
                raise ValueError("当前需求仍在运行。")
            if not self.workspace_available:
                raise ValueError(f"工作区目录不可用：{self.root}")
            self._initialize()
            if self.status in {"interrupted", "cancelled"}:
                self.resume_notice = _INTERRUPTED
                details = self.resume_details()
                if details and (action := details["action"]):
                    self.resume_notice += f" Last action state: {action['state']} ({action['purpose']})."
                latest = next((i for i, event in reversed(list(enumerate(self.events))) if event["kind"] == "user"), 0)
                evidence = [event["evidence"] for event in self.events[latest:] if event.get("evidence")]
                if evidence:
                    self.resume_notice += " Recorded output IDs: " + ", ".join(dict.fromkeys(evidence[-8:]))
                recent_request = self.events[latest] if self.events else {}
                last_task = next((
                    m["extra"].get("task_text") for m in reversed(self.agent.messages)
                    if m.get("extra", {}).get("context_category") == "task"
                ), None)
                if recent_request.get("kind") == "user" and recent_request["text"] != last_task:
                    self.resume_notice += " The interrupted user request was: " + recent_request["text"]
            self.verification_attempts = 0
            self.agent.n_calls = 0
            self.agent.cost = 0.0
            self.agent.n_consecutive_format_errors = 0
            self.agent._start_time = time.time()
            self.cancel_event.clear()
            self.turn_id = uuid4().hex
            self.status = "running"
            self.updated_at = time.time()
            self.save_metadata()
            self.emit("user", text=task)
            self.worker = Thread(target=self._run, args=(task,), daemon=True)
            self.worker.start()

    def _run(self, task: str) -> None:
        status = "error"
        try:
            if self.workspace.checks.verify:
                with self.commands.acquire(self.root, self.cancel_event):
                    self.turn_version = self.workspace.version()
            else:
                self.turn_version = None
            result = self.preflight() or self.agent.run_turn(task)
            if result["exit_status"] == "PreflightFailed":
                write_json(self.directory / "preflight.json", result)
            write_json(self.directory / "changes.json", self.changes_payload())
            self.settle_actions(cancelled=result["exit_status"] == "Cancelled")
            self.emit("finish", reason=result.get("exit_status", ""),
                      text="" if result.get("answered") else result.get("submission", ""))
            status = "cancelled" if result.get("exit_status") == "Cancelled" else "ready"
        except TurnCancelled as exc:
            self.agent.add_messages(*exc.messages)
            self.agent.save(None)
            write_json(self.directory / "changes.json", self.changes_payload())
            self.settle_actions(cancelled=True)
            self.emit("finish", reason="Cancelled", text="")
            status = "cancelled"
        except Exception as exc:
            self.settle_actions(cancelled=self.cancel_event.is_set())
            self.emit("error", text=f"{type(exc).__name__}: {str(exc)[:1000]}")
        finally:
            with self.condition:
                self.calls += self.agent.n_calls
                self.cost += self.agent.cost
                self.status = status
                self.updated_at = time.time()
                self.save_metadata()

    def preflight(self) -> dict | None:
        argv = self.workspace.checks.dependency_check()
        if not argv:
            return None
        label = shlex.join(argv)
        action = ToolAction(label, "preflight")
        if not self.confirm([action]):
            return {"exit_status": "PreflightFailed", "submission": "已拒绝环境预检。"}
        with self.commands.acquire(self.root, self.cancel_event):
            self.record_action(action, "started")
            result = self.workspace.check(argv, self.cancel_event)
        self.record_output(action, result)
        check_cancelled(self.cancel_event)
        if result["returncode"] != 0:
            return {"exit_status": "PreflightFailed", "submission": "环境预检失败，请查看命令输出后再试。"}
        return None

    def turn_requires_verification(self) -> bool:
        if self.turn_version is None:
            return False
        with self.commands.acquire(self.root, self.cancel_event):
            return self.workspace.version() != self.turn_version

    def verify_submission(self, submitted: Submitted) -> tuple[dict, Submitted | None]:
        argv = self.workspace.checks.verify
        submission = submitted.messages[-1]["extra"]["submission"]
        if not argv:
            return {"output": _SUBMIT_MARKER + "\n" + submission, "returncode": 0, "exception_info": ""}, submitted
        label = shlex.join(argv)
        action = ToolAction(label, "verification")
        self.verification_attempts += 1
        if self.confirm([action]):
            with self.commands.acquire(self.root, self.cancel_event):
                before = self.workspace.version()
                self.record_action(action, "started")
                result = self.workspace.check(argv, self.cancel_event)
                version = self.workspace.version()
            status = "stale" if before != version else "passed" if result["returncode"] == 0 else "failed"
        else:
            result = {"output": "", "returncode": -1, "exception_info": "Verification command was declined."}
            version, status = None, "declined"
        archived = self.record_output(action, result, state="declined" if status == "declined" else None)
        check_cancelled(self.cancel_event)
        self.verification = {
            "status": status, "version": version, "evidence": archived["extra"]["evidence_id"],
            "returncode": result["returncode"], "command": label, "attempt": self.verification_attempts,
        }
        self.emit("verification", **self.verification)
        write_json(self.directory / "verification.json", self.verification)
        if status == "passed":
            return {"output": _SUBMIT_MARKER + "\n" + submission, "returncode": 0, "exception_info": ""}, submitted
        terminal = None
        if status == "declined" or self.verification_attempts > self.workspace.checks.verification_retries:
            terminal = Submitted({
                "role": "exit", "content": "Verification failed.",
                "extra": {"exit_status": "VerificationFailed", "submission": ""},
            })
        return ({
            "output": f"Submission blocked: verification {status}.\n{archived['output']}",
            "returncode": 1, "exception_info": "Review the verification output before attempting submission again.",
        }, terminal)

    def current_verification(self) -> dict:
        result = self.verification
        if result["status"] == "passed":
            try:
                current = Workspace.version_of(self.root) if self.workspace_available else None
            except OSError:
                current = None
            if current != result["version"]:
                return result | {"status": "stale"}
        return result

    def changes_payload(self) -> dict:
        if self.baseline is None or not self.workspace_available:
            recorded = _read_json(self.directory / "changes.json", {"files": [], "limited": False})
            return recorded | {"verification": self.current_verification()}
        current = WorkspaceSnapshot.capture(self.root)
        changes = self.baseline.changes(current)
        return {
            "files": changes, "limited": current.limited or self.baseline.limited or len(changes) == 100,
            "verification": self.current_verification(),
        }

    def view(self, offset: int = 0) -> dict:
        with self.condition:
            return self.listing() | {
                "model": self.model_name,
                "workspace_available": self.workspace_available,
                "backend": self.config.backend,
                "context_strategy": self.strategy,
                "context_settings": self.context_settings(),
                "resume": self.resume_details(),
                "verification": self.verification, "archive": str(self.directory),
                "pending": self.pending if self.status == "running" and self.decision is None else None,
                "turn_id": self.turn_id,
                "calls": self.calls + (self.agent.n_calls if self.running else 0),
                "cost": self.cost + (self.agent.cost if self.running else 0),
                "events": [render_event(event) for event in self.events[offset : offset + 100]],
                "cursor": min(offset + 100, len(self.events)),
                "more": offset + 100 < len(self.events),
            }


class WorkbenchServer(ThreadingHTTPServer):
    def __init__(self, port: int, workspace: Path, config: dict):
        super().__init__(("127.0.0.1", port), WorkbenchHandler)
        self.workspace = workspace
        self.config = config
        self.sessions: dict[str, Session] = {}
        self.lock = Lock()
        self.commands = WorkspaceCommands()
        for directory in _storage_dir(config).glob("*"):
            if directory.name.startswith(".") or not directory.is_dir() or directory.is_symlink():
                continue
            try:
                session = Session.restore(directory, config)
            except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
                _LOGGER.warning("Could not load workbench session %s: %s", directory.name, exc)
                continue
            if session is not None:
                self.register(session)

    def register(self, session: Session) -> None:
        session.commands = self.commands
        self.sessions[session.id] = session

    def start_turn(self, session: Session, task: str) -> None:
        with self.lock:
            if self.sessions.get(session.id) is not session:
                raise ValueError("找不到此会话。")
            session.start(task)

    def delete_session(self, session: Session) -> None:
        with self.lock, session.condition:
            if self.sessions.get(session.id) is not session:
                raise ValueError("找不到此会话。")
            if session.running:
                raise ValueError("请先停止这条会话正在运行的请求。")
            if session.directory.parent != _storage_dir(self.config) or session.directory.is_symlink():
                raise ValueError("这个目录不在工作台会话存储中。")
            removed = session.directory.with_name(f".deleted-{session.id}-{uuid4().hex}")
            session.directory.rename(removed)
            del self.sessions[session.id]
        if session.config.backend == "docker" and session.workspace is not None:
            session.workspace.env.cleanup()
        shutil.rmtree(removed)


class WorkbenchHandler(BaseHTTPRequestHandler):
    server: WorkbenchServer

    def log_message(self, format: str, *args) -> None:
        pass

    @property
    def host(self) -> str:
        return f"127.0.0.1:{self.server.server_port}"

    def respond(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
        )
        self.end_headers()
        self.wfile.write(data)

    def respond_json(self, status: int, data: dict) -> None:
        self.respond(status, json.dumps(data).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if self.headers.get("Host") != self.host:
            self.respond_json(403, {"error": "请使用终端显示的本地地址。"})
            return
        url = urlsplit(self.path)
        if asset := {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}.get(url.path):
            content_type = mimetypes.guess_type(asset)[0] or "text/plain"
            self.respond(200, (_ASSETS / asset).read_bytes(), f"{content_type}; charset=utf-8")
            return
        if url.path == "/api/sessions":
            with self.server.lock:
                sessions = sorted((session.listing() for session in self.server.sessions.values()),
                                  key=lambda session: session["updated_at"])
            self.respond_json(200, {"workspace": str(self.server.workspace), "sessions": sessions})
            return
        parts = url.path.removeprefix("/api/sessions/").split("/") if url.path.startswith("/api/sessions/") else [""]
        with self.server.lock:
            session = self.server.sessions.get(parts[0])
        if session is None:
            self.respond_json(404, {"error": "找不到此会话。"})
        elif len(parts) == 1:
            raw_offset = parse_qs(url.query).get("since", ["0"])[0]
            offset = int(raw_offset) if raw_offset.isascii() and raw_offset.isdecimal() and len(raw_offset) <= 10 else 0
            self.respond_json(200, session.view(offset))
        elif parts[1:] == ["changes"]:
            self.respond_json(200, session.changes_payload())
        elif len(parts) == 3 and parts[1] == "evidence":
            query = parse_qs(url.query)
            try:
                start = int(query.get("start", ["1"])[0])
                end = int(query["end"][0]) if "end" in query else None
                result = session.evidence.recall(parts[2], start, end)
            except (ValueError, OSError) as exc:
                self.respond_json(400, {"error": str(exc)})
            else:
                self.respond_json(200, result)
        else:
            self.respond_json(404, {"error": "找不到此接口。"})

    def do_POST(self) -> None:
        if self.headers.get("Host") != self.host or self.headers.get("Origin") != f"http://{self.host}":
            self.respond_json(403, {"error": "只接受工作台页面发出的请求。"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 128_000 or self.headers.get_content_type() != "application/json":
                raise ValueError("请求必须是有效的 JSON。")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("请求必须是 JSON 对象。")
            result = self.handle_post(urlsplit(self.path).path, data)
        except (ValueError, OSError) as exc:
            self.respond_json(400, {"error": str(exc)})
            return
        self.respond_json(200, result)

    def handle_post(self, path: str, data: dict) -> dict:
        if path == "/api/sessions":
            workspace = data.get("workspace")
            if not isinstance(workspace, str) or not workspace.strip():
                raise ValueError("请输入工作区目录。")
            root = Path(workspace.strip()).expanduser().resolve()
            if not root.is_dir():
                raise ValueError(f"目录不存在：{root}")
            session = Session(root, self.server.config)
            with self.server.lock:
                self.server.register(session)
            return session.view()
        parts = path.split("/")
        if len(parts) != 5 or parts[1:3] != ["api", "sessions"]:
            raise ValueError("找不到此接口。")
        with self.server.lock:
            session = self.server.sessions.get(parts[3])
        if session is None:
            raise ValueError("找不到此会话。")
        if parts[4] == "messages":
            task = data.get("message")
            if not isinstance(task, str) or not task.strip():
                raise ValueError("请输入需求。")
            self.server.start_turn(session, task.strip())
        elif parts[4] == "stop":
            turn_id = data.get("turn_id")
            if not isinstance(turn_id, str):
                raise ValueError("缺少本轮请求编号。")
            session.stop(turn_id)
        elif parts[4] == "delete":
            self.server.delete_session(session)
        elif parts[4] == "context":
            with self.server.lock:
                if self.server.sessions.get(session.id) is not session:
                    raise ValueError("找不到此会话。")
                session.apply_startup_context()
                return {"ok": True, "context_settings": session.context_settings()}
        elif parts[4] == "approval":
            approval_id, approved = data.get("id"), data.get("approved")
            if not isinstance(approval_id, str) or not isinstance(approved, bool):
                raise ValueError("请选择批准或拒绝。")
            session.decide(approval_id, approved)
        else:
            raise ValueError("找不到此接口。")
        return {"ok": True}


@app.command()
def main(
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w", exists=True, file_okay=False, resolve_path=True),
    config_spec: list[str] = typer.Option([], "--config", "-c", help="YAML files or key=value overrides."),
    port: int = typer.Option(8765, "--port", min=0, max=65535),
) -> None:
    """Start a local web interface for mini-SWE-agent."""
    specs = [builtin_config_dir / "mini.yaml", "agent.step_limit=30", *config_spec]
    config = recursive_merge(*(get_config_from_spec(spec) for spec in specs))
    get_model_name(config=config.get("model", {}))
    context = ContextConfig.model_validate(config.get("workbench", {}).get("context", {}))
    with WorkbenchServer(port, workspace, config) as server:
        typer.echo(f"Workbench: http://127.0.0.1:{server.server_port}")
        thresholds = " / ".join(
            f"{ratio * 100:g}%" for ratio in (context.soft_threshold, context.hard_threshold, context.summary_threshold)
        )
        typer.echo(f"New session context: {context.strategy}, {context.input_limit:,} input tokens, {thresholds}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            with server.lock:
                sessions = list(server.sessions.values())
            for session in sessions:
                with session.condition:
                    if session.running and session.turn_id is not None:
                        session.stop(session.turn_id)
            for session in sessions:
                if session.worker is not None:
                    session.worker.join(timeout=2)
                if session.config.backend == "docker" and session.workspace is not None:
                    session.workspace.env.cleanup()


if __name__ == "__main__":
    app()
