import asyncio
import time
from pathlib import Path
from threading import Event, Thread

import pytest
from litellm import ModelResponse

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models.test_models import make_toolcall_output
from minisweagent.run.utilities.workbench import Session, ToolAction, render_event
from minisweagent.run.utilities.workbench_changes import WorkspaceSnapshot
from minisweagent.run.utilities.workbench_lifecycle import TurnCancelled, WorkspaceCommands, wait_for_model
from minisweagent.run.utilities.workbench_workspace import WorkbenchConfig, Workspace


def _action(command: str, call_id: str) -> dict:
    return make_toolcall_output(None, [], [{"command": command, "tool_call_id": call_id}])


def _wait_for(session: Session, field: str, value=None) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        view = session.view()
        if (value is None and view[field]) or (value is not None and view[field] == value):
            return view
        time.sleep(0.01)
    pytest.fail(f"Expected {field} to be {value!r}, got {session.view()!r}")


def test_workbench_confirms_commands_and_continues_same_session(tmp_path: Path) -> None:
    config = get_config_from_spec(builtin_config_dir / "mini.yaml")
    config["workbench"] = {"storage_dir": tmp_path / "runs"}
    workspace = tmp_path / "project"
    workspace.mkdir()
    submit = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    config["model"].update(
        model_class="minisweagent.models.test_models.DeterministicToolcallModel",
        model_name="test",
        outputs=[
            _action("echo original > orders.py", "write"),
            _action(submit, "submit-first"),
            _action("echo changed > orders.py", "rejected"),
            _action(submit, "submit-second"),
        ],
    )
    session = Session(workspace, config)

    session.start("Create the order file.")
    pending = _wait_for(session, "pending")["pending"]
    assert pending["commands"] == ["echo original > orders.py"]
    assert not (workspace / "orders.py").exists()
    session.decide(pending["id"], True)
    session.decide(_wait_for(session, "pending")["pending"]["id"], True)
    assert _wait_for(session, "status", "ready")["calls"] == 2

    config["model"]["outputs"] = config["model"]["outputs"][2:]
    session = Session.restore(session.directory, config)
    assert session.agent is None
    assert session.view()["calls"] == 2
    assert session.evidence.recall("E000001")["output"] == ""
    assert session.changes_payload()["files"][0]["path"] == "orders.py"

    session.start("Keep working on the order file.")
    pending = _wait_for(session, "pending")["pending"]
    with pytest.raises(ValueError, match="已处理"):
        session.decide("stale-id", True)
    session.decide(pending["id"], False)
    session.decide(_wait_for(session, "pending")["pending"]["id"], True)
    assert _wait_for(session, "status", "ready")["calls"] == 4
    assert (workspace / "orders.py").read_text() == "original\n"
    assert [m["tool_call_id"] for m in session.agent.messages if m.get("role") == "tool"] == [
        "write",
        "submit-first",
        "rejected",
        "submit-second",
    ]
    assert [m["role"] for m in session.agent.messages].count("exit") == 1
    tasks = [m["extra"] for m in session.agent.messages if m.get("extra", {}).get("context_category") == "task"]
    assert [t["task_text"] for t in tasks] == ["Create the order file.", "Keep working on the order file."]
    assert all("completion" in t for t in tasks)
    assert session.evidence.count == 4
    output = next(event for event in session.view()["events"] if event.get("evidence") == "E000001")
    assert [event["state"] for event in session.events if event.get("action_id") == output["action_id"]] == [
        "awaiting_approval", "approved", "started", "completed",
    ]
    assert [event["text"] for event in session.view()["events"] if event["kind"] == "user"] == [
        "Create the order file.",
        "Keep working on the order file.",
    ]

    assert session.baseline.changes(WorkspaceSnapshot.capture(workspace)) == [
        {
            "path": "orders.py",
            "status": "added",
            "diff": "--- /dev/null\n+++ b/orders.py\n@@ -0,0 +1 @@\n+original",
            "truncated": False,
        }
    ]


def test_restores_empty_and_interrupted_sessions_without_replaying_commands(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = get_config_from_spec(builtin_config_dir / "mini.yaml")
    config["workbench"] = {"storage_dir": tmp_path / "runs"}
    config["model"].update(
        model_class="minisweagent.models.test_models.DeterministicToolcallModel",
        model_name="test",
        outputs=[_action("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "finish")],
    )
    original = Session(root, config)
    assert Session.restore(original.directory, config).view()["status"] == "ready"

    original.status = "running"
    original.save_metadata()
    original.emit("user", text="Create the first file.")
    uncertain = ToolAction("touch never-replay")
    original.record_action(uncertain, "awaiting_approval")
    original.record_action(uncertain, "approved")
    original.record_action(uncertain, "started")
    original.emit("command_pending", commands=["touch never-replay"], purpose="command")
    with (original.directory / "events.jsonl").open("a") as stream:
        stream.write('{"kind":')

    restored = Session.restore(original.directory, config)
    assert restored.view()["status"] == "interrupted"
    assert restored.view()["pending"] is None
    assert restored.view()["resume"] == {
        "goal": "Create the first file.",
        "action": {"action_id": uncertain.id, "command": uncertain.label, "purpose": "command", "state": "unknown"},
    }
    restored.start("Check the project and finish.")
    approval = _wait_for(restored, "pending")["pending"]
    assert approval["commands"] == ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]
    restored.decide(approval["id"], True)
    assert _wait_for(restored, "status", "ready")["calls"] == 1
    assert not (root / "never-replay").exists()
    assert any("The interrupted user request was: Create the first file." in (m.get("content") or "")
               for m in restored.agent.messages)
    assert [event["kind"] for event in Session.restore(original.directory, config).events].count("finish") == 1


def test_preflight_and_verification_use_direct_exit_codes(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "check.py").write_text("raise SystemExit(4)\n")
    config = get_config_from_spec(builtin_config_dir / "mini.yaml")
    config["workbench"] = {
        "storage_dir": tmp_path / "runs",
        "checks": {"preflight": ["missing-workbench-dependency"], "verify": ["python3", "check.py"]},
    }
    submit = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    config["model"].update(
        model_class="minisweagent.models.test_models.DeterministicToolcallModel",
        model_name="test",
        outputs=[
            _action(submit, "submit-fails"),
            _action("printf 'print(123)\\n' > check.py", "repair"),
            _action(submit, "submit-passes"),
        ],
    )
    session = Session(root, config)
    assert session.workspace.env.execute({"command": "python3 check.py | tail -n 1"})["returncode"] == 0
    session.start("Verify the workspace.")
    approval = _wait_for(session, "pending")["pending"]
    assert approval["purpose"] == "preflight"
    session.decide(approval["id"], True)
    view = _wait_for(session, "status", "ready")
    assert view["calls"] == 0
    assert view["events"][-1]["reason"] == "PreflightFailed"

    session.workspace.checks.preflight = []
    session.start("Repair the verifier and submit.")
    for purpose in ("preflight", "command", "verification", "command", "command", "verification"):
        approval = _wait_for(session, "pending")["pending"]
        assert approval["purpose"] == purpose
        session.decide(approval["id"], True)
    view = _wait_for(session, "status", "ready")
    assert view["events"][-1]["reason"] == "Submitted"
    assert [(event["status"], event["returncode"]) for event in view["events"] if event["kind"] == "verification"] == [
        ("failed", 4), ("passed", 0),
    ]
    assert session.current_verification()["status"] == "passed"
    (root / "check.py").write_text("raise SystemExit(5)\n")
    assert session.current_verification()["status"] == "stale"


def test_structured_patch_checks_source_version_and_workspace(tmp_path: Path) -> None:

    root = tmp_path / "project"
    root.mkdir()
    (root / "config.py").write_text("DEFAULT = 20\n")
    workspace = Workspace(root, WorkbenchConfig(storage_dir=tmp_path / "runs"), {}, {})
    source = workspace.read_file("config.py")
    assert source["output"] == "DEFAULT = 20"
    workspace.apply_patch("config.py", "50", "20", source["extra"]["sha256"])
    with pytest.raises(ValueError, match="source changed"):
        workspace.apply_patch("config.py", "30", "50", source["extra"]["sha256"])
    with pytest.raises(ValueError, match="relative"):
        workspace.read_file("../config.py")
    assert (root / "config.py").read_text() == "DEFAULT = 50\n"


def test_plain_answer_finishes_workbench_but_truncated_reply_does_not(tmp_path: Path) -> None:
    config = get_config_from_spec(builtin_config_dir / "mini.yaml")
    config["workbench"] = {"storage_dir": tmp_path / "runs"}
    config["model"]["model_name"] = "openai/test"
    root = tmp_path / "project"
    root.mkdir()
    session = Session(root, config)
    text = "The project has an order module."
    message = {"role": "assistant", "content": text}
    stopped = ModelResponse(choices=[{"message": message, "finish_reason": "stop"}])
    assert session.agent.model._parse_actions(stopped) == []
    with pytest.raises(Submitted) as result:
        session.agent.execute_actions(message | {"extra": {"actions": []}})
    assert result.value.messages[0]["extra"] == {
        "exit_status": "Submitted", "submission": text, "answered": True,
    }

    for reason, content in (("length", text), ("stop", " ")):
        response = ModelResponse(choices=[{"message": message | {"content": content}, "finish_reason": reason}])
        with pytest.raises(FormatError):
            session.agent.model._parse_actions(response)


def test_assistant_markdown_renders_code_and_does_not_execute_raw_html() -> None:
    text = "## 与对账的连接\n\n调用 `reports.totals_by_customer`。\n\n<script>danger()</script>\n\n[x](javascript:alert(1))"
    event = render_event({"kind": "assistant", "text": text})
    assert "<h2>与对账的连接</h2>" in event["html"]
    assert "<code>reports.totals_by_customer</code>" in event["html"]
    assert "&lt;script&gt;" in event["html"] and "<script>" not in event["html"]
    assert 'href="javascript:' not in event["html"]
    assert event["text"] == text


def test_stopping_pending_turn_does_not_stop_another_session(tmp_path: Path) -> None:
    config = get_config_from_spec(builtin_config_dir / "mini.yaml")
    config["workbench"] = {"storage_dir": tmp_path / "runs"}
    config["model"].update(
        model_class="minisweagent.models.test_models.DeterministicToolcallModel",
        model_name="test",
        outputs=[
            _action("touch marker", "write"),
            _action("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "finish"),
        ],
    )
    root = tmp_path / "project"
    root.mkdir()
    first, second = Session(root, config), Session(root, config)
    first.commands = second.commands = WorkspaceCommands()
    first.start("First request.")
    second.start("Second request.")
    pending_first = _wait_for(first, "pending")["pending"]
    pending_second = _wait_for(second, "pending")["pending"]

    with pytest.raises(ValueError, match="已结束"):
        first.stop("stale-turn")
    first.stop(first.turn_id)
    assert _wait_for(first, "status", "cancelled")["pending"] is None
    with pytest.raises(ValueError, match="已处理"):
        first.decide(pending_first["id"], True)
    assert not (root / "marker").exists()

    second.decide(pending_second["id"], True)
    second.decide(_wait_for(second, "pending")["pending"]["id"], True)
    assert _wait_for(second, "status", "ready")["events"][-1]["reason"] == "Submitted"


def test_stop_terminates_local_command_and_preserves_partial_output(tmp_path: Path) -> None:
    config = get_config_from_spec(builtin_config_dir / "mini.yaml")
    config["workbench"] = {"storage_dir": tmp_path / "runs"}
    config["model"].update(
        model_class="minisweagent.models.test_models.DeterministicToolcallModel",
        model_name="test",
        outputs=[_action("printf begun; touch started; sleep 20; touch should-not-run", "long")],
    )
    root = tmp_path / "project"
    root.mkdir()
    session = Session(root, config)
    session.start("Run the long command.")
    session.decide(_wait_for(session, "pending")["pending"]["id"], True)
    deadline = time.monotonic() + 5
    while not (root / "started").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert (root / "started").exists()

    session.stop(session.turn_id)
    final = _wait_for(session, "status", "cancelled")
    tools = [event for event in final["events"] if event["kind"] == "tool"]
    assert len(tools) == 1
    assert tools[0]["returncode"] != 0 and tools[0]["text"] == "begun"
    assert "cancelled" in tools[0]["error"]
    assert not (root / "should-not-run").exists()
    assert final["calls"] == 1 and final["events"][-1]["reason"] == "Cancelled"


def test_stop_cancels_inflight_model_coroutine() -> None:
    cancel, started, closed = Event(), Event(), Event()
    outcome = []

    async def provider() -> None:
        started.set()
        try:
            await asyncio.sleep(20)
        finally:
            closed.set()

    def run() -> None:
        with pytest.raises(TurnCancelled):
            wait_for_model(provider, cancel)
        outcome.append("cancelled")

    worker = Thread(target=run, daemon=True)
    worker.start()
    assert started.wait(3)
    cancel.set()
    worker.join(timeout=3)
    assert not worker.is_alive() and outcome == ["cancelled"]
    assert closed.wait(3)


def test_workspace_commands_queue_overlapping_paths_only(tmp_path: Path) -> None:
    commands = WorkspaceCommands()
    cancel = Event()
    root = tmp_path / "project"
    nested_waiting, nested_cancelled, other_ran = Event(), Event(), Event()

    def nested() -> None:
        nested_waiting.set()
        with pytest.raises(TurnCancelled):
            with commands.acquire(root / "src", cancel):
                pytest.fail("An overlapping command ran before the active command released its lock.")
        nested_cancelled.set()

    def unrelated() -> None:
        with commands.acquire(tmp_path / "other-project", Event()):
            other_ran.set()

    with commands.acquire(root, Event()):
        Thread(target=nested, daemon=True).start()
        assert nested_waiting.wait(3)
        Thread(target=unrelated, daemon=True).start()
        assert other_ran.wait(3)
        cancel.set()
        assert nested_cancelled.wait(3)


def test_applies_startup_context_to_restored_session_and_reconsiders_history(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = get_config_from_spec(builtin_config_dir / "mini.yaml")
    config["model"]["model_name"] = "openai/test"
    config["workbench"] = {"storage_dir": tmp_path / "runs"}
    low = {
        "context_window": 16_384, "soft_threshold": 0.15, "hard_threshold": 0.25,
        "summary_threshold": 0.35, "recent_groups": 1,
    }
    session = Session(root, config | {"workbench": config["workbench"] | {"context": low}})
    history = [
        {"role": "system", "content": "Work in the project."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "old"}]},
        {"role": "tool", "tool_call_id": "old", "content": "a" * 7_800, "extra": {"evidence_id": "E000001"}},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "new"}]},
        {"role": "tool", "tool_call_id": "new", "content": "b" * 11_000},
    ]
    session.agent.messages = history
    session.agent.save(None)
    session.agent.model.context.prepare(history, [], list)
    assert session.agent.model.context.removed == {1, 2}

    restored = Session.restore(session.directory, config)
    settings = restored.view()["context_settings"]
    assert settings["effective"]["soft_threshold"] == 0.15
    assert settings["startup"]["soft_threshold"] == 0.50
    assert settings["differs_from_startup"]
    restored.apply_startup_context()
    assert not restored.view()["context_settings"]["differs_from_startup"]

    reopened = Session.restore(session.directory, config)
    reopened._initialize()
    budget = reopened.agent.model.context
    assert budget.requests == 1
    assert budget.removed == set()
    assert budget.prepare(history, [], list) == history
    assert budget.requests == 2
    assert reopened.view()["context_settings"]["effective"]["summary_threshold"] == 0.85
    assert budget.input_limit == 61_440


def test_resumed_model_validates_archived_read_file_evidence(tmp_path: Path) -> None:
    config = get_config_from_spec(builtin_config_dir / "mini.yaml")
    config["model"]["model_name"] = "openai/test"
    config["workbench"] = {"storage_dir": tmp_path / "runs"}
    root = tmp_path / "project"
    root.mkdir()
    path = root / "config.py"
    path.write_text("PAGE_SIZE = 20\n")
    session = Session(root, config)
    action = ToolAction("read_file config.py")
    output = session.agent.execute_action(
        {"tool": "read_file", "arguments": {"path": "config.py"}}, action,
    )
    archived = session.record_output(action, output)
    evidence = archived["extra"]["evidence_id"]
    session.agent.messages = [
        {"role": "user", "content": "Check old configuration.", "extra": {"completion": {}, "context_category": "task"}},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "read"}]},
        {"role": "tool", "tool_call_id": "read", "content": archived["output"], "extra": archived["extra"]},
    ]
    session.agent.save(None)
    path.write_text("PAGE_SIZE = 50\n")
    restored = Session.restore(session.directory, config)
    restored._initialize()
    request = restored.agent.model._prepare_messages_for_api(restored.agent.messages)
    assert "Use read_file for the current content" in request[-1]["content"]
    assert request[-1]["tool_call_id"] == "read"
    assert restored.events[-1]["source_checks"][0]["status"] == "changed"
    historical = restored.agent.execute_action(
        {"tool": "recall_output", "arguments": {"id": evidence}}, ToolAction("recall_output"),
    )["output"]
    assert "Historical snapshot" in historical and "PAGE_SIZE = 20" in historical
