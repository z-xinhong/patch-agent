import json
import subprocess
import sys
from pathlib import Path
import pytest

from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from minisweagent.run.utilities.workbench_context import Constraint, ContextBudget, ContextConfig
from minisweagent.run.utilities.workbench_workspace import EvidenceStore, WorkbenchConfig, Workspace


def test_strategies_keep_tool_pairs_and_recover_late_legacy_evidence(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[2] / "examples" / "workbench_compat" / "project"
    raw_log = subprocess.check_output([sys.executable, str(fixture / "diagnostics.py")], text=True)
    contract = "LEGACY EXPORT CONTRACT: ID column is order_ref and a missing order ID is rendered as --."
    preview = raw_log[:5000] + raw_log[-5000:]
    assert contract in raw_log and contract not in preview
    archive = EvidenceStore(tmp_path / "archive")
    evidence_id = archive.save("python3 diagnostics.py", {"output": raw_log, "returncode": 0})

    history = [
        {"role": "system", "content": "Keep changes in the workspace."},
        {"role": "system", "content": "Keep legacy compatibility.", "extra": {"context_category": "project_rules"}},
        {"role": "user", "content": "Add the export.", "extra": {"context_category": "task"}},
    ]
    for index, content in enumerate((preview, "Read orders.py", "Ready to implement")):
        call = f"call_{index}"
        history.extend([
            {"role": "assistant", "content": None, "tool_calls": [{"id": call}]},
            {"role": "tool", "tool_call_id": call, "content": content},
        ])
    history[4]["extra"] = {"evidence_id": evidence_id, "label": "diagnostics"}
    requests = {}
    for strategy in ("original", "window", "evidence"):
        directory = tmp_path / strategy
        directory.mkdir()
        events = []

        def emit(kind: str, **data) -> None:
            events.append({"kind": kind, **data})

        budget = ContextBudget(
            ContextConfig(strategy=strategy, context_window=1750, output_reserve=256, recent_groups=1,
                          threshold=0.6, preview_chars=320),
            directory,
            emit,
        )
        wire = budget.prepare(history, [BASH_TOOL], lambda messages: [
            {key: value for key, value in message.items() if key != "extra"} for message in messages
        ])
        assert wire[:3] == [{k: v for k, v in message.items() if k != "extra"} for message in history[:3]]
        assert [m["tool_call_id"] for m in wire if m["role"] == "tool"] == [
            call["id"] for m in wire if m["role"] == "assistant" for call in m["tool_calls"]
        ]
        request = json.loads((directory / "request-0001.json").read_text())
        assert request["messages"] == wire
        requests[strategy] = request
        budget.record_usage({"extra": {"response": {"usage": {
            "prompt_tokens": 1200, "prompt_cache_hit_tokens": 1000,
        }}, "cost": 0.02}})
        assert events[-1]["uncached"] == 200

    assert requests["original"]["estimated"] > requests["window"]["estimated"]
    assert 4 in requests["window"]["removed"]
    assert requests["evidence"]["estimated"] < requests["original"]["estimated"]
    assert any(evidence_id in (m.get("content") or "") for m in requests["evidence"]["messages"])
    assert archive.recall(evidence_id, 161, 161)["output"] == contract


def _wire(messages: list[dict]) -> list[dict]:
    return [{key: value for key, value in message.items() if key != "extra"} for message in messages]


def test_compaction_shortens_at_soft_threshold_and_removes_pairs_at_hard_threshold(tmp_path: Path) -> None:
    events = []
    budget = ContextBudget(
        ContextConfig(context_window=2_000, output_reserve=256, recent_groups=1, preview_chars=380),
        tmp_path,
        lambda kind, **data: events.append(data),
    )
    history = [
        {"role": "user", "content": "Current task.", "extra": {"context_category": "task"}},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "old"}]},
        {"role": "tool", "tool_call_id": "old", "content": "x" * 3_030,
         "extra": {"evidence_id": "E000001", "label": "test"}},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "new"}]},
        {"role": "tool", "tool_call_id": "new", "content": "Passing"},
    ]
    wire = budget.prepare(history, [], _wire)
    report = events[-1]
    assert report["thresholds"][0] <= report["before"] < report["thresholds"][1]
    assert report["shortened"] == [2] and report["removed"] == []
    assert wire[2]["tool_call_id"] == "old" and "E000001" in wire[2]["content"]

    history[-1]["content"] = "y" * 3_880
    wire = budget.prepare(history, [], _wire)
    assert events[-1]["removed"] == [1, 2]
    assert wire[-1]["content"] == "Archived observations (recall_output): E000001 (test)"
    assert [message["tool_call_id"] for message in wire if message["role"] == "tool"] == ["new"]


def test_summarizes_only_completed_tasks_and_preserves_quotes_and_evidence(tmp_path: Path) -> None:
    event_log = []

    def emit(kind: str, **data) -> None:
        event_log.append({"kind": kind, **data})

    budget = ContextBudget(ContextConfig(context_window=4_000, output_reserve=400, recent_groups=1), tmp_path, emit)
    constraint = "Keep the old order API compatible."
    original = "Add CSV with these notes: " + "reference " * 780 + constraint
    history = [
        {"role": "system", "content": "Workflow instructions."},
        {"role": "user", "content": original, "extra": {
            "context_category": "task",
            "completion": {"end": 4, "submission": "CSV added.", "verification": "E000009"},
        }},
        {"role": "assistant", "content": "I changed the export."},
        {"role": "tool", "content": "Passed", "extra": {"evidence_id": "E000009", "label": "pytest"}},
        {"role": "user", "content": "Still unfinished", "extra": {"context_category": "task"}},
        {"role": "assistant", "content": "Could not finish."},
        {"role": "user", "content": "Work on this next.", "extra": {"context_category": "task"}},
        {"role": "assistant", "content": "Starting.", "tool_calls": [{"id": "current"}]},
        {"role": "tool", "content": "diagnostic " * 310, "tool_call_id": "current"},
    ]

    def invalid_summary(messages: list[dict], reserve: int) -> str:
        return json.dumps({"progress": "CSV added, verified E900000.", "tasks": [1], "constraints": []})

    initial = budget.prepare(history, [], _wire, invalid_summary)
    assert event_log[0]["kind"] == "summary" and event_log[0]["status"] == "failed"
    assert initial[1]["content"] == original and not (tmp_path / "task-summary.json").exists()

    def summarize(messages: list[dict], reserve: int) -> str:
        payload = json.loads(messages[-1]["content"])
        assert [(item["task"], item["request"]) for item in payload["completed"]] == [(1, original)]
        assert payload["completed"][0]["verification_evidence"] == "E000009"
        assert 0 < reserve <= 400
        return json.dumps({
            "progress": "CSV added and verified (E000009).", "tasks": [1],
            "constraints": [{"task": 1, "quote": constraint}],
        })

    wire = budget.prepare(history, [], _wire, summarize)
    assert event_log[-1]["summarized"] == [1]
    assert [message["content"] for message in wire if message["role"] == "user"] == [
        f"Completed task history:\nCSV added and verified (E000009).\n"
        f"User constraints (newer explicit instructions take precedence):\n- Task 1: {constraint}",
        "Still unfinished",
        "Work on this next.",
    ]
    assert history[1]["content"] == original
    assert json.loads((tmp_path / "task-summary.json").read_text())["tasks"] == [1]

    restored = ContextBudget(budget.config, tmp_path, emit)
    restored.restore(history, event_log.copy())
    assert restored.prepare(history, [], _wire) == wire
    assert restored.requests == 3 and restored.summary_requests == 2


def test_skips_summary_without_a_model_call_for_short_completed_requests(tmp_path: Path) -> None:
    events = []
    config = ContextConfig(context_window=1_600, output_reserve=256, recent_groups=1)
    budget = ContextBudget(config, tmp_path, lambda kind, **details: events.append((kind, details)))
    history = [
        {"role": "user", "content": "简单介绍项目", "extra": {
            "context_category": "task", "completion": {"end": 1, "submission": "已经介绍项目"},
        }},
        {"role": "user", "content": "再解释一下业务", "extra": {
            "context_category": "task", "completion": {"end": 2, "submission": "已经解释业务"},
        }},
        {"role": "user", "content": "Explain these details. " * 185, "extra": {"context_category": "task"}},
    ]

    def summarize(messages: list[dict], reserve: int) -> str:
        pytest.fail("A short batch should not invoke the summary model.")

    assert budget.prepare(history, [], _wire, summarize) == _wire(history)
    assert budget.prepare(history, [], _wire, summarize) == _wire(history)
    skipped = [event for kind, event in events if kind == "summary"]
    assert len(skipped) == 1
    assert skipped[0]["status"] == "skipped"
    assert skipped[0]["tasks"] == [0, 1] and skipped[0]["input_tokens"] < skipped[0]["min_tokens"]
    assert budget.summary_requests == 0 and not list(tmp_path.glob("summary-request-*.json"))


def test_budget_overflow_shortens_largest_recent_output_without_dropping_tool_pairs(tmp_path: Path) -> None:
    assert ContextConfig().input_limit == 61_440
    events = []
    budget = ContextBudget(
        ContextConfig(context_window=2_500, output_reserve=256, preview_chars=300),
        tmp_path,
        lambda kind, **details: events.append((kind, details)),
    )
    history = [{"role": "user", "content": "Explain the project.", "extra": {"context_category": "task"}}]
    outputs = [("old", "Earlier directory listing"), ("short", "Recent details"), ("long", "x" * 9_000)]
    for call_id, output in outputs:
        history.extend([
            {"role": "assistant", "content": None, "tool_calls": [{"id": call_id}]},
            {"role": "tool", "tool_call_id": call_id, "content": output,
             "extra": {"evidence_id": f"E{len(history):06}"}},
        ])

    wire = budget.prepare(history, [], _wire)
    request = events[-1][1]
    assert request["emergency_shortened"] == [6]
    assert request["removed"] == [1, 2]
    assert not request["blocked"] and request["estimated"] <= budget.input_limit
    assert [m["tool_call_id"] for m in wire if m["role"] == "tool"] == ["short", "long"]
    assert [m["tool_calls"][0]["id"] for m in wire if m["role"] == "assistant"] == ["short", "long"]
    assert wire[2]["content"] == "Recent details"
    assert wire[4]["content"].startswith("Evidence E000005:")
    assert history[-1]["content"] == outputs[-1][1]


def test_workspace_source_change_filters_dependent_facts_but_keeps_constraints(tmp_path: Path) -> None:

    root = tmp_path / "project"
    root.mkdir()
    config_file = root / "config.py"
    config_file.write_text("PAGE_SIZE = 20\n")
    (root / "orders.py").write_text("from config import PAGE_SIZE\n")
    workspace = Workspace(root, WorkbenchConfig(storage_dir=tmp_path / "state"), {}, {})
    archive = EvidenceStore(tmp_path / "state")
    config = workspace.read_file("config.py")
    config_ref = archive.save("read_file config.py", config)
    orders_ref = archive.save("read_file orders.py", workspace.read_file("orders.py"))
    events = []
    budget = ContextBudget(ContextConfig(), tmp_path, lambda kind, **data: events.append(data))
    budget.progress = f"Default is 20 ({config_ref}, {orders_ref}).\nOrders imports config ({orders_ref})."
    budget.constraints = [Constraint(task=0, quote="Keep the old interface.")]
    history = [
        {"role": "assistant", "tool_calls": [{"id": "read"}], "content": None},
        {"role": "tool", "tool_call_id": "read", "content": config["output"],
         "extra": config["extra"] | {"evidence_id": config_ref}},
    ]

    def check(id: str) -> dict | None:
        source = archive.source(id)
        return source | {"status": workspace.check_source(source)} if source else None

    budget.check_source = check
    assert any("Default is 20" in (m.get("content") or "") for m in budget.prepare(history, [], _wire))
    config_file.write_text("PAGE_SIZE = 50\n")
    wire = budget.prepare(history, [], _wire)
    payload = "\n".join(m.get("content") or "" for m in wire)
    assert "Default is 20" not in payload and "PAGE_SIZE = 20" not in payload
    assert "Archived source needs recheck: config.py" in payload
    assert "Orders imports config" in payload and "Keep the old interface." in payload
    assert [m["tool_call_id"] for m in wire if m["role"] == "tool"] == ["read"]
    assert {e["evidence"]: e["status"] for e in events[-1]["source_checks"]} == {
        config_ref: "changed", orders_ref: "current",
    }
    assert archive.recall(config_ref)["output"] == "PAGE_SIZE = 20"
    config_file.unlink()
    assert check(config_ref)["status"] == "unavailable"
