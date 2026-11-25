"""Summarize one workbench run without calling the model."""

import json
from pathlib import Path

import typer

app = typer.Typer()


def _total(events: list[dict], key: str) -> dict:
    reported = [event[key] for event in events if isinstance(event.get(key), (int, float))]
    return {"total": sum(reported) if reported else None, "samples": len(reported)}


@app.command()
def main(
    directory: Path = typer.Argument(..., exists=True, file_okay=False),
    actual_cost: float | None = typer.Option(None, min=0, help="Actual provider charge, if known."),
    currency: str = typer.Option("CNY", help="Currency for --actual-cost."),
) -> None:
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    requests = [event for event in events if event["kind"] == "context"]
    usage = [event for event in events if event["kind"] == "usage"]
    summary_usage = [event for event in usage if event.get("purpose") == "summary"]
    recall = [event for event in events if event["kind"] == "recall"]
    terminal = [event["reason"] for event in events if event["kind"] == "finish"]
    change_path = directory / "changes.json"
    changes = json.loads(change_path.read_text(encoding="utf-8")) if change_path.is_file() else None
    typer.echo(json.dumps({
        "strategy": requests[-1]["strategy"] if requests else None,
        "exit_statuses": terminal, "requests": sum(not event["blocked"] for event in requests),
        "blocked_requests": sum(event["blocked"] for event in requests), "reported_usage_samples": len(usage),
        "estimated_peak_input": max((event["estimated"] for event in requests), default=None),
        **{key: _total(usage, key) for key in ("input", "cached", "uncached", "reported_cost")},
        "summary_calls_with_usage": len(summary_usage), "summary_input": _total(summary_usage, "input"),
        "recall_calls": len(recall), "recall_characters": _total(recall, "characters"),
        "recall_estimated_tokens": _total(recall, "estimated_tokens"),
        "recall_elapsed_ms": _total(recall, "elapsed_ms"),
        "actual_cost": {"amount": actual_cost, "currency": currency} if actual_cost is not None else None,
        "verification": changes["verification"] if changes else None,
        "diff": str(change_path) if changes else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
