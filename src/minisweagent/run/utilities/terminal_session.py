"""Run the shared session engine in a terminal."""

import fcntl
import re
import sys
from pathlib import Path

import typer

from minisweagent.run.utilities.workbench import Session, _storage_dir


def _print_event(event: dict) -> None:
    match event["kind"]:
        case "assistant" | "notice" | "error":
            typer.echo(event["text"])
        case "tool":
            heading = event.get("recall_heading") or f"[{event['evidence']}] {event['command']}"
            typer.echo(f"{heading} (exit {event['returncode']})")
            text = event["text"]
            if text:
                typer.echo(text if len(text) <= 2400 else text[:1200] + "\n…\n" + text[-1200:])
            if event["error"]:
                typer.echo(event["error"])
        case "context":
            change = f"{event['before']:,} -> {event['estimated']:,} estimated input tokens"
            typer.echo(f"Context {event['request']} ({event['strategy']}): {change} / {event['limit']:,}")
            if budgets := event.get("section_budgets"):
                sections = [
                    f"{name.replace('_', ' ')} {event['categories'][name]:,}" + (f"/{limit:,}" if limit is not None else "")
                    for name, limit in budgets.items()
                ]
                typer.echo("  " + " · ".join(sections))
        case "usage":
            typer.echo(f"Model usage: input={event['input']}, cached={event['cached']}, cost={event['reported_cost']}")
        case "summary":
            typer.echo(f"Task summary: {event['status']}")
        case "evidence_search":
            typer.echo(f"Evidence search: {event['query']} ({event['hits']} hits)")
        case "recall":
            typer.echo(f"Evidence recalled: {event['evidence']}")
        case "verification":
            typer.echo(f"Verification: {event['status']} ({event['evidence']})")
        case "finish":
            if event["text"]:
                typer.echo(event["text"])
            typer.echo(f"Turn finished: {event['reason']}")


def _run_turn(session: Session, task: str, *, yolo: bool) -> int:
    offset = len(session.events)
    session.start(task)
    exit_code = 0
    try:
        while True:
            with session.condition:
                session.condition.wait_for(lambda: len(session.events) > offset or not session.running)
                events = session.events[offset:]
                offset += len(events)
                running = session.running
                pending = session.pending if running and session.decision is None else None
            for event in events:
                _print_event(event)
                if event["kind"] == "error" or (event["kind"] == "finish" and event["reason"] != "Submitted"):
                    exit_code = 1
            if pending:
                for command in pending["commands"]:
                    typer.echo(f"  {command}")
                session.decide(pending["id"], yolo or typer.confirm("Approve these actions?"))
            if not running:
                break
    except (KeyboardInterrupt, typer.Abort):
        exit_code = 130
    finally:
        with session.condition:
            if session.running:
                session.stop(session.turn_id)
        session.worker.join()
    if exit_code == 130:
        typer.echo("Turn stopped. Resume with --resume " + session.id)
    else:
        changes = session.changes_payload()
        typer.echo(f"Workspace changes: {len(changes['files'])} file(s) · verification {changes['verification']['status']}")
        for change in changes["files"]:
            typer.echo(f"  {change['status']} {change['path']}")
    return exit_code


def run_terminal_session(
    config: dict, task: str | None, *, workspace: Path | None, resume: str | None,
    yolo: bool, exit_immediately: bool, output: Path | None,
) -> None:
    storage = _storage_dir(config) / ".terminal"
    settings = config.get("workbench", {}) | {"storage_dir": storage}
    config = config | {"workbench": settings}
    if resume is None:
        session = Session(workspace or Path.cwd().resolve(), config)
        directory = session.directory
    else:
        directory = storage / resume
        if not re.fullmatch(r"[0-9a-f]{32}", resume) or not directory.is_dir() or directory.is_symlink():
            raise typer.BadParameter("Use a terminal session ID from this storage directory.", param_hint="--resume")

    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise typer.BadParameter("This terminal session is already open.", param_hint="--resume") from exc
        if resume is not None:
            try:
                session = Session.restore(directory, config)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--resume") from exc
            if session is None or not session.workspace_available:
                raise typer.BadParameter("Session or workspace unavailable.", param_hint="--resume")
            if workspace is not None and workspace != session.root:
                raise typer.BadParameter("Resume uses the session's original workspace.", param_hint="--workspace")

        typer.echo(f"Session ID: {session.id}\nArchive: {session.directory}\nWorkspace: {session.root}")
        typer.echo(f"Model: {session.model_name} · backend {session.config.backend} · context {session.strategy or 'unavailable'}")
        if info := session.resume_details():
            typer.echo(f"Previous task: {info['goal']}")
            if action := info["action"]:
                typer.echo(f"Last action ({action['state']}): {action['command']}")
            for warning in info.get("warnings", []):
                typer.echo(warning)
        exit_code = 0
        try:
            if task is None:
                task = typer.prompt("Task (Enter to quit)", default="", show_default=False).strip()
            while task:
                exit_code = _run_turn(session, task, yolo=yolo)
                if exit_code == 130 or exit_immediately or not sys.stdin.isatty():
                    break
                task = typer.prompt("Next task (Enter to quit)", default="", show_default=False).strip()
        finally:
            session.cleanup()
            trajectory = directory / "trajectory.json"
            if output is not None and trajectory.is_file():
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(trajectory.read_bytes())
                typer.echo(f"Saved trajectory to {output}")
        if exit_code:
            raise typer.Exit(exit_code)
