#!/usr/bin/env python3

"""Run mini-SWE-agent in your local environment. This is the default executable `mini`."""
# Read this first: https://mini-swe-agent.com/latest/usage/mini/  (usage)

import os
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from minisweagent import global_config_dir
from minisweagent.agents import get_agent
from minisweagent.agents.utils.prompt_user import _multiline_prompt
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.utilities.config import configure_if_first_time
from minisweagent.utils.serialize import UNSET, recursive_merge

DEFAULT_CONFIG_FILE = Path(os.getenv("MSWEA_MINI_CONFIG_PATH", builtin_config_dir / "mini.yaml"))
DEFAULT_OUTPUT_FILE = global_config_dir / "last_mini_run.traj.json"


_HELP_TEXT = """Run mini-SWE-agent in your local environment.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/mini/[/bold green]
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """YAML files, built-in config names, or key=value overrides, merged left to right.

Session mode starts with the built-in mini.yaml. For ordinary mini, include [bold green]-c mini.yaml[/bold green]
when supplying overrides to the default config.
"""

console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use",),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'litellm' or 'minisweagent.models.litellm_model.LitellmModel')", rich_help_panel="Advanced"),
    agent_class: str | None = typer.Option(None, "--agent-class", help="Agent class to use (e.g., 'interactive' or 'minisweagent.agents.interactive.InteractiveAgent')", rich_help_panel="Advanced"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment class to use (e.g., 'local' or 'minisweagent.environments.local.LocalEnvironment')", rich_help_panel="Advanced"),
    session: Annotated[bool, typer.Option("--session", help="Start a persistent session with file and evidence tools.")] = False,
    resume: Annotated[str | None, typer.Option("--resume", help="Resume a terminal session by its printed ID.")] = None,
    workspace: Annotated[Path | None, typer.Option("-w", "--workspace", exists=True, file_okay=False, resolve_path=True, help="Workspace for a persistent session.")] = None,
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: list[str] = typer.Option([], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT),
    output: Path | None = typer.Option(None, "-o", "--output", help="Export the trajectory to this file."),
    exit_immediately: bool = typer.Option(False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Advanced"),
) -> Any:
    # fmt: on
    if session or resume is not None or workspace is not None:
        from minisweagent.run.utilities.terminal_session import run_terminal_session
        from minisweagent.run.utilities.workbench import get_workbench_config

        if agent_class or environment_class:
            raise typer.BadParameter("Session mode uses the shared agent. Set workbench.backend to select its environment.")
        config = recursive_merge(get_workbench_config(config_spec), {
            "agent": {"cost_limit": cost_limit if cost_limit is not None else UNSET},
            "model": {"model_class": model_class or UNSET, "model_name": model_name or UNSET},
        })
        return run_terminal_session(
            config, task or config.get("run", {}).get("task"), workspace=workspace, resume=resume,
            yolo=yolo, exit_immediately=exit_immediately, output=output,
        )

    configure_if_first_time()

    # Build the config from the command line arguments
    config_spec = config_spec or [str(DEFAULT_CONFIG_FILE)]
    console.print(f"Building agent config from specs: [bold green]{config_spec}[/bold green]")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "run": {
            "task": task or UNSET,
        },
        "agent": {
            "agent_class": agent_class or UNSET,
            "mode": "yolo" if yolo else UNSET,
            "cost_limit": cost_limit if cost_limit is not None else UNSET,
            "confirm_exit": False if exit_immediately else UNSET,
            "output_path": output or DEFAULT_OUTPUT_FILE,
        },
        "model": {
            "model_class": model_class or UNSET,
            "model_name": model_name or UNSET,
        },
        "environment": {
            "environment_class": environment_class or UNSET,
        },
    })
    config = recursive_merge(*configs)

    if (run_task := config.get("run", {}).get("task", UNSET)) is UNSET:
        console.print("[bold yellow]What do you want to do?")
        run_task = _multiline_prompt()
        console.print("[bold green]Got that, thanks![/bold green]")

    model = get_model(config=config.get("model", {}))
    env = get_environment(config.get("environment", {}), default_type="local")
    agent = get_agent(model, env, config.get("agent", {}), default_type="interactive")
    agent.run(run_task)
    if (output_path := config.get("agent", {}).get("output_path")):
        console.print(f"Saved trajectory to [bold green]'{output_path}'[/bold green]")
    return agent


if __name__ == "__main__":
    app()
