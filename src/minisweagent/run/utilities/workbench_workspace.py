import hashlib
import json
import os
import re
import shlex
import stat
from pathlib import Path
from threading import Event
from tempfile import NamedTemporaryFile
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from minisweagent.environments.docker import DockerEnvironment
from minisweagent.environments.local import LocalEnvironment
from minisweagent.run.utilities.workbench_changes import _workspace_files
from minisweagent.run.utilities.workbench_context import ContextConfig, write_json

_MAX_READ_BYTES = 2_000_000


class ProjectChecks(BaseModel):
    preflight: list[str] = Field(default_factory=list)
    verify: list[str] = Field(default_factory=list)
    verification_retries: int = Field(2, ge=0, le=10)

    @field_validator("preflight", "verify")
    @classmethod
    def validate_command(cls, argv: list[str]) -> list[str]:
        shells = {"sh", "bash", "dash", "zsh", "fish", "powershell", "cmd"}
        if argv and (not argv[0].strip() or Path(argv[0]).name in shells):
            raise ValueError("Use an executable argument list, for example [python, -m, pytest, -q].")
        return argv

    def dependency_check(self) -> list[str]:
        if self.preflight:
            return self.preflight
        if not self.verify:
            return []
        if Path(self.verify[0]).name.startswith("python") and self.verify[1:3] == ["-m", "pytest"]:
            return [*self.verify[:3], "--version"]
        return ["command", "-v", self.verify[0]]


class WorkbenchConfig(BaseModel):
    backend: Literal["local", "docker"] = "local"
    docker_image: str | None = None
    storage_dir: Path
    context: ContextConfig = Field(default_factory=ContextConfig)
    checks: ProjectChecks = Field(default_factory=ProjectChecks)


class Workspace:
    def __init__(self, root: Path, config: WorkbenchConfig, env_config: dict, explicit_checks: dict):
        self.root = root
        check_file = root / ".mini-workbench.yaml"
        try:
            project_checks = yaml.safe_load(check_file.read_text(encoding="utf-8")) if check_file.is_file() else {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid workspace configuration: {check_file.name}") from exc
        if project_checks is not None and not isinstance(project_checks, dict):
            raise ValueError(f"Expected a mapping in {check_file.name}.")
        self.checks = ProjectChecks.model_validate((project_checks or {}) | explicit_checks)
        rules = root / "AGENTS.md"
        self.rules = rules.read_text(encoding="utf-8") if rules.is_file() else ""
        env = {k: v for k, v in env_config.items() if k in {"timeout", "env"}}
        self.backend = config.backend
        if self.backend == "docker":
            if not config.docker_image or ":" in str(root):
                raise ValueError("Docker requires workbench.docker_image and a workspace path without a colon.")
            self.env = DockerEnvironment(
                image=config.docker_image,
                cwd="/workspace",
                run_args=[
                    "--rm", "--init", "--network=none", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                    "--user", f"{os.getuid()}:{os.getgid()}", "--volume", f"{root}:/workspace",
                ],
                **env,
            )
        else:
            self.env = LocalEnvironment(cwd=str(root), **env)

    def path(self, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts or candidate.parts[0] == ".git":
            raise ValueError("Use a file path relative to the workspace.")
        path = self.root / candidate
        if path.is_symlink() or path.resolve() != path or not path.is_relative_to(self.root):
            raise ValueError("File tools do not follow symlinks.")
        return path

    def version(self) -> str:
        return self.version_of(self.root)

    @staticmethod
    def version_of(root: Path) -> str:
        manifest = hashlib.sha256()
        for path in _workspace_files(root):
            if path.is_symlink() or not path.is_file() or path.resolve() != path:
                continue
            relative = path.relative_to(root).as_posix()
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(128 * 1024), b""):
                    digest.update(chunk)
            manifest.update(json.dumps([relative, digest.hexdigest()]).encode("utf-8"))
        return manifest.hexdigest()

    def check(self, argv: list[str], cancel_event: Event | None = None) -> dict:
        return self.env.execute({"command": shlex.join(argv)}, check_submission=False, cancel_event=cancel_event)

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> dict:
        target = self.path(path)
        data = _read_text_file(target)
        text = data.decode("utf-8")
        selected = _lines(text, start_line, end_line)
        return {
            "output": selected,
            "extra": {
                "path": path, "start_line": start_line,
                "end_line": min(end_line or start_line + 119, start_line + 119, len(text.splitlines())),
                "sha256": hashlib.sha256(data).hexdigest(),
            },
        }

    def check_source(self, source: dict) -> str:
        try:
            content = _read_text_file(self.path(source["path"]))
        except (ValueError, OSError, RuntimeError):
            return "unavailable"
        return "current" if hashlib.sha256(content).hexdigest() == source["sha256"] else "changed"

    def apply_patch(self, path: str, new_text: str, old_text: str | None = None, sha256: str | None = None) -> dict:
        if not isinstance(new_text, str) or (old_text is not None and not isinstance(old_text, str)):
            raise ValueError("Patch fragments must be strings.")
        target = self.path(path)
        payload = new_text.encode()
        if len(payload) > _MAX_READ_BYTES:
            raise ValueError("The replacement exceeds the text tool size limit.")
        if old_text is None:
            with target.open("xb") as stream:
                stream.write(payload)
        else:
            data = _read_text_file(target)
            if sha256 != hashlib.sha256(data).hexdigest():
                raise ValueError("The source changed or its SHA-256 was omitted. Read the file again.")
            old = old_text.encode()
            if not old or data.count(old) != 1:
                raise ValueError("old_text must match exactly once.")
            payload = data.replace(old, payload, 1)
            with NamedTemporaryFile(dir=target.parent, delete=False) as temp:
                temp_path = Path(temp.name)
                try:
                    temp.write(payload)
                    temp.flush()
                    temp_path.chmod(stat.S_IMODE(target.stat().st_mode))
                    if hashlib.sha256(target.read_bytes()).hexdigest() != sha256:
                        raise ValueError("The source changed while applying the patch.")
                    temp_path.replace(target)
                finally:
                    temp_path.unlink(missing_ok=True)
        return {"output": f"Updated {path}.", "extra": {"path": path, "sha256": hashlib.sha256(payload).hexdigest()}}


def _read_text_file(path: Path) -> bytes:
    if not path.is_file():
        raise ValueError(f"No regular file at {path.name}.")
    with path.open("rb") as stream:
        data = stream.read(_MAX_READ_BYTES + 1)
    if len(data) > _MAX_READ_BYTES or b"\0" in data:
        raise ValueError("File is binary or exceeds the text tool size limit.")
    return data


def _lines(text: str, start_line: int, end_line: int | None) -> str:
    if type(start_line) is not int or start_line < 1 or (
        end_line is not None and (type(end_line) is not int or end_line < start_line)
    ):
        raise ValueError("Line ranges use positive, inclusive line numbers.")
    end = min(end_line or start_line + 119, start_line + 119)
    selected = "\n".join(text.splitlines()[start_line - 1:end])
    return selected[:12_000] + ("\n[Preview truncated]" if len(selected) > 12_000 else "")


class EvidenceStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.count = max(
            (int(path.stem[1:]) for path in directory.glob("E*.json") if re.fullmatch(r"E[0-9]{6,}", path.stem)),
            default=0,
        )

    def save(self, label: str, result: dict) -> str:
        self.count += 1
        evidence = f"E{self.count:06}"
        write_json(self.directory / f"{evidence}.json", {"label": label, "result": result})
        return evidence

    def _load(self, id: str) -> dict:
        if not isinstance(id, str) or not re.fullmatch(r"E[0-9]{6,}", id):
            raise ValueError("Invalid evidence ID.")
        file = self.directory / f"{id}.json"
        if not file.is_file():
            raise ValueError("Evidence not found in this session.")
        return json.loads(file.read_text(encoding="utf-8"))

    def source(self, id: str) -> dict | None:
        try:
            extra = self._load(id)["result"].get("extra", {})
        except (ValueError, OSError):
            return None
        if all(key in extra for key in ("path", "sha256", "start_line", "end_line")):
            return {key: extra[key] for key in ("path", "sha256", "start_line", "end_line")}
        return None

    def recall(self, id: str, start_line: int = 1, end_line: int | None = None) -> dict:
        evidence = self._load(id)
        return {"output": _lines(evidence["result"]["output"], start_line, end_line), "extra": {
            "recalled": id, "start_line": start_line, "label": evidence["label"],
        }}
