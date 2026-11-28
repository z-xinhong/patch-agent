import hashlib
import json
import os
import re
import shlex
import stat
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Event
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from minisweagent.environments.docker import DockerEnvironment
from minisweagent.environments.local import LocalEnvironment
from minisweagent.run.utilities.workbench_changes import _workspace_files
from minisweagent.run.utilities.workbench_context import ContextConfig, write_json
from minisweagent.run.utilities.workbench_lifecycle import check_cancelled

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
        selected, end, total, truncated = _line_preview(text, start_line, end_line)
        return {
            "output": selected,
            "extra": {
                "path": path, "start_line": start_line,
                "end_line": end if end is not None else total,
                "file_lines": total, "read_truncated": truncated,
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


def _line_preview(text: str, start_line: int, end_line: int | None) -> tuple[str, int | None, int, bool]:
    if type(start_line) is not int or start_line < 1 or (
        end_line is not None and (type(end_line) is not int or end_line < start_line)
    ):
        raise ValueError("Line ranges use positive, inclusive line numbers.")
    lines = text.splitlines()
    end = min(end_line or start_line + 119, start_line + 119, len(lines))
    if start_line > end:
        return "", None, len(lines), False
    selected = "\n".join(lines[start_line - 1:end])
    if len(selected) <= 12_000:
        return selected, end, len(lines), False
    preview = selected[:12_000]
    last_visible = start_line + preview.removesuffix("\n").count("\n")
    return preview + "\n[Preview truncated]", last_visible, len(lines), True


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
        while True:
            try:
                extra = self._load(id)["result"].get("extra", {})
            except (ValueError, OSError):
                return None
            keys = ("path", "sha256", "start_line", "end_line")
            if all(key in extra for key in keys):
                return {key: extra[key] for key in keys}
            parent = extra.get("recalled")
            if not isinstance(parent, str) or not re.fullmatch(r"E[0-9]{6,}", parent) or int(parent[1:]) >= int(id[1:]):
                return None
            id = parent

    def search(
        self, query: str, limit: int = 5, *, cancel_event: Event | None = None, with_context: bool = False,
    ) -> list[dict]:
        if not isinstance(query, str) or not query.strip() or len(query) > 200:
            raise ValueError("Use a search query of 1 to 200 characters.")
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("Search limit must be an integer from 1 to 10.")

        words = [re.escape(word) for word in query.split()]
        terms = [re.compile(word, re.IGNORECASE) for word in dict.fromkeys(words)]
        phrase = re.compile(r"\s+".join(words), re.IGNORECASE)
        query_path = query.strip().removeprefix("./").casefold()
        matches = []
        for path in self.directory.glob("E*.json"):
            if cancel_event is not None:
                check_cancelled(cancel_event)
            if not re.fullmatch(r"E[0-9]{6,}", path.stem):
                continue
            evidence = self._load(path.stem)
            label, result = evidence["label"], evidence["result"]
            extra = result.get("extra", {})
            if "recalled" in extra or "search_sources" in extra:
                continue
            source_path = extra.get("path", "")
            metadata = f"{label} {source_path}"
            label_terms = {term for term in terms if term.search(metadata)}
            label_phrase = bool(phrase.search(metadata))
            score = (len(label_terms), label_phrase, 0)
            line_number, preview = None, ""
            lines = result["output"].splitlines()
            for number, line in enumerate(lines, 1):
                positions = {term: match.start() for term in terms if (match := term.search(line))}
                if not positions:
                    continue
                exact = phrase.search(line)
                rank = (len(label_terms | positions.keys()), label_phrase or bool(exact), len(positions))
                if rank > score:
                    score = rank
                    offset = exact.start() if exact else min(positions.values())
                    start = max(0, offset - 40)
                    preview = ("…" if start else "") + line[start:start + 200]
                    if start + 200 < len(line):
                        preview += "…"
                    line_number = number
            exact_path = query_path == source_path.casefold()
            if score[0] or exact_path:
                if with_context and exact_path and line_number is None:
                    line_number = next((i for i, line in enumerate(lines, 1) if line.strip()), None)
                    if line_number is not None:
                        line = lines[line_number - 1]
                        preview = line[:200] + ("…" if len(line) > 200 else "")
                hit = {
                    "id": path.stem, "label": label, "line": line_number, "preview": preview,
                }
                if with_context and line_number is not None and line_number < len(lines):
                    current, following = lines[line_number - 1 : line_number + 1]
                    if following.strip() and (
                        exact_path or current.rstrip().endswith((":", "(", "{")) or any(term.search(following) for term in terms)
                    ):
                        hit["following"] = following[:200] + ("…" if len(following) > 200 else "")
                matches.append(((exact_path, *score, int(path.stem[1:])), hit))
        return [hit for _, hit in sorted(matches, key=lambda match: match[0], reverse=True)[:limit]]

    def recall(self, id: str, start_line: int = 1, end_line: int | None = None) -> dict:
        evidence = self._load(id)
        extra = evidence["result"].get("extra", {})
        output, last_visible, total, truncated = _line_preview(evidence["result"]["output"], start_line, end_line)
        return {"output": output, "extra": {
            "recalled": id, "start_line": start_line, "label": evidence["label"],
            "recalled_end_line": last_visible, "archive_lines": total, "recall_truncated": truncated,
            "derived_recall": "recalled" in extra or "search_sources" in extra,
        }}
