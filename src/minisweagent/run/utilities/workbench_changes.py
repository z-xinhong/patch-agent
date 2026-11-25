import difflib
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from subprocess import run

_MAX_FILES = 3000
_MAX_FILE_BYTES = 128_000
_MAX_TOTAL_BYTES = 8_000_000
_IGNORED_DIRS = {".git", ".venv", ".idea", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__", "node_modules"}


@dataclass
class WorkspaceSnapshot:
    files: dict[str, tuple[str, str | None]]
    limited: bool = False

    @classmethod
    def capture(cls, root: Path) -> "WorkspaceSnapshot":
        snapshot = cls({})
        total = 0
        for path in _workspace_files(root):
            if len(snapshot.files) == _MAX_FILES:
                snapshot.limited = True
                break
            try:
                if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                    continue
                stat = path.stat()
                if stat.st_size > _MAX_FILE_BYTES or total + stat.st_size > _MAX_TOTAL_BYTES:
                    snapshot.files[path.relative_to(root).as_posix()] = (
                        f"{stat.st_size}:{stat.st_mtime_ns}", None
                    )
                    snapshot.limited = True
                    continue
                data = path.read_bytes()
            except OSError:
                snapshot.limited = True
                continue
            total += len(data)
            content = None if b"\0" in data else data.decode("utf-8", errors="replace")
            snapshot.files[path.relative_to(root).as_posix()] = (hashlib.sha256(data).hexdigest(), content)
        return snapshot

    def changes(self, current: "WorkspaceSnapshot") -> list[dict]:
        changes = []
        for path in sorted(self.files.keys() | current.files.keys()):
            before, after = self.files.get(path), current.files.get(path)
            if before == after:
                continue
            diff = None
            if (before is None or before[1] is not None) and (after is None or after[1] is not None):
                diff = "\n".join(
                    difflib.unified_diff(
                        before[1].splitlines() if before else [],
                        after[1].splitlines() if after else [],
                        fromfile=f"a/{path}" if before else "/dev/null",
                        tofile=f"b/{path}" if after else "/dev/null",
                        lineterm="",
                    )
                )
            changes.append(
                {
                    "path": path,
                    "status": "added" if before is None else "deleted" if after is None else "modified",
                    "diff": diff[:48_000] if diff is not None else None,
                    "truncated": diff is not None and len(diff) > 48_000,
                }
            )
            if len(changes) == 100:
                break
        return changes


def _workspace_files(root: Path) -> Iterator[Path]:
    if git := which("git"):
        result = run(
            [git, "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            capture_output=True,
        )
        if result.returncode == 0:
            names = sorted(set(result.stdout.split(b"\0")) - {b""})
            yield from (root / name.decode(errors="surrogateescape") for name in names)
            return

    def walk(path: Path) -> Iterator[Path]:
        for child in sorted(path.iterdir()):
            if child.is_symlink() or child.name in _IGNORED_DIRS:
                continue
            if child.is_dir():
                yield from walk(child)
            elif child.is_file():
                yield child

    yield from walk(root)
