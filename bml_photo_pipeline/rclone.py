from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


class RcloneError(RuntimeError):
    pass


def run_rclone(args: list[str], *, timeout: int = 300) -> str:
    base_args = [
        "rclone",
        "--retries",
        "3",
        "--low-level-retries",
        "10",
        "--contimeout",
        "20s",
        "--timeout",
        "120s",
        "--stats",
        "0",
    ]
    proc = subprocess.run(
        [*base_args, *args],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RcloneError(f"rclone {' '.join(args)} failed: {detail}")
    return proc.stdout


def mkdir(remote_path: str) -> None:
    run_rclone(["mkdir", remote_path])


def list_json(remote_path: str, *, recursive: bool = False) -> list[dict[str, Any]]:
    args = ["lsjson", remote_path]
    if recursive:
        args.append("--recursive")
    output = run_rclone(args)
    if not output.strip():
        return []
    return json.loads(output)


def copy_file(remote_path: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    run_rclone(["copyto", remote_path, str(local_dir / Path(remote_path).name)])


def copyto_local(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run_rclone(["copyto", remote_path, str(local_path)])


def copyto_remote(local_path: Path, remote_path: str) -> None:
    run_rclone(["copyto", str(local_path), remote_path])


def copy_dir_to_remote(local_dir: Path, remote_path: str) -> None:
    run_rclone(["copy", str(local_dir), remote_path])


def sync_dir_to_remote(local_dir: Path, remote_path: str) -> None:
    run_rclone(["sync", str(local_dir), remote_path])


def moveto_remote(source_remote: str, dest_remote: str) -> None:
    run_rclone(["moveto", source_remote, dest_remote])
