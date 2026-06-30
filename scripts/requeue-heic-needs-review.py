#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bml_photo_pipeline.config import load_config, resolve_path
from bml_photo_pipeline.state import load_state, save_state


def remote_join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *[part.strip("/") for part in parts if part]])


def run_rclone_moveto(source: str, target: str) -> bool:
    result = subprocess.run(["rclone", "moveto", source, target], text=True, capture_output=True, check=False)
    if result.returncode == 0:
        return True
    detail = result.stderr.strip() or result.stdout.strip()
    print(f"remote move skipped: {source} -> {target}: {detail}")
    return False


def heic_failure(record: dict) -> bool:
    name = str(record.get("name") or "").lower()
    error = str(record.get("error") or "").lower()
    return name.endswith((".heic", ".heif")) and "cannot identify image file" in error


def main() -> int:
    parser = argparse.ArgumentParser(description="Requeue HEIC files that failed before pillow-heif was installed.")
    parser.add_argument("--config", default="config/default.json", help="Path to config JSON")
    parser.add_argument("--apply", action="store_true", help="Move files and clear failed state entries")
    args = parser.parse_args()

    base_dir = Path.cwd()
    config = load_config(resolve_path(base_dir, args.config))
    state_path = resolve_path(base_dir, config.get("state_file", "state/processed.json"))
    state = load_state(state_path)
    processed = state.setdefault("processed", {})
    folders = config["folders"]
    root = config["remote_root"]
    work_dir = resolve_path(base_dir, config["local_work_dir"])
    incoming_dir = work_dir / "incoming"
    review_dir = work_dir / "needs-review"

    candidates = [(key, record) for key, record in processed.items() if isinstance(record, dict) and heic_failure(record)]
    if not candidates:
        print("No HEIC identify failures found in state.")
        return 0

    print(f"Found {len(candidates)} HEIC file(s) to requeue.")
    if not args.apply:
        for _, record in candidates[:20]:
            print(f"dry-run: {record.get('name')}")
        if len(candidates) > 20:
            print(f"dry-run: ... and {len(candidates) - 20} more")
        print("Run again with --apply to move files back to incoming and clear failed state.")
        return 0

    incoming_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    missing_local = 0
    for key, record in candidates:
        name = str(record.get("name") or "")
        local_review = review_dir / name
        local_incoming = incoming_dir / name
        source_remote = remote_join(root, folders["needs_review"], name)
        target_remote = remote_join(root, folders["incoming"], name)

        remote_moved = run_rclone_moveto(source_remote, target_remote)
        if local_review.exists():
            shutil.move(str(local_review), str(local_incoming))
        else:
            missing_local += 1
        processed.pop(key, None)
        moved += 1 if remote_moved or local_incoming.exists() else 0

    save_state(state_path, state)
    print(f"Requeued {moved} HEIC file(s).")
    if missing_local:
        print(f"{missing_local} local Needs Review file(s) were missing, but state was cleared for remote retry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
