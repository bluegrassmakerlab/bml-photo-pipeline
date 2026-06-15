from __future__ import annotations

import argparse
import fcntl
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config, resolve_path
from .processing import create_posting_pack, create_upload_ready_pack, media_type, process_file
from .rclone import copy_dir_to_remote, copyto_local, copyto_remote, list_json, mkdir, moveto_remote
from .state import file_key, load_state, save_state


def remote_join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *[part.strip("/") for part in parts if part]])


def ensure_remote_folders(config: dict) -> None:
    root = config["remote_root"]
    mkdir(root)
    for folder in config["folders"].values():
        mkdir(remote_join(root, folder))


def supported_extensions(config: dict) -> set[str]:
    if "supported_extensions" in config:
        return {extension.lower() for extension in config["supported_extensions"]}
    return {
        extension.lower()
        for key in ["supported_image_extensions", "supported_video_extensions"]
        for extension in config.get(key, [])
    }


def is_supported(name: str, extensions: set[str]) -> bool:
    suffix = Path(name).suffix.lower()
    return suffix in extensions and media_type(Path(name)) is not None


def upload_ready_groups(items: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    for item in sorted(items, key=lambda value: value["source"].name.lower()):
        current.append(item)
        if media_type(item["source"]) == "video":
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def process_once(config: dict, base_dir: Path) -> int:
    work_dir = resolve_path(base_dir, config["local_work_dir"])
    incoming_dir = work_dir / "incoming"
    processed_dir = work_dir / "processed"
    archive_dir = work_dir / "archive"
    needs_review_dir = work_dir / "needs-review"
    for path in [incoming_dir, processed_dir, archive_dir, needs_review_dir]:
        path.mkdir(parents=True, exist_ok=True)

    state_path = resolve_path(base_dir, config["state_file"])
    state = load_state(state_path)
    processed = state.setdefault("processed", {})
    upload_ready_state = state.setdefault("upload_ready", {})

    root = config["remote_root"]
    folders = config["folders"]
    incoming_remote = remote_join(root, folders["incoming"])
    extensions = supported_extensions(config)
    entries = [
        entry
        for entry in list_json(incoming_remote)
        if not entry.get("IsDir") and is_supported(entry.get("Name", ""), extensions)
    ]

    count = 0
    upload_ready_items = []
    for entry in entries:
        key = file_key(entry)
        if key in processed:
            continue

        name = entry["Name"]
        source_remote = remote_join(incoming_remote, name)
        local_source = incoming_dir / name
        local_review = needs_review_dir / name

        try:
            copyto_local(source_remote, local_source)
            exports = process_file(local_source, processed_dir, config)
            posting_pack_exports = create_posting_pack(local_source, exports, processed_dir, config)

            for export_name, local_path in exports.items():
                remote_folder = folders[export_name]
                copyto_remote(local_path, remote_join(root, remote_folder, local_path.name))

            posting_pack_remote = None
            if posting_pack_exports:
                posting_pack_remote = remote_join(root, folders["posting_pack"], local_source.stem)
                for local_path in posting_pack_exports.values():
                    copyto_remote(local_path, remote_join(posting_pack_remote, local_path.name))

            archive_remote = remote_join(root, folders["archive_originals"], name)
            moveto_remote(source_remote, archive_remote)

            processed[key] = {
                "name": name,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "archive_remote": archive_remote,
                "exports": {export_name: str(path) for export_name, path in exports.items()},
                "posting_pack_remote": posting_pack_remote,
                "posting_pack": {export_name: str(path) for export_name, path in posting_pack_exports.items()},
            }
            upload_ready_items.append({"source": local_source, "exports": exports})
            save_state(state_path, state)
            count += 1
        except Exception as exc:
            if local_source.exists():
                shutil.copy2(local_source, local_review)
            review_remote = remote_join(root, folders["needs_review"], name)
            try:
                moveto_remote(source_remote, review_remote)
            except Exception:
                pass
            processed[key] = {
                "name": name,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }
            save_state(state_path, state)

    if upload_ready_items:
        for upload_ready_group in upload_ready_groups(upload_ready_items):
            try:
                pack_dir, upload_ready_files = create_upload_ready_pack(upload_ready_group, processed_dir, config)
                if pack_dir and upload_ready_files:
                    upload_ready_remote = remote_join(root, folders.get("upload_ready", "30_Upload_Ready"), pack_dir.name)
                    copy_dir_to_remote(pack_dir, upload_ready_remote)
                    upload_ready_state[pack_dir.name] = {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "remote": upload_ready_remote,
                        "files": [str(path) for path in upload_ready_files],
                        "source_files": [str(item["source"].name) for item in upload_ready_group],
                    }
                    save_state(state_path, state)
            except Exception as exc:
                errors = state.setdefault("upload_ready_errors", [])
                errors.append(
                    {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "error": str(exc),
                        "source_files": [str(item["source"].name) for item in upload_ready_group],
                    }
                )
                save_state(state_path, state)

    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bluegrass Maker Lab product photo pipeline")
    parser.add_argument("--config", default="config/default.json", help="Path to config JSON")
    parser.add_argument("--once", action="store_true", help="Run one polling pass and exit")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between polling passes")
    args = parser.parse_args(argv)

    base_dir = Path.cwd()
    config_path = resolve_path(base_dir, args.config)
    config = load_config(config_path)
    state_path = resolve_path(base_dir, config["state_file"])
    lock_path = state_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another photo pipeline run is already active", flush=True)
            return 2

        ensure_remote_folders(config)

        while True:
            count = process_once(config, base_dir)
            print(f"processed {count} file(s)", flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
