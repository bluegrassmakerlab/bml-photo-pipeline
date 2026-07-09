from __future__ import annotations

import argparse
import fcntl
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config, resolve_path
from .heic_convert import HEIC_EXTENSIONS, convert_heic_file
from .processing import (
    create_posting_pack,
    create_upload_ready_pack,
    load_tracker_products,
    media_type,
    process_file,
    source_output_stem,
    upload_ready_settings,
)
from .rclone import copyto_local, copyto_remote, list_json, mkdir, moveto_remote, sync_dir_to_remote
from .state import file_key, load_state, save_state


def remote_join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *[part.strip("/") for part in parts if part]])


def uses_local_storage(config: dict) -> bool:
    return str(config.get("storage_mode") or "").lower() == "local"


def local_storage_root(config: dict, base_dir: Path) -> Path:
    return resolve_path(base_dir, str(config.get("local_root") or "."))


def local_folder(config: dict, base_dir: Path, folder_key: str) -> Path:
    return local_storage_root(config, base_dir) / config["folders"][folder_key]


def local_entry(root: Path, path: Path) -> dict:
    stat = path.stat()
    relative = path.relative_to(root)
    return {
        "Path": relative.as_posix(),
        "Name": path.name,
        "IsDir": path.is_dir(),
        "Size": stat.st_size,
        "ModTime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def list_local_entries(root: Path, *, recursive: bool = False) -> list[dict]:
    if not root.exists():
        return []
    paths = root.rglob("*") if recursive else root.iterdir()
    return [local_entry(root, path) for path in paths]


def existing_local_relative_paths(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix().lower()
        for path in root.rglob("*")
        if path.is_file()
    } if root.exists() else set()


def copy_file_to_folder(source: Path, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / source.name
    shutil.copy2(source, target)
    return target


def sync_dir_to_local(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def move_local_unique(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate = target
    suffix_index = 2
    while candidate.exists():
        candidate = target.with_name(f"{target.stem}-{suffix_index}{target.suffix}")
        suffix_index += 1
    shutil.move(str(source), str(candidate))
    return candidate


def ensure_remote_folders(config: dict) -> None:
    if uses_local_storage(config):
        root = local_storage_root(config, Path.cwd())
        root.mkdir(parents=True, exist_ok=True)
        for folder in config["folders"].values():
            (root / folder).mkdir(parents=True, exist_ok=True)
        return

    root = config["remote_root"]
    mkdir(root)
    for folder in config["folders"].values():
        mkdir(remote_join(root, folder))


def safe_product_folder_name(name: str) -> str:
    cleaned = " ".join(str(name or "").replace("/", "-").replace("\\", "-").split())
    return cleaned.strip(" .")


def sync_tracker_product_incoming_folders(config: dict, state: dict | None = None) -> dict[str, list[str]]:
    settings = upload_ready_settings(config)
    if not settings.get("sync_tracker_product_folders", False):
        return {"created": [], "renamed": [], "conflicts": []}

    if uses_local_storage(config):
        incoming_path = local_folder(config, Path.cwd(), "incoming")
        incoming_path.mkdir(parents=True, exist_ok=True)
        existing = {path.name for path in incoming_path.iterdir() if path.is_dir()}
    else:
        root = config["remote_root"]
        incoming_remote = remote_join(root, config["folders"]["incoming"])
        existing = {
            entry_relative_path(entry).name.rstrip("/")
            for entry in list_json(incoming_remote, recursive=False)
            if entry.get("IsDir")
        }

    result: dict[str, list[str]] = {"created": [], "renamed": [], "conflicts": []}
    folder_state = state.setdefault("tracker_product_folders", {}) if state is not None else {}
    updated_at = datetime.now(timezone.utc).isoformat()
    for product in load_tracker_products(settings):
        product_key = str(product.get("id") or product.get("sku") or product.get("name") or "").strip()
        folder_name = safe_product_folder_name(str(product.get("name") or ""))
        if not product_key or not folder_name:
            continue

        previous = folder_state.get(product_key) if isinstance(folder_state, dict) else None
        previous_folder = safe_product_folder_name(str(previous.get("folder") or "")) if isinstance(previous, dict) else ""
        if previous_folder and previous_folder != folder_name:
            if previous_folder in existing and folder_name not in existing:
                if uses_local_storage(config):
                    (incoming_path / previous_folder).rename(incoming_path / folder_name)
                else:
                    moveto_remote(remote_join(incoming_remote, previous_folder), remote_join(incoming_remote, folder_name))
                existing.discard(previous_folder)
                existing.add(folder_name)
                result["renamed"].append(f"{previous_folder} -> {folder_name}")
            elif previous_folder in existing and folder_name in existing:
                result["conflicts"].append(f"{previous_folder} -> {folder_name}")

        if folder_name not in existing:
            if uses_local_storage(config):
                (incoming_path / folder_name).mkdir(parents=True, exist_ok=True)
            else:
                mkdir(remote_join(incoming_remote, folder_name))
            existing.add(folder_name)
            result["created"].append(folder_name)

        if isinstance(folder_state, dict):
            folder_state[product_key] = {
                "product_id": product.get("id"),
                "product_name": product.get("name") or "",
                "folder": folder_name,
                "updated_at": updated_at,
            }
    return result


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


def is_heic_name(name: str) -> bool:
    return Path(name).suffix.lower() in HEIC_EXTENSIONS


def entry_relative_path(entry: dict) -> Path:
    return Path(entry.get("Path") or entry.get("Name", ""))


def jpeg_relative_path(source_relative_path: Path) -> Path:
    return source_relative_path.with_suffix(".jpg")


def existing_remote_relative_paths(remote_path: str) -> set[str]:
    try:
        entries = list_json(remote_path, recursive=True)
    except Exception:
        return set()
    return {
        entry_relative_path(entry).as_posix().lower()
        for entry in entries
        if not entry.get("IsDir")
    }


def unique_remote_relative_path(relative_path: Path, existing_paths: set[str]) -> Path:
    candidate = relative_path
    suffix_index = 2
    while candidate.as_posix().lower() in existing_paths:
        candidate = relative_path.with_name(f"{relative_path.stem}-{suffix_index}{relative_path.suffix}")
        suffix_index += 1
    existing_paths.add(candidate.as_posix().lower())
    return candidate


def convert_remote_heic_inbox(config: dict, base_dir: Path) -> dict[str, int]:
    if uses_local_storage(config):
        folders = config["folders"]
        source_folder = folders.get("heic_inbox", "00_HEIC_To_Convert")
        output_folder = folders.get("jpeg_for_editing", "05_JPEG_For_Editing")
        archive_folder = folders.get("archive_originals", "90_Archive/Originals")
        root = local_storage_root(config, base_dir)
        source_root = root / source_folder
        output_root = root / output_folder
        archive_root = root / archive_folder / source_folder
        existing_outputs = existing_local_relative_paths(output_root)
        existing_archives = existing_local_relative_paths(archive_root)
        quality = int(config.get("heic_conversion", {}).get("jpeg_quality", 95))
        quality = max(1, min(100, quality))

        entries = [
            path
            for path in source_root.rglob("*")
            if path.is_file() and is_heic_name(path.name)
        ] if source_root.exists() else []

        counts = {"converted": 0, "skipped": 0, "failed": 0, "archived": 0, "archive_failed": 0}
        for source_path in entries:
            relative_path = source_path.relative_to(source_root)
            output_relative = jpeg_relative_path(relative_path)
            output_path = output_root / output_relative

            if output_relative.as_posix().lower() in existing_outputs:
                counts["skipped"] += 1
                archive_relative = unique_remote_relative_path(relative_path, existing_archives)
                try:
                    move_local_unique(source_path, archive_root / archive_relative)
                    counts["archived"] += 1
                    print(f"archived: {relative_path} -> {archive_relative}", flush=True)
                except Exception as exc:
                    counts["archive_failed"] += 1
                    print(f"archive failed: {relative_path}: {exc}", flush=True)
                continue

            try:
                result = convert_heic_file(source_path, output_path, quality=quality, overwrite=True)
                if result.status == "failed":
                    counts["failed"] += 1
                    print(f"failed: {relative_path}: {result.message}", flush=True)
                    continue
                existing_outputs.add(output_relative.as_posix().lower())
                counts["converted"] += 1
                print(f"converted: {relative_path} -> {output_relative}", flush=True)
                archive_relative = unique_remote_relative_path(relative_path, existing_archives)
                try:
                    move_local_unique(source_path, archive_root / archive_relative)
                    counts["archived"] += 1
                    print(f"archived: {relative_path} -> {archive_relative}", flush=True)
                except Exception as exc:
                    counts["archive_failed"] += 1
                    print(f"archive failed: {relative_path}: {exc}", flush=True)
            except Exception as exc:
                counts["failed"] += 1
                print(f"failed: {relative_path}: {exc}", flush=True)

        print(
            (
                "heic conversion summary: "
                f"{counts['converted']} converted, "
                f"{counts['skipped']} skipped, "
                f"{counts['failed']} failed, "
                f"{counts['archived']} archived, "
                f"{counts['archive_failed']} archive failed"
            ),
            flush=True,
        )
        return counts

    root = config["remote_root"]
    folders = config["folders"]
    source_folder = folders.get("heic_inbox", "00_HEIC_To_Convert")
    output_folder = folders.get("jpeg_for_editing", "05_JPEG_For_Editing")
    archive_folder = folders.get("archive_originals", "90_Archive/Originals")
    source_remote_root = remote_join(root, source_folder)
    output_remote_root = remote_join(root, output_folder)
    archive_remote_root = remote_join(root, archive_folder, source_folder)
    existing_outputs = existing_remote_relative_paths(output_remote_root)
    existing_archives = existing_remote_relative_paths(archive_remote_root)

    work_dir = resolve_path(base_dir, config["local_work_dir"])
    local_source_root = work_dir / "heic-to-jpeg" / "source"
    local_output_root = work_dir / "heic-to-jpeg" / "jpeg"
    quality = int(config.get("heic_conversion", {}).get("jpeg_quality", 95))
    quality = max(1, min(100, quality))

    entries = [
        entry
        for entry in list_json(source_remote_root, recursive=True)
        if not entry.get("IsDir") and is_heic_name(entry.get("Name", ""))
    ]

    counts = {"converted": 0, "skipped": 0, "failed": 0, "archived": 0, "archive_failed": 0}
    for entry in entries:
        relative_path = entry_relative_path(entry)
        output_relative = jpeg_relative_path(relative_path)
        source_remote = remote_join(source_remote_root, relative_path.as_posix())

        if output_relative.as_posix().lower() in existing_outputs:
            counts["skipped"] += 1
            archive_relative = unique_remote_relative_path(relative_path, existing_archives)
            archive_remote = remote_join(archive_remote_root, archive_relative.as_posix())
            try:
                mkdir(str(Path(archive_remote).parent))
                moveto_remote(source_remote, archive_remote)
                counts["archived"] += 1
                print(f"archived: {relative_path} -> {archive_relative}", flush=True)
            except Exception as exc:
                counts["archive_failed"] += 1
                print(f"archive failed: {relative_path}: {exc}", flush=True)
            continue

        local_source = local_source_root / relative_path
        local_target = local_output_root / output_relative
        output_remote = remote_join(output_remote_root, output_relative.as_posix())

        try:
            copyto_local(source_remote, local_source)
            result = convert_heic_file(local_source, local_target, quality=quality, overwrite=True)
            if result.status == "failed":
                counts["failed"] += 1
                print(f"failed: {relative_path}: {result.message}", flush=True)
                continue
            copyto_remote(local_target, output_remote)
            existing_outputs.add(output_relative.as_posix().lower())
            counts["converted"] += 1
            print(f"converted: {relative_path} -> {output_relative}", flush=True)
            archive_relative = unique_remote_relative_path(relative_path, existing_archives)
            archive_remote = remote_join(archive_remote_root, archive_relative.as_posix())
            try:
                mkdir(str(Path(archive_remote).parent))
                moveto_remote(source_remote, archive_remote)
                counts["archived"] += 1
                print(f"archived: {relative_path} -> {archive_relative}", flush=True)
            except Exception as exc:
                counts["archive_failed"] += 1
                print(f"archive failed: {relative_path}: {exc}", flush=True)
        except Exception as exc:
            counts["failed"] += 1
            print(f"failed: {relative_path}: {exc}", flush=True)

    print(
        (
            "heic conversion summary: "
            f"{counts['converted']} converted, "
            f"{counts['skipped']} skipped, "
            f"{counts['failed']} failed, "
            f"{counts['archived']} archived, "
            f"{counts['archive_failed']} archive failed"
        ),
        flush=True,
    )
    return counts


def upload_ready_groups(items: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    by_parent: dict[Path, list[dict]] = {}
    for item in items:
        by_parent.setdefault(Path(item["source"]).parent, []).append(item)
    for parent in sorted(by_parent, key=lambda value: value.as_posix().lower()):
        current: list[dict] = []
        leading_videos: list[dict] = []
        for item in sorted(by_parent[parent], key=lambda value: value["source"].name.lower()):
            if media_type(item["source"]) == "video":
                if current:
                    current.append(item)
                    groups.append(current)
                    current = []
                else:
                    if leading_videos:
                        groups.append(leading_videos)
                        leading_videos = []
                    leading_videos.append(item)
                continue
            if leading_videos and not current:
                current.extend(leading_videos)
                leading_videos = []
            current.append(item)
        if current:
            groups.append(current)
        for video in leading_videos:
            groups.append([video])
    return groups


def is_loose_incoming_group(group: list[dict]) -> bool:
    parents = {Path(item["source"]).parent.name.lower() for item in group}
    return parents == {"incoming"}


def split_group_by_media_limits(group: list[dict], max_auto_images: int, max_auto_videos: int) -> list[list[dict]]:
    split_groups: list[list[dict]] = []
    current: list[dict] = []
    current_images = 0
    current_videos = 0

    for item in group:
        source_type = media_type(item["source"])
        next_images = current_images + (1 if source_type == "image" else 0)
        next_videos = current_videos + (1 if source_type == "video" else 0)
        if current and (next_images > max_auto_images or next_videos > max_auto_videos):
            split_groups.append(current)
            current = []
            current_images = 0
            current_videos = 0
        current.append(item)
        current_images += 1 if source_type == "image" else 0
        current_videos += 1 if source_type == "video" else 0

    if current:
        split_groups.append(current)
    return split_groups


def split_ambiguous_groups_by_product(groups: list[list[dict]], config: dict) -> list[list[dict]]:
    settings = upload_ready_settings(config)
    max_auto_images = int(settings.get("max_auto_images", 4))
    max_auto_videos = int(settings.get("max_auto_videos", 1))
    split_groups: list[list[dict]] = []

    for group in groups:
        images = [item for item in group if media_type(item["source"]) == "image"]
        videos = [item for item in group if media_type(item["source"]) == "video"]
        if len(images) <= max_auto_images and len(videos) <= max_auto_videos:
            split_groups.append(group)
            continue
        if is_loose_incoming_group(group):
            split_groups.extend(split_group_by_media_limits(group, max_auto_images, max_auto_videos))
            continue

        current: list[dict] = []
        current_key = ""
        leading_videos: list[dict] = []
        for item in group:
            source_type = media_type(item["source"])
            if source_type == "video":
                current_videos = [entry for entry in current if media_type(entry["source"]) == "video"]
                if current and len(current_videos) >= max_auto_videos:
                    split_groups.append(current)
                    current = []
                    leading_videos = [item]
                elif current:
                    current.append(item)
                else:
                    if len(leading_videos) >= max_auto_videos:
                        split_groups.append(leading_videos)
                        leading_videos = []
                    leading_videos.append(item)
                continue

            item_settings = upload_ready_settings(config, [item])
            item_key = str(item_settings.get("tracker_product_id") or "")
            item_hint = str(item_settings.get("sku") or item_settings.get("product_name") or "")
            next_item = {**item}
            if item_hint:
                next_item["product_hint"] = item_hint

            if leading_videos and not current:
                current.extend(leading_videos)
                leading_videos = []
            if current and item_key and current_key and item_key != current_key:
                split_groups.append(current)
                current = []
            current.append(next_item)
            if item_key:
                current_key = item_key

        if current:
            split_groups.append(current)
        for video in leading_videos:
            split_groups.append([video])

    return split_groups


def upload_ready_source_names(upload_ready_state: dict) -> set[str]:
    names: set[str] = set()
    for record in upload_ready_state.values():
        if not isinstance(record, dict):
            continue
        for source in record.get("source_files", []) or []:
            names.add(Path(str(source)).name)
    return names


def archive_relative_path(config: dict, archive_remote: str, fallback_name: str) -> Path:
    if uses_local_storage(config):
        archive_root = (
            local_storage_root(config, Path.cwd()) / config["folders"]["archive_originals"]
        ).as_posix().rstrip("/") + "/"
        if archive_remote.startswith(archive_root):
            relative = archive_remote[len(archive_root) :].strip("/")
            if relative:
                return Path(relative)

    archive_root = remote_join(config["remote_root"], config["folders"]["archive_originals"]).rstrip("/") + "/"
    if archive_remote.startswith(archive_root):
        relative = archive_remote[len(archive_root) :].strip("/")
        if relative:
            return Path(relative)
    return Path(fallback_name)


def pending_upload_ready_items(state: dict, base_dir: Path, config: dict) -> list[dict]:
    processed = state.get("processed", {}) if isinstance(state, dict) else {}
    upload_ready_state = state.get("upload_ready", {}) if isinstance(state, dict) else {}
    packeted_names = upload_ready_source_names(upload_ready_state)
    work_dir = resolve_path(base_dir, config["local_work_dir"])
    incoming_dir = work_dir / "incoming"
    items = []

    for record in processed.values():
        if not isinstance(record, dict) or record.get("error") or record.get("upload_ready_skipped"):
            continue
        name = str(record.get("name") or "")
        if not name or Path(name).name in packeted_names:
            continue
        exports = record.get("exports", {})
        if not isinstance(exports, dict) or not exports:
            continue
        resolved_exports = {export_name: resolve_path(base_dir, str(path)) for export_name, path in exports.items()}
        if not any(path.exists() for path in resolved_exports.values()):
            continue
        relative_path = archive_relative_path(config, str(record.get("archive_remote") or ""), name)
        items.append({"source": incoming_dir / relative_path, "exports": resolved_exports})

    return items


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

    folders = config["folders"]
    folder_sync = sync_tracker_product_incoming_folders(config, state)
    if any(folder_sync.values()):
        save_state(state_path, state)
    extensions = supported_extensions(config)
    local_mode = uses_local_storage(config)
    if local_mode:
        root_path = local_storage_root(config, base_dir)
        incoming_root = root_path / folders["incoming"]
        entries = [
            entry
            for entry in list_local_entries(incoming_root, recursive=bool(config.get("incoming_recursive", True)))
            if not entry.get("IsDir") and is_supported(entry.get("Name", ""), extensions)
        ]
    else:
        root = config["remote_root"]
        incoming_remote = remote_join(root, folders["incoming"])
        entries = [
            entry
            for entry in list_json(incoming_remote, recursive=bool(config.get("incoming_recursive", True)))
            if not entry.get("IsDir") and is_supported(entry.get("Name", ""), extensions)
        ]

    count = 0
    upload_ready_items = []
    for entry in entries:
        key = file_key(entry)
        if key in processed:
            continue

        relative_path = entry_relative_path(entry)
        name = relative_path.name
        local_source = incoming_dir / relative_path
        local_review = needs_review_dir / relative_path

        try:
            if local_mode:
                source_path = incoming_root / relative_path
                shutil.copy2(source_path, local_source)
            else:
                source_remote = remote_join(incoming_remote, relative_path.as_posix())
                copyto_local(source_remote, local_source)
            exports = process_file(local_source, processed_dir, config)
            posting_pack_exports = create_posting_pack(local_source, exports, processed_dir, config)

            for export_name, local_path in exports.items():
                if local_mode:
                    copy_file_to_folder(local_path, root_path / folders[export_name])
                else:
                    remote_folder = folders[export_name]
                    copyto_remote(local_path, remote_join(root, remote_folder, local_path.name))

            posting_pack_remote = None
            if posting_pack_exports:
                if local_mode:
                    posting_pack_remote_path = root_path / folders["posting_pack"] / source_output_stem(local_source)
                    posting_pack_remote_path.mkdir(parents=True, exist_ok=True)
                    for local_path in posting_pack_exports.values():
                        copy_file_to_folder(local_path, posting_pack_remote_path)
                    posting_pack_remote = str(posting_pack_remote_path)
                else:
                    posting_pack_remote = remote_join(root, folders["posting_pack"], source_output_stem(local_source))
                    for local_path in posting_pack_exports.values():
                        copyto_remote(local_path, remote_join(posting_pack_remote, local_path.name))

            if local_mode:
                archive_path = move_local_unique(source_path, root_path / folders["archive_originals"] / relative_path)
                archive_remote = str(archive_path)
            else:
                archive_remote = remote_join(root, folders["archive_originals"], relative_path.as_posix())
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
                local_review.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_source, local_review)
            if local_mode:
                source_path = incoming_root / relative_path
                if source_path.exists():
                    try:
                        move_local_unique(source_path, root_path / folders["needs_review"] / relative_path)
                    except Exception:
                        pass
            else:
                review_remote = remote_join(root, folders["needs_review"], relative_path.as_posix())
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

    upload_ready_items = pending_upload_ready_items(state, base_dir, config)
    if upload_ready_items:
        groups = split_ambiguous_groups_by_product(upload_ready_groups(upload_ready_items), config)
        for upload_ready_group in groups:
            try:
                pack_dir, upload_ready_files = create_upload_ready_pack(upload_ready_group, processed_dir, config)
                if pack_dir and upload_ready_files:
                    if local_mode:
                        upload_ready_remote_path = root_path / folders.get("upload_ready", "30_Upload_Ready") / pack_dir.name
                        sync_dir_to_local(pack_dir, upload_ready_remote_path)
                        upload_ready_remote = str(upload_ready_remote_path)
                    else:
                        upload_ready_remote = remote_join(root, folders.get("upload_ready", "30_Upload_Ready"), pack_dir.name)
                        sync_dir_to_remote(pack_dir, upload_ready_remote)
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
    parser.add_argument(
        "--convert-heic",
        action="store_true",
        help="Convert HEIC/HEIF files from the configured HEIC inbox to JPEGs for manual editing, then exit",
    )
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

        if args.convert_heic:
            counts = convert_remote_heic_inbox(config, base_dir)
            return 1 if counts["failed"] or counts["archive_failed"] else 0

        while True:
            count = process_once(config, base_dir)
            print(f"processed {count} file(s)", flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
