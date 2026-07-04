from pathlib import Path

import bml_photo_pipeline.cli as cli
from bml_photo_pipeline.cli import (
    convert_remote_heic_inbox,
    jpeg_relative_path,
    pending_upload_ready_items,
    safe_product_folder_name,
    split_ambiguous_groups_by_product,
    sync_tracker_product_incoming_folders,
    upload_ready_groups,
)


def test_jpeg_relative_path_changes_only_extension() -> None:
    assert jpeg_relative_path(Path("Bigfoot Soap Holder/IMG_0001.HEIC")) == Path("Bigfoot Soap Holder/IMG_0001.jpg")


def test_convert_remote_heic_inbox_skips_existing_jpegs(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_list_json(remote_path: str, *, recursive: bool = False):
        if remote_path.endswith("00_HEIC_To_Convert"):
            return [{"Path": "Bigfoot Soap Holder/IMG_0001.HEIC", "Name": "IMG_0001.HEIC", "IsDir": False}]
        if remote_path.endswith("05_JPEG_For_Editing"):
            return [{"Path": "Bigfoot Soap Holder/IMG_0001.jpg", "Name": "IMG_0001.jpg", "IsDir": False}]
        return []

    monkeypatch.setattr(cli, "list_json", fake_list_json)
    monkeypatch.setattr(cli, "copyto_local", lambda *args: calls.append(("copyto_local", args)))
    monkeypatch.setattr(cli, "copyto_remote", lambda *args: calls.append(("copyto_remote", args)))

    counts = convert_remote_heic_inbox(
        {
            "remote_root": "onedrive:Bluegrass Maker Lab/Product Photo Pipeline",
            "folders": {
                "heic_inbox": "00_HEIC_To_Convert",
                "jpeg_for_editing": "05_JPEG_For_Editing",
            },
            "local_work_dir": "work",
        },
        tmp_path,
    )

    assert counts == {"converted": 0, "skipped": 1, "failed": 0}
    assert calls == []


def test_upload_ready_groups_end_at_videos() -> None:
    items = [
        {"source": Path("IMG_0001.jpeg")},
        {"source": Path("IMG_0002.jpeg")},
        {"source": Path("IMG_0003.MOV")},
        {"source": Path("IMG_0004.jpeg")},
        {"source": Path("IMG_0005.MOV")},
    ]

    groups = upload_ready_groups(items)

    assert [[item["source"].name for item in group] for group in groups] == [
        ["IMG_0001.jpeg", "IMG_0002.jpeg", "IMG_0003.MOV"],
        ["IMG_0004.jpeg", "IMG_0005.MOV"],
    ]


def test_upload_ready_groups_attach_leading_video_to_next_images() -> None:
    items = [
        {"source": Path("IMG_0001.MOV")},
        {"source": Path("IMG_0002.jpeg")},
        {"source": Path("IMG_0003.jpeg")},
    ]

    groups = upload_ready_groups(items)

    assert [[item["source"].name for item in group] for group in groups] == [
        ["IMG_0001.MOV", "IMG_0002.jpeg", "IMG_0003.jpeg"],
    ]


def test_upload_ready_groups_split_extra_leading_videos() -> None:
    items = [
        {"source": Path("IMG_0001.MOV")},
        {"source": Path("IMG_0002.MOV")},
        {"source": Path("IMG_0003.jpeg")},
        {"source": Path("IMG_0004.jpeg")},
    ]

    groups = upload_ready_groups(items)

    assert [[item["source"].name for item in group] for group in groups] == [
        ["IMG_0001.MOV"],
        ["IMG_0002.MOV", "IMG_0003.jpeg", "IMG_0004.jpeg"],
    ]


def test_upload_ready_groups_keep_product_folders_separate() -> None:
    items = [
        {"source": Path("Duck Soap Holder/IMG_0001.jpeg")},
        {"source": Path("Duck Soap Holder/IMG_0002.MOV")},
        {"source": Path("Chicken Soap Holder/IMG_0003.jpeg")},
        {"source": Path("Chicken Soap Holder/IMG_0004.MOV")},
    ]

    groups = upload_ready_groups(items)

    assert [[item["source"].as_posix() for item in group] for group in groups] == [
        ["Chicken Soap Holder/IMG_0003.jpeg", "Chicken Soap Holder/IMG_0004.MOV"],
        ["Duck Soap Holder/IMG_0001.jpeg", "Duck Soap Holder/IMG_0002.MOV"],
    ]


def test_pending_upload_ready_items_resume_from_processed_state(tmp_path: Path) -> None:
    export = tmp_path / "work" / "processed" / "etsy_main" / "IMG_0001_etsy_main.jpg"
    export.parent.mkdir(parents=True)
    export.write_bytes(b"image")
    state = {
        "processed": {
            "Product A/IMG_0001.jpeg|1|now": {
                "name": "IMG_0001.jpeg",
                "archive_remote": "onedrive:Bluegrass Maker Lab/Product Photo Pipeline/90_Archive/Originals/Product A/IMG_0001.jpeg",
                "exports": {"etsy_main": str(export)},
            },
            "Product A/IMG_0002.jpeg|1|now": {
                "name": "IMG_0002.jpeg",
                "archive_remote": "onedrive:Bluegrass Maker Lab/Product Photo Pipeline/90_Archive/Originals/Product A/IMG_0002.jpeg",
                "exports": {"etsy_main": str(tmp_path / "missing.jpg")},
            },
            "Product A/IMG_0003.jpeg|1|now": {
                "name": "IMG_0003.jpeg",
                "error": "bad photo",
            },
            "Product A/IMG_0004.jpeg|1|now": {
                "name": "IMG_0004.jpeg",
                "archive_remote": "onedrive:Bluegrass Maker Lab/Product Photo Pipeline/90_Archive/Originals/Product A/IMG_0004.jpeg",
                "exports": {"etsy_main": str(export)},
                "upload_ready_skipped": True,
            },
        },
        "upload_ready": {},
    }
    config = {
        "remote_root": "onedrive:Bluegrass Maker Lab/Product Photo Pipeline",
        "folders": {"archive_originals": "90_Archive/Originals"},
        "local_work_dir": "work",
    }

    items = pending_upload_ready_items(state, tmp_path, config)

    assert len(items) == 1
    assert items[0]["source"] == tmp_path / "work" / "incoming" / "Product A" / "IMG_0001.jpeg"
    assert items[0]["exports"]["etsy_main"] == export


def test_safe_product_folder_name_removes_path_separators() -> None:
    assert safe_product_folder_name("  Dragon / Egg \\ Set  ") == "Dragon - Egg - Set"
    assert safe_product_folder_name("Trailing. ") == "Trailing"


def test_sync_tracker_product_incoming_folders_creates_missing_dirs(monkeypatch) -> None:
    created: list[str] = []
    state: dict = {}

    def fake_list_json(remote_path, *, recursive=False):
        assert remote_path == "onedrive:Root/00_Incoming"
        assert recursive is False
        return [{"Name": "Existing Product", "Path": "Existing Product", "IsDir": True}]

    def fake_mkdir(remote_path):
        created.append(remote_path)

    def fake_load_tracker_products(settings):
        assert settings["tracker_db_path"] == "/tmp/tracker.db"
        return [
            {"id": 1, "name": "Existing Product"},
            {"id": 2, "name": "New Product"},
            {"id": 3, "name": "Dragon / Egg"},
        ]

    monkeypatch.setattr(cli, "list_json", fake_list_json)
    monkeypatch.setattr(cli, "mkdir", fake_mkdir)
    monkeypatch.setattr(cli, "load_tracker_products", fake_load_tracker_products)

    synced = sync_tracker_product_incoming_folders(
        {
            "remote_root": "onedrive:Root",
            "folders": {"incoming": "00_Incoming"},
            "upload_ready": {
                "sync_tracker_product_folders": True,
                "tracker_db_path": "/tmp/tracker.db",
            },
        },
        state,
    )

    assert synced == {"created": ["New Product", "Dragon - Egg"], "renamed": [], "conflicts": []}
    assert created == [
        "onedrive:Root/00_Incoming/New Product",
        "onedrive:Root/00_Incoming/Dragon - Egg",
    ]
    assert state["tracker_product_folders"]["2"]["folder"] == "New Product"


def test_sync_tracker_product_incoming_folders_renames_changed_product(monkeypatch) -> None:
    created: list[str] = []
    moved: list[tuple[str, str]] = []
    state = {"tracker_product_folders": {"7": {"folder": "Old Product Name"}}}

    def fake_list_json(_remote_path, *, recursive=False):
        assert recursive is False
        return [{"Name": "Old Product Name", "Path": "Old Product Name", "IsDir": True}]

    def fake_load_tracker_products(_settings):
        return [{"id": 7, "name": "New Product Name"}]

    monkeypatch.setattr(cli, "list_json", fake_list_json)
    monkeypatch.setattr(cli, "mkdir", lambda remote_path: created.append(remote_path))
    monkeypatch.setattr(cli, "moveto_remote", lambda source, dest: moved.append((source, dest)))
    monkeypatch.setattr(cli, "load_tracker_products", fake_load_tracker_products)

    synced = sync_tracker_product_incoming_folders(
        {
            "remote_root": "onedrive:Root",
            "folders": {"incoming": "00_Incoming"},
            "upload_ready": {"sync_tracker_product_folders": True},
        },
        state,
    )

    assert synced == {
        "created": [],
        "renamed": ["Old Product Name -> New Product Name"],
        "conflicts": [],
    }
    assert moved == [
        (
            "onedrive:Root/00_Incoming/Old Product Name",
            "onedrive:Root/00_Incoming/New Product Name",
        )
    ]
    assert created == []
    assert state["tracker_product_folders"]["7"]["folder"] == "New Product Name"


def test_sync_tracker_product_incoming_folders_reports_rename_conflict(monkeypatch) -> None:
    state = {"tracker_product_folders": {"7": {"folder": "Old Product Name"}}}

    monkeypatch.setattr(
        cli,
        "list_json",
        lambda _remote_path, recursive=False: [
            {"Name": "Old Product Name", "Path": "Old Product Name", "IsDir": True},
            {"Name": "New Product Name", "Path": "New Product Name", "IsDir": True},
        ],
    )
    monkeypatch.setattr(cli, "load_tracker_products", lambda _settings: [{"id": 7, "name": "New Product Name"}])

    synced = sync_tracker_product_incoming_folders(
        {
            "remote_root": "onedrive:Root",
            "folders": {"incoming": "00_Incoming"},
            "upload_ready": {"sync_tracker_product_folders": True},
        },
        state,
    )

    assert synced == {
        "created": [],
        "renamed": [],
        "conflicts": ["Old Product Name -> New Product Name"],
    }


def test_split_ambiguous_groups_by_product_uses_vision_resolved_hints(monkeypatch) -> None:
    group = [
        {"source": Path("IMG_0001.jpeg")},
        {"source": Path("IMG_0002.jpeg")},
        {"source": Path("IMG_0003.jpeg")},
        {"source": Path("IMG_0004.jpeg")},
        {"source": Path("IMG_0005.jpeg")},
        {"source": Path("IMG_0006.MOV")},
    ]

    def fake_settings(_config, media_items=None):
        if not media_items:
            return {"max_auto_images": 4, "max_auto_videos": 1}
        name = media_items[0]["source"].name
        if name in {"IMG_0001.jpeg", "IMG_0002.jpeg"}:
            return {"tracker_product_id": 1, "sku": "DSH-002", "product_name": "Duck Soap Holder"}
        return {"tracker_product_id": 2, "sku": "CSH-002", "product_name": "Chicken Soap Holder"}

    monkeypatch.setattr(cli, "upload_ready_settings", fake_settings)

    groups = split_ambiguous_groups_by_product([group], {})

    assert [[item["source"].name for item in split] for split in groups] == [
        ["IMG_0001.jpeg", "IMG_0002.jpeg"],
        ["IMG_0003.jpeg", "IMG_0004.jpeg", "IMG_0005.jpeg", "IMG_0006.MOV"],
    ]
    assert groups[0][0]["product_hint"] == "DSH-002"
    assert groups[1][0]["product_hint"] == "CSH-002"


def test_split_ambiguous_groups_chunks_loose_incoming_without_item_vision(monkeypatch) -> None:
    group = [
        {"source": Path("/tmp/work/incoming/IMG_0001.jpeg")},
        {"source": Path("/tmp/work/incoming/IMG_0002.jpeg")},
        {"source": Path("/tmp/work/incoming/IMG_0003.jpeg")},
        {"source": Path("/tmp/work/incoming/IMG_0004.jpeg")},
        {"source": Path("/tmp/work/incoming/IMG_0005.jpeg")},
        {"source": Path("/tmp/work/incoming/IMG_0006.jpeg")},
    ]

    def fake_settings(_config, media_items=None):
        assert media_items is None
        return {"max_auto_images": 4, "max_auto_videos": 1}

    monkeypatch.setattr(cli, "upload_ready_settings", fake_settings)

    groups = split_ambiguous_groups_by_product([group], {})

    assert [[item["source"].name for item in split] for split in groups] == [
        ["IMG_0001.jpeg", "IMG_0002.jpeg", "IMG_0003.jpeg", "IMG_0004.jpeg"],
        ["IMG_0005.jpeg", "IMG_0006.jpeg"],
    ]


def test_split_ambiguous_groups_by_product_keeps_leading_video_with_first_product(monkeypatch) -> None:
    group = [
        {"source": Path("IMG_0001.MOV")},
        {"source": Path("IMG_0002.jpeg")},
        {"source": Path("IMG_0003.jpeg")},
        {"source": Path("IMG_0004.jpeg")},
        {"source": Path("IMG_0005.jpeg")},
        {"source": Path("IMG_0006.jpeg")},
        {"source": Path("IMG_0007.MOV")},
    ]

    def fake_settings(_config, media_items=None):
        if not media_items:
            return {"max_auto_images": 4, "max_auto_videos": 1}
        name = media_items[0]["source"].name
        if name in {"IMG_0002.jpeg", "IMG_0003.jpeg"}:
            return {"tracker_product_id": 1, "sku": "DSH-002", "product_name": "Duck Soap Holder"}
        return {"tracker_product_id": 2, "sku": "CSH-002", "product_name": "Chicken Soap Holder"}

    monkeypatch.setattr(cli, "upload_ready_settings", fake_settings)

    groups = split_ambiguous_groups_by_product([group], {})

    assert [[item["source"].name for item in split] for split in groups] == [
        ["IMG_0001.MOV", "IMG_0002.jpeg", "IMG_0003.jpeg"],
        ["IMG_0004.jpeg", "IMG_0005.jpeg", "IMG_0006.jpeg", "IMG_0007.MOV"],
    ]
    assert groups[0][1]["product_hint"] == "DSH-002"
    assert groups[1][0]["product_hint"] == "CSH-002"


def test_split_ambiguous_groups_by_product_splits_extra_videos(monkeypatch) -> None:
    group = [
        {"source": Path("IMG_0001.MOV")},
        {"source": Path("IMG_0002.MOV")},
        {"source": Path("IMG_0003.jpeg")},
        {"source": Path("IMG_0004.jpeg")},
    ]

    def fake_settings(_config, media_items=None):
        if not media_items:
            return {"max_auto_images": 4, "max_auto_videos": 1}
        return {"tracker_product_id": 1, "sku": "GSH-011", "product_name": "Goose Soap Holder"}

    monkeypatch.setattr(cli, "upload_ready_settings", fake_settings)

    groups = split_ambiguous_groups_by_product([group], {})

    assert [[item["source"].name for item in split] for split in groups] == [
        ["IMG_0001.MOV"],
        ["IMG_0002.MOV", "IMG_0003.jpeg", "IMG_0004.jpeg"],
    ]
