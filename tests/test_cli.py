from pathlib import Path

import bml_photo_pipeline.cli as cli
from bml_photo_pipeline.cli import split_ambiguous_groups_by_product, upload_ready_groups


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
