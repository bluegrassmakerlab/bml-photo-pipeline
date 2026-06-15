from pathlib import Path

from bml_photo_pipeline.cli import upload_ready_groups


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
