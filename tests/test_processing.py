from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw
import pytest

from bml_photo_pipeline.processing import create_posting_pack, create_upload_ready_pack, media_type, process_file


def test_media_type_detects_images_and_videos() -> None:
    assert media_type(Path("sample.jpg")) == "image"
    assert media_type(Path("sample.MOV")) == "video"
    assert media_type(Path("sample.txt")) is None


def test_process_file_creates_expected_exports(tmp_path: Path) -> None:
    source = tmp_path / "sample.jpg"
    image = Image.new("RGB", (1200, 900), (245, 245, 242))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((360, 240, 840, 660), radius=40, fill=(40, 120, 220))
    image.save(source)

    config = {
        "processing": {
            "trim_background": True,
            "trim_threshold": 20,
            "trim_padding_percent": 0.08,
            "autocontrast_cutoff": 1,
            "brightness": 1.05,
            "contrast": 1.08,
            "color": 1.02,
            "sharpness": 1.18,
            "remove_background": False,
            "background_color": [248, 248, 245],
        },
        "image_exports": {
            "etsy_main": {"width": 2000, "height": 2000},
            "etsy_gallery": {"width": 2000, "height": 1500},
            "social_4x5": {"width": 1600, "height": 2000},
            "social_9x16": {"width": 1440, "height": 2560},
        },
    }

    exports = process_file(source, tmp_path / "out", config)

    assert set(exports) == {"etsy_main", "etsy_gallery", "social_4x5", "social_9x16"}
    assert Image.open(exports["etsy_main"]).size == (2000, 2000)
    assert Image.open(exports["etsy_gallery"]).size == (2000, 1500)
    assert Image.open(exports["social_4x5"]).size == (1600, 2000)
    assert Image.open(exports["social_9x16"]).size == (1440, 2560)


def test_create_posting_pack_creates_contact_sheet_and_manifests(tmp_path: Path) -> None:
    source = tmp_path / "sample.jpg"
    source_image = Image.new("RGB", (600, 400), (245, 245, 242))
    source_image.save(source)

    export_dir = tmp_path / "exports"
    exports = {}
    for name, size in {
        "etsy_main": (400, 400),
        "social_4x5": (320, 400),
    }.items():
        target = export_dir / name / f"sample_{name}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (40, 120, 220)).save(target)
        exports[name] = target

    config = {
        "processing": {"background_color": [248, 248, 245]},
        "posting_pack": {
            "enabled": True,
            "contact_sheet_width": 1200,
            "thumbnail_width": 180,
            "thumbnail_height": 180,
        },
    }

    pack = create_posting_pack(source, exports, tmp_path / "out", config)

    assert set(pack) == {
        "posting_pack_contact_sheet",
        "posting_pack_manifest_csv",
        "posting_pack_manifest_html",
    }
    assert pack["posting_pack_contact_sheet"].exists()
    assert pack["posting_pack_manifest_csv"].read_text(encoding="utf-8").count("sample_") == 2
    html = pack["posting_pack_manifest_html"].read_text(encoding="utf-8")
    assert "Etsy listing photo #1" in html
    assert "Instagram/Facebook feed" in html


def test_create_upload_ready_pack_creates_ordered_assets_and_copy(tmp_path: Path) -> None:
    export_dir = tmp_path / "exports"
    exports = {}
    for name, suffix in {
        "etsy_main": ".jpg",
        "etsy_gallery": ".jpg",
        "social_4x5": ".jpg",
        "social_9x16": ".jpg",
        "etsy_video": ".mp4",
        "social_reels": ".mp4",
        "video_thumbnail": ".jpg",
    }.items():
        target = export_dir / name / f"sample_{name}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".jpg":
            Image.new("RGB", (400, 400), (40, 120, 220)).save(target)
        else:
            target.write_bytes(b"fake mp4")
        exports[name] = target

    config = {
        "upload_ready": {
            "enabled": True,
            "default_product_name": "Sample Product",
            "default_price": "12.00",
            "default_quantity": "3",
            "default_sku": "SAMPLE-001",
            "default_material": "PLA",
            "shop_name": "Bluegrass Maker Lab",
        }
    }

    pack_dir, files = create_upload_ready_pack(
        [{"source": tmp_path / "sample.jpg", "exports": exports}],
        tmp_path / "out",
        config,
    )

    assert pack_dir is not None
    assert (pack_dir / "UPLOAD_ME_FIRST.txt").exists()
    assert (pack_dir / "Etsy_Upload" / "01_MAIN_sample-product.jpg").exists()
    assert (pack_dir / "Etsy_Upload" / "listing-copy.txt").exists()
    assert (pack_dir / "Social_Upload" / "captions.txt").exists()
    assert (pack_dir / "Notes" / "upload-ready-manifest.csv").exists()
    listing = (pack_dir / "Etsy_Upload" / "etsy-step-by-step.md").read_text(encoding="utf-8")
    assert "Sample Product" in listing
    assert "SAMPLE-001" in listing
    assert files


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for video export")
def test_process_file_creates_video_exports(tmp_path: Path) -> None:
    source = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x480:rate=30",
            "-t",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
    )

    config = {
        "processing": {"background_color": [248, 248, 245]},
        "video_processing": {"max_duration_seconds": 1, "crf": 28, "preset": "ultrafast"},
        "video_exports": {
            "etsy_video": {"width": 320, "height": 320},
            "social_reels": {"width": 270, "height": 480},
        },
        "video_thumbnail": {"width": 320, "height": 320, "timestamp_seconds": 0},
    }

    exports = process_file(source, tmp_path / "out", config)

    assert set(exports) == {"etsy_video", "social_reels", "video_thumbnail"}
    assert exports["etsy_video"].suffix == ".mp4"
    assert exports["social_reels"].suffix == ".mp4"
    assert Image.open(exports["video_thumbnail"]).size == (320, 320)
