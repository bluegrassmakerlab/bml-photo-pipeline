from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw
import pytest

from bml_photo_pipeline.processing import media_type, process_file


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
