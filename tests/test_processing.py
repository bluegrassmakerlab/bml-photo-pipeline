from pathlib import Path
import shutil
import sqlite3
import subprocess

from PIL import Image, ImageDraw
import pytest

import bml_photo_pipeline.processing as processing
from bml_photo_pipeline.processing import (
    assess_photo_quality,
    autocontrast_luminance,
    background_luminance,
    content_bounds,
    crop_rotation_fill,
    create_posting_pack,
    create_upload_ready_pack,
    ExportSpec,
    fit_on_canvas,
    media_type,
    match_tracker_product,
    process_file,
    lift_neutral_background,
    straighten_subject,
    source_output_stem,
    subject_bounds,
    subject_luminance,
    vision_source_image,
    white_balance_background,
)


def assert_subject_centered(path: Path, threshold: int = 24, tolerance: int = 18) -> None:
    image = Image.open(path).convert("RGB")
    bounds = content_bounds(image, threshold)
    assert bounds is not None
    left, top, right, bottom = bounds
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    assert abs(center_x - image.width / 2) <= tolerance
    assert abs(center_y - image.height / 2) <= tolerance


def test_media_type_detects_images_and_videos() -> None:
    assert media_type(Path("sample.jpg")) == "image"
    assert media_type(Path("sample.MOV")) == "video"
    assert media_type(Path("sample.txt")) is None


def test_source_output_stem_includes_product_folder() -> None:
    assert source_output_stem(Path("work/incoming/Daisy Flower Soap Holder/IMG_0001.heic")) == (
        "daisy-flower-soap-holder_IMG_0001"
    )
    assert source_output_stem(Path("work/incoming/IMG_0001.heic")) == "IMG_0001"


def test_vision_source_image_uses_video_thumbnail(tmp_path: Path) -> None:
    thumbnail = tmp_path / "thumb.jpg"
    thumbnail.write_bytes(b"not really an image")

    source = vision_source_image(
        [
            {
                "source": tmp_path / "IMG_0001.MOV",
                "exports": {"video_thumbnail": thumbnail},
            }
        ]
    )

    assert source == thumbnail


def test_vision_product_prompt_includes_category_and_specificity_rules() -> None:
    prompt = processing.vision_product_prompt(
        [
            {"sku": "GSH-011", "name": "Goose Soap Holder", "category": "Soap Holder"},
            {"sku": "WD-004", "name": "White Duck", "category": "Farm Animal"},
        ]
    )

    assert "Category: Soap Holder" in prompt
    assert "prefer a Soap Holder product" in prompt
    assert "standalone animal product" in prompt


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
            "social_9x16": {"width": 1080, "height": 1920},
        },
    }

    exports = process_file(source, tmp_path / "out", config)

    assert set(exports) == {"etsy_main", "etsy_gallery", "social_4x5", "social_9x16"}
    assert Image.open(exports["etsy_main"]).size == (2000, 2000)
    assert Image.open(exports["etsy_gallery"]).size == (2000, 1500)
    assert Image.open(exports["social_4x5"]).size == (1600, 2000)
    assert Image.open(exports["social_9x16"]).size == (1080, 1920)


def test_process_file_preserves_manual_photo_edits_when_configured(tmp_path: Path) -> None:
    source = tmp_path / "edited.jpg"
    image = Image.new("RGB", (100, 100), (80, 110, 140))
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 25, 75, 75), fill=(160, 90, 40))
    image.save(source, quality=100, subsampling=0)

    config = {
        "processing": {
            "preserve_photo_edits": True,
            "white_balance": True,
            "trim_background": True,
            "autocontrast_luminance": True,
            "autocontrast_cutoff": 10,
            "brightness": 1.8,
            "contrast": 1.8,
            "color": 1.8,
            "sharpness": 2.0,
            "background_color": [248, 248, 245],
        },
        "image_exports": {
            "etsy_main": {"width": 100, "height": 100},
        },
    }

    exports = process_file(source, tmp_path / "out", config)

    source_pixel = Image.open(source).convert("RGB").getpixel((50, 50))
    exported_pixel = Image.open(exports["etsy_main"]).convert("RGB").getpixel((50, 50))
    assert exported_pixel == pytest.approx(source_pixel, abs=3)


def test_process_file_recenters_off_center_subject(tmp_path: Path) -> None:
    source = tmp_path / "off-center.jpg"
    image = Image.new("RGB", (1200, 900), (245, 245, 242))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((80, 310, 380, 610), radius=40, fill=(40, 120, 220))
    image.save(source)

    config = {
        "processing": {
            "trim_background": False,
            "center_subject": True,
            "subject_threshold": 20,
            "subject_padding_percent": 0.18,
            "autocontrast_cutoff": 1,
            "brightness": 1,
            "contrast": 1,
            "color": 1,
            "sharpness": 1,
            "remove_background": False,
            "background_color": [248, 248, 245],
            "white_balance": False,
            "autocontrast_luminance": False,
        },
        "image_exports": {
            "etsy_main": {"width": 1000, "height": 1000},
            "social_4x5": {"width": 800, "height": 1000},
        },
    }

    exports = process_file(source, tmp_path / "out", config)

    assert_subject_centered(exports["etsy_main"])
    assert_subject_centered(exports["social_4x5"])


def test_subject_bounds_ignore_neutral_table_area() -> None:
    image = Image.new("RGB", (1000, 1000), (248, 249, 244))
    draw = ImageDraw.Draw(image)
    draw.rectangle((160, 140, 840, 900), fill=(229, 231, 228))
    draw.ellipse((260, 680, 740, 840), fill=(186, 184, 180))
    draw.rounded_rectangle((290, 230, 710, 540), radius=80, fill=(218, 74, 118))
    draw.rectangle((360, 520, 640, 680), fill=(196, 54, 98))

    bounds = subject_bounds(image, threshold=18, saturation_threshold=45)

    assert bounds is not None
    left, top, right, bottom = bounds
    assert top < 260
    assert bottom < 720
    assert left > 250
    assert right < 750


def test_subject_bounds_fall_back_when_saturated_area_is_sparse() -> None:
    image = Image.new("RGB", (1000, 1000), (248, 249, 244))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((220, 120, 780, 880), radius=120, fill=(150, 132, 104))
    draw.ellipse((340, 300, 440, 400), fill=(245, 245, 238))
    draw.ellipse((560, 300, 660, 400), fill=(245, 245, 238))
    draw.ellipse((390, 340, 420, 370), fill=(12, 12, 12))
    draw.ellipse((610, 340, 640, 370), fill=(12, 12, 12))

    bounds = subject_bounds(image, threshold=18, saturation_threshold=45)

    assert bounds is not None
    left, top, right, bottom = bounds
    assert left < 260
    assert top < 160
    assert right > 740
    assert bottom > 840


def test_subject_bounds_include_neutral_body_with_saturated_accents() -> None:
    image = Image.new("RGB", (1000, 1000), (248, 249, 244))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((260, 170, 740, 810), radius=170, fill=(226, 228, 220))
    draw.ellipse((330, 130, 480, 300), fill=(235, 236, 230))
    draw.ellipse((520, 130, 670, 300), fill=(235, 236, 230))
    draw.polygon([(450, 330), (550, 330), (500, 420)], fill=(226, 86, 26))
    draw.ellipse((310, 760, 430, 860), fill=(214, 78, 24))
    draw.ellipse((570, 760, 690, 860), fill=(214, 78, 24))

    bounds = subject_bounds(image, threshold=18, saturation_threshold=45)

    assert bounds is not None
    left, top, right, bottom = bounds
    assert left < 330
    assert top < 200
    assert right > 700
    assert bottom > 780


def test_process_file_normalizes_dim_subject_and_quality_passes(tmp_path: Path) -> None:
    source = tmp_path / "dim-off-center.jpg"
    image = Image.new("RGB", (1200, 900), (245, 245, 242))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((90, 320, 420, 650), radius=50, fill=(34, 72, 120))
    image.save(source)

    config = {
        "processing": {
            "trim_background": False,
            "center_subject": True,
            "subject_threshold": 20,
            "subject_padding_percent": 0.18,
            "normalize_subject_brightness": True,
            "target_subject_luminance": 165,
            "max_subject_brightness_adjustment": 0.45,
            "autocontrast_cutoff": 1,
            "brightness": 1,
            "contrast": 1,
            "color": 1,
            "sharpness": 1,
            "remove_background": False,
            "background_color": [248, 248, 245],
            "white_balance": False,
            "autocontrast_luminance": False,
            "auto_rotate": False,
        },
        "image_exports": {"etsy_main": {"width": 1000, "height": 1000}},
    }

    exports = process_file(source, tmp_path / "out", config)
    output = Image.open(exports["etsy_main"]).convert("RGB")
    quality = assess_photo_quality(output, threshold=20)
    bounds = subject_bounds(output, threshold=20)

    assert quality.subject_found
    assert abs(quality.center_offset_x) <= 0.035
    assert abs(quality.center_offset_y) <= 0.035
    assert bounds is not None
    assert subject_luminance(output, bounds) >= 95


def test_fit_on_canvas_targets_subject_height() -> None:
    image = Image.new("RGB", (1600, 1200), (220, 224, 222))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((690, 520, 910, 780), radius=40, fill=(220, 92, 34))

    output = fit_on_canvas(
        image,
        ExportSpec(800, 1000),
        (220, 224, 222),
        center_subject=True,
        subject_threshold=20,
        subject_padding_percent=0.12,
        subject_saturation_threshold=45,
        target_subject_height_percent=0.7,
        max_subject_width_percent=0.86,
    )
    bounds = subject_bounds(output, threshold=20)

    assert bounds is not None
    assert 0.62 <= ((bounds[3] - bounds[1]) / output.height) <= 0.78


def test_lift_neutral_background_protects_colored_subject() -> None:
    image = Image.new("RGB", (600, 500), (188, 192, 190))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((210, 150, 390, 360), radius=40, fill=(210, 82, 32))
    before_subject = image.getpixel((300, 250))

    lifted = lift_neutral_background(image, 18, 45, target_luminance=236, max_lift=44)

    assert background_luminance(lifted) > background_luminance(image) + 18
    after_subject = lifted.getpixel((300, 250))
    assert max(abs(after_subject[index] - before_subject[index]) for index in range(3)) <= 8


def test_polish_straightens_small_camera_tilt(tmp_path: Path) -> None:
    source = tmp_path / "tilted.jpg"
    image = Image.new("RGB", (900, 900), (245, 245, 242))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((220, 340, 680, 560), radius=40, fill=(44, 124, 190))
    image = image.rotate(4, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(245, 245, 242))
    image.save(source)

    config = {
        "processing": {
            "trim_background": False,
            "center_subject": True,
            "subject_threshold": 20,
            "subject_padding_percent": 0.18,
            "normalize_subject_brightness": False,
            "autocontrast_cutoff": 1,
            "brightness": 1,
            "contrast": 1,
            "color": 1,
            "sharpness": 1,
            "remove_background": False,
            "background_color": [245, 245, 242],
            "white_balance": False,
            "autocontrast_luminance": False,
            "auto_rotate": True,
            "auto_rotate_max_degrees": 6,
        },
        "image_exports": {"etsy_main": {"width": 1000, "height": 1000}},
    }

    exports = process_file(source, tmp_path / "out", config)
    quality = assess_photo_quality(Image.open(exports["etsy_main"]).convert("RGB"), threshold=20)

    assert quality.tilt_degrees is not None
    assert abs(quality.tilt_degrees) < 2.5


def test_straighten_crops_rotation_fill_corners() -> None:
    image = Image.new("RGB", (600, 420), (245, 245, 242))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((150, 150, 450, 270), radius=24, fill=(44, 124, 190))
    tilted = image.rotate(4, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 0, 255))

    corrected = crop_rotation_fill(tilted, image.size, 4)

    corners = [
        corrected.getpixel((0, 0)),
        corrected.getpixel((corrected.width - 1, 0)),
        corrected.getpixel((0, corrected.height - 1)),
        corrected.getpixel((corrected.width - 1, corrected.height - 1)),
    ]
    assert all(pixel != (255, 0, 255) for pixel in corners)


def test_polish_helpers_preserve_neutral_background() -> None:
    image = Image.new("RGB", (200, 160), (218, 224, 232))
    draw = ImageDraw.Draw(image)
    draw.rectangle((70, 50, 130, 110), fill=(238, 236, 230))

    balanced = white_balance_background(image, 0.85)
    contrasted = autocontrast_luminance(balanced, 0.35)
    corner = contrasted.getpixel((10, 10))

    assert max(corner) - min(corner) <= 8


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
    assert (pack_dir / "Buffer_Upload" / "01_FEED_POST_IMAGE_buffer-safe-4x5.jpg").exists()
    assert (pack_dir / "Buffer_Upload" / "02_REEL_TIKTOK_SHORT_video.mp4").exists()
    assert (pack_dir / "Buffer_Upload" / "buffer-instructions.txt").exists()
    assert (pack_dir / "Buffer_Upload" / "buffer-post-draft.json").exists()
    assert (pack_dir / "Buffer_Upload" / "buffer-queue.csv").exists()
    assert Image.open(pack_dir / "Social_Upload" / "story-tiktok-photo.jpg").size == (400, 400)
    assert Image.open(pack_dir / "Buffer_Upload" / "03_STORY_ONLY_IMAGE_9x16.jpg").size == (400, 400)
    assert (pack_dir / "Notes" / "photo-consistency-report.txt").exists()
    assert (pack_dir / "Notes" / "upload-ready-manifest.csv").exists()
    listing = (pack_dir / "Etsy_Upload" / "etsy-step-by-step.md").read_text(encoding="utf-8")
    assert "Sample Product" in listing
    assert "SAMPLE-001" in listing
    captions = (pack_dir / "Social_Upload" / "captions.txt").read_text(encoding="utf-8")
    assert "Alternate captions:" in captions
    assert "TikTok/Reels hook:" in captions
    buffer_queue = (pack_dir / "Buffer_Upload" / "buffer-queue.csv").read_text(encoding="utf-8")
    assert "instagram;facebook;tiktok" in buffer_queue
    assert "Sample Product" in buffer_queue
    assert "Photo consistency QA" in (pack_dir / "Notes" / "photo-consistency-report.txt").read_text(encoding="utf-8")
    assert files


def test_create_upload_ready_pack_caps_vertical_buffer_images(tmp_path: Path) -> None:
    export_dir = tmp_path / "exports"
    social_9x16 = export_dir / "social_9x16" / "oversized.jpg"
    social_9x16.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1440, 2560), (40, 120, 220)).save(social_9x16)

    etsy_main = export_dir / "etsy_main" / "main.jpg"
    etsy_main.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2000, 2000), (40, 120, 220)).save(etsy_main)

    config = {
        "upload_ready": {
            "enabled": True,
            "default_product_name": "Sample Product",
            "shop_name": "Bluegrass Maker Lab",
        }
    }

    pack_dir, _files = create_upload_ready_pack(
        [{"source": tmp_path / "sample.jpg", "exports": {"etsy_main": etsy_main, "social_9x16": social_9x16}}],
        tmp_path / "out",
        config,
    )

    assert Image.open(pack_dir / "Social_Upload" / "story-tiktok-photo.jpg").size == (1080, 1920)
    assert Image.open(pack_dir / "Buffer_Upload" / "03_STORY_ONLY_IMAGE_9x16.jpg").size == (1080, 1920)


def test_create_upload_ready_pack_skips_ambiguous_large_groups(tmp_path: Path) -> None:
    media_items = []
    for index in range(5):
        export = tmp_path / f"sample_{index}_etsy_main.jpg"
        Image.new("RGB", (400, 400), (40, 120, 220)).save(export)
        media_items.append({"source": tmp_path / f"IMG_{index:04d}.jpg", "exports": {"etsy_main": export}})

    config = {
        "upload_ready": {
            "enabled": True,
            "max_auto_images": 4,
            "max_auto_videos": 1,
        }
    }

    with pytest.raises(ValueError, match="ambiguous upload-ready group"):
        create_upload_ready_pack(media_items, tmp_path / "out", config)


def test_create_upload_ready_pack_allows_larger_matched_groups(tmp_path: Path) -> None:
    tracker_db = tmp_path / "tracker.db"
    with sqlite3.connect(tracker_db) as conn:
        conn.execute(
            """
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                sku TEXT,
                event_price REAL,
                quantity_in_stock INTEGER,
                discontinued INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            "INSERT INTO products (id, name, sku, event_price, quantity_in_stock, discontinued) VALUES (?, ?, ?, ?, ?, ?)",
            (100, "Chicken Soap Holder", "CSH-002", 25.0, 4, 0),
        )

    media_items = []
    for index in range(5):
        export = tmp_path / f"sample_{index}_etsy_main.jpg"
        Image.new("RGB", (400, 400), (40, 120, 220)).save(export)
        media_items.append(
            {
                "source": tmp_path / f"IMG_{index:04d}.jpg",
                "exports": {"etsy_main": export},
                "product_hint": "CSH-002",
            }
        )

    config = {
        "upload_ready": {
            "enabled": True,
            "max_auto_images": 4,
            "tracker_db_path": str(tracker_db),
        }
    }

    pack_dir, files = create_upload_ready_pack(media_items, tmp_path / "out", config)

    assert pack_dir is not None
    assert pack_dir.name == "chicken-soap-holder"
    assert files


def test_create_upload_ready_pack_uses_tracker_product_match(tmp_path: Path) -> None:
    tracker_db = tmp_path / "tracker.db"
    with sqlite3.connect(tracker_db) as conn:
        conn.execute(
            """
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                sku TEXT,
                event_price REAL,
                quantity_in_stock INTEGER,
                discontinued INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            "INSERT INTO products (id, name, sku, event_price, quantity_in_stock, discontinued) VALUES (?, ?, ?, ?, ?, ?)",
            (99, "Duck Soap Holder", "DSH-002", 25.0, 3, 0),
        )

    export = tmp_path / "exports" / "duck_etsy_main.jpg"
    export.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (400, 400), (40, 120, 220)).save(export)
    config = {
        "upload_ready": {
            "enabled": True,
            "require_product_match": True,
            "tracker_db_path": str(tracker_db),
        }
    }

    pack_dir, _files = create_upload_ready_pack(
        [{"source": tmp_path / "Duck Soap Holder" / "IMG_0001.jpg", "exports": {"etsy_main": export}}],
        tmp_path / "out",
        config,
    )

    assert pack_dir is not None
    assert pack_dir.name == "duck-soap-holder"
    listing = (pack_dir / "Etsy_Upload" / "etsy-step-by-step.md").read_text(encoding="utf-8")
    assert "SKU: DSH-002" in listing
    assert "Recommended price: 25.00" in listing
    assert "Quantity: 3" in listing
    assert "Foaming Hand Soap Bottle Holder" in listing
    assert "Bathroom sink decor" in listing
    assert "foaming hand soap" in listing.lower()
    captions = (pack_dir / "Social_Upload" / "captions.txt").read_text(encoding="utf-8")
    assert "sink" in captions.lower()
    assert "#SoapHolder" in captions
    assert "bar soap" not in captions.lower()


def test_tracker_exact_name_match_beats_generic_partial_match(tmp_path: Path) -> None:
    tracker_db = tmp_path / "tracker.db"
    with sqlite3.connect(tracker_db) as conn:
        conn.execute(
            """
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                sku TEXT,
                event_price REAL,
                quantity_in_stock INTEGER,
                discontinued INTEGER DEFAULT 0
            )
            """
        )
        conn.executemany(
            "INSERT INTO products (id, name, sku, event_price, quantity_in_stock, discontinued) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (13, "Bigfoot", "B-004", 0.0, 2, 0),
                (179, "Bigfoot Soap Holder", "BSH-003", 25.0, 1, 0),
            ],
        )

    product = match_tracker_product("Bigfoot Soap Holder", {"tracker_db_path": str(tracker_db)})

    assert product is not None
    assert product["sku"] == "BSH-003"


def test_soap_holder_social_copy_varies_by_product() -> None:
    from bml_photo_pipeline.processing import create_social_text

    base_settings = {
        "price": "25.00",
        "quantity": "3",
        "sku": "",
        "material": "3D printed plastic / PLA",
        "shop_name": "Bluegrass Maker Lab",
        "category": "Soap Holder",
    }
    duck = create_social_text({**base_settings, "product_name": "Duck Soap Holder"}, True)
    hedgehog = create_social_text({**base_settings, "product_name": "Hedgehog Soap Holder"}, True)

    assert "duck" in duck.lower()
    assert "hedgehog" in hedgehog.lower()
    assert "sink" in duck.lower()
    assert "sink" in hedgehog.lower()
    assert duck != hedgehog

    duck_variants = {
        create_social_text({**base_settings, "product_name": "Duck Soap Holder", "copy_seed": f"duck-{index}"}, True)
        for index in range(6)
    }
    assert len(duck_variants) > 1


def test_create_upload_ready_pack_requires_tracker_match_when_configured(tmp_path: Path) -> None:
    tracker_db = tmp_path / "tracker.db"
    with sqlite3.connect(tracker_db) as conn:
        conn.execute(
            """
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                sku TEXT,
                event_price REAL,
                quantity_in_stock INTEGER,
                discontinued INTEGER DEFAULT 0
            )
            """
        )

    export = tmp_path / "exports" / "unknown_etsy_main.jpg"
    export.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (400, 400), (40, 120, 220)).save(export)
    config = {
        "upload_ready": {
            "enabled": True,
            "require_product_match": True,
            "tracker_db_path": str(tracker_db),
        }
    }

    with pytest.raises(ValueError, match="no confident Tracker product match"):
        create_upload_ready_pack(
            [{"source": tmp_path / "Mystery Item" / "IMG_0001.jpg", "exports": {"etsy_main": export}}],
            tmp_path / "out",
            config,
        )


def test_create_upload_ready_pack_uses_vision_when_folder_match_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracker_db = tmp_path / "tracker.db"
    with sqlite3.connect(tracker_db) as conn:
        conn.execute(
            """
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                sku TEXT,
                event_price REAL,
                quantity_in_stock INTEGER,
                discontinued INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            "INSERT INTO products (id, name, sku, event_price, quantity_in_stock, discontinued) VALUES (?, ?, ?, ?, ?, ?)",
            (100, "Duck Soap Holder", "DSH-002", 25.0, 3, 0),
        )

    export = tmp_path / "exports" / "IMG_0001_etsy_main.jpg"
    export.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (400, 400), (40, 120, 220)).save(export)

    def fake_run(command, **_kwargs):
        assert command[:5] == ["openclaw", "infer", "model", "run", "--gateway"]

        class Result:
            returncode = 0
            stdout = (
                '{"ok": true, "outputs": ['
                '{"text": "{\\"product_name\\":\\"Duck Soap Holder\\",\\"sku\\":\\"DSH-002\\",\\"confidence\\":0.96}"}'
                "]}"
            )

        return Result()

    monkeypatch.setattr(processing.subprocess, "run", fake_run)
    config = {
        "upload_ready": {
            "enabled": True,
            "require_product_match": True,
            "tracker_db_path": str(tracker_db),
            "vision_match_enabled": True,
        }
    }

    pack_dir, _files = create_upload_ready_pack(
        [{"source": tmp_path / "incoming" / "IMG_0001.jpg", "exports": {"etsy_main": export}}],
        tmp_path / "out",
        config,
    )

    assert pack_dir is not None
    assert pack_dir.name == "duck-soap-holder"


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
